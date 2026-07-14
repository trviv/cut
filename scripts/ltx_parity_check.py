#!/usr/bin/env python3
import torch
import numpy as np
import os
import struct
import argparse

def read_embedding_file(path):
    with open(path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0x4C545845:
            raise ValueError("Invalid embedding file magic")
        n_tokens = struct.unpack('<I', f.read(4))[0]
        dim = struct.unpack('<I', f.read(4))[0]
        data = np.frombuffer(f.read(n_tokens * dim * 4), dtype=np.float32)
        return data.reshape(n_tokens, dim)

def write_latent_file(path, data, f, h, w, c):
    with open(path, 'wb') as f_out:
        f_out.write(struct.pack('<I', 0x4C54584C))
        f_out.write(struct.pack('<I', f))
        f_out.write(struct.pack('<I', h))
        f_out.write(struct.pack('<I', w))
        f_out.write(struct.pack('<I', c))
        f_out.write(data.tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--prompt-emb', required=True)
    parser.add_argument('--dims', default='2,8,8')
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--workdir', default='/tmp')
    args = parser.parse_args()

    F, H, W = map(int, args.dims.split(','))
    S = F * H * W

    # Set seed and create random latent
    torch.manual_seed(args.seed)
    x0 = torch.randn(1, S, 128, dtype=torch.float32)

    # Write x0 to parity_x0.bin
    os.makedirs(args.workdir, exist_ok=True)
    x0_np = x0.numpy()[0]
    write_latent_file(os.path.join(args.workdir, 'parity_x0.bin'), x0_np, F, H, W, 128)

    # Load prompt embeddings
    emb_data = read_embedding_file(args.prompt_emb)
    emb = torch.from_numpy(emb_data).unsqueeze(0)  # [1, n_tokens, 4096]

    # Load model and run forward pass
    from diffusers import LTXVideoTransformer3DModel
    model = LTXVideoTransformer3DModel.from_pretrained(
        os.path.join(args.model_dir, "transformer"),
        torch_dtype=torch.float32
    )
    model.eval()

    timestep = torch.tensor([1000.0])
    mask = torch.ones(1, emb.shape[1])

    with torch.no_grad():
        pred = model(
            hidden_states=x0,
            encoder_hidden_states=emb,
            timestep=timestep,
            encoder_attention_mask=mask,
            num_frames=F,
            height=H,
            width=W,
            rope_interpolation_scale=(1.0 / (25.0 / 8.0), 32, 32),
            return_dict=False
        )[0][0]  # [S, 128]

    # Print C++ command
    print(f"ltx_example {args.model_dir} --prompt-emb {args.prompt_emb} "
          f"--init-latents {args.workdir}/parity_x0.bin "
          f"--frames {(F-1)*8+1} --height {H*32} --width {W*32} "
          f"--steps 1 --guidance 1 --seed 0 --out {args.workdir}/parity_x1.bin")

    # If output exists, compare
    x1_path = os.path.join(args.workdir, 'parity_x1.bin')
    if os.path.exists(x1_path):
        with open(x1_path, 'rb') as f:
            magic = struct.unpack('<I', f.read(4))[0]
            if magic != 0x4C54584C:
                raise ValueError("Invalid latent file magic")
            f_read, h_read, w_read, c_read = struct.unpack('<IIII', f.read(16))
            x1_data = np.frombuffer(f.read(f_read * h_read * w_read * c_read * 4), dtype=np.float32)
            x1 = x1_data.reshape(f_read * h_read * w_read, c_read)

        # Euler update: x1 = x0 - pred * sigma
        pred_cpp = x0.numpy()[0] - x1

        max_abs_diff = np.max(np.abs(pred.numpy() - pred_cpp))
        mean_abs_diff = np.mean(np.abs(pred.numpy() - pred_cpp))
        denom = np.maximum(np.abs(pred.numpy()), 1e-3)
        rel_error = np.max(np.abs(pred.numpy() - pred_cpp) / denom)

        print(f"Max absolute difference: {max_abs_diff}")
        print(f"Mean absolute difference: {mean_abs_diff}")
        print(f"Relative error: {rel_error}")

        print("First 5 elements of PyTorch prediction:")
        print(pred.numpy()[:5])
        print("First 5 elements of C++ prediction:")
        print(pred_cpp[:5])

if __name__ == '__main__':
    main()
