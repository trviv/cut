#!/usr/bin/env python3
import argparse
import struct
import torch
import numpy as np
from diffusers import AutoencoderKLLTXVideo
from diffusers.utils import export_to_video
from diffusers.video_processor import VideoProcessor

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latents", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--fps", type=int, default=25)
    args = parser.parse_args()

    with open(args.latents, "rb") as f:
        magic, latent_frames, latent_height, latent_width, channels = struct.unpack("<IIIII", f.read(20))
        assert magic == 0x4C54584C, "Invalid latent file magic"
        data = np.frombuffer(f.read(), dtype=np.float32).copy()
        latents = data.reshape(latent_frames, latent_height, latent_width, channels)
        latents = np.transpose(latents, (3, 0, 1, 2))  # [C, F, H, W]
        latents = torch.from_numpy(latents).unsqueeze(0)  # [1, C, F, H, W]

    vae = AutoencoderKLLTXVideo.from_pretrained(
        args.model_dir + "/vae",
        torch_dtype=torch.float32
    )
    vae.eval()

    mean = vae.latents_mean.view(1, -1, 1, 1, 1)
    std = vae.latents_std.view(1, -1, 1, 1, 1)
    latents = latents * std / vae.config.scaling_factor + mean

    if vae.config.timestep_conditioning:
        temb = torch.tensor([0.0], dtype=latents.dtype)
    else:
        temb = None

    with torch.no_grad():
        video = vae.decode(latents, temb, return_dict=False)[0]

    video_processor = VideoProcessor(vae_scale_factor=32)
    frames = video_processor.postprocess_video(video, output_type="np")[0]

    export_to_video(frames, args.out, fps=args.fps)
    print(f"Saved {len(frames)} frames to {args.out}")

if __name__ == "__main__":
    main()
