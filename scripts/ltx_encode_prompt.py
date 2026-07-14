#!/usr/bin/env python3
import argparse
import struct
import torch
from transformers import T5EncoderModel, T5TokenizerFast

def encode(text, tokenizer, model, max_length):
    ids = tokenizer(text, truncation=True, max_length=max_length, return_tensors="pt").input_ids
    with torch.no_grad():
        emb = model(ids).last_hidden_state[0].float().numpy()
    return emb

def write_emb(path, emb):
    with open(path, "wb") as f:
        f.write(struct.pack("<III", 0x4C545845, emb.shape[0], emb.shape[1]))
        f.write(emb.astype("<f4").tobytes())

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--negative", default="")
    parser.add_argument("--neg-out")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    tokenizer = T5TokenizerFast.from_pretrained(args.model_dir + "/tokenizer")
    model = T5EncoderModel.from_pretrained(
        args.model_dir + "/text_encoder",
        torch_dtype=torch.bfloat16
    )
    model.eval()

    pos_emb = encode(args.prompt, tokenizer, model, args.max_length)
    write_emb(args.out, pos_emb)
    print(f"Encoded {pos_emb.shape[0]} tokens to {args.out}")

    if args.neg_out:
        neg_emb = encode(args.negative, tokenizer, model, args.max_length)
        write_emb(args.neg_out, neg_emb)
        print(f"Encoded {neg_emb.shape[0]} negative tokens to {args.neg_out}")

if __name__ == "__main__":
    main()
