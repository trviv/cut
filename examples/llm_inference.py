#!/usr/bin/env python3
"""
Unified LLM Inference Script

This script provides a unified interface for loading and running inference
on LLM models in both GGUF and SafeTensor formats using PyTorch CPU backend.

Supported formats:
  - GGUF (.gguf) - Quantized models from llama.cpp ecosystem
  - SafeTensor (.safetensors) - Hugging Face model format

Usage:
    python llm_inference.py [--model MODEL_PATH] [--prompt PROMPT] [--max-tokens N]

Examples:
    python llm_inference.py --model models/llama.gguf --prompt "Hello"
    python llm_inference.py --model models/model.safetensors --prompt "Hello"
    python llm_inference.py --info  # Show model info without loading PyTorch
"""

import argparse
import os
from pathlib import Path
from typing import List, Optional, Union
from enum import Enum, auto


class ModelFormat(Enum):
    """Supported model formats."""
    GGUF = auto()
    SAFETENSOR = auto()
    UNKNOWN = auto()


def detect_format(path: str) -> ModelFormat:
    """Detect model format from file extension."""
    path_lower = path.lower()
    if path_lower.endswith('.gguf'):
        return ModelFormat.GGUF
    elif path_lower.endswith('.safetensors'):
        return ModelFormat.SAFETENSOR
    else:
        return ModelFormat.UNKNOWN


def find_models(models_dir: str) -> List[str]:
    """Find all supported model files in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []

    models = []
    models.extend(models_path.glob('*.gguf'))
    models.extend(models_path.glob('*.safetensors'))
    return [str(p) for p in sorted(models)]


def show_info(model_path: str):
    """Show model information without loading PyTorch."""
    fmt = detect_format(model_path)

    print(f"\nModel: {model_path}")
    print(f"Format: {fmt.name}")
    print("=" * 60)

    if fmt == ModelFormat.GGUF:
        from gguf_inference import GGUFReader, GGML_TYPE_INFO

        reader = GGUFReader(model_path)

        print(f"\nModel Information:")
        print(f"  Architecture: {reader.architecture}")
        print(f"  Name: {reader.name}")
        print(f"  Tensors: {len(reader.tensors)}")

        print(f"\nMetadata:")
        for key, value in sorted(reader.metadata.items()):
            if not key.startswith('tokenizer.ggml'):
                print(f"  {key}: {value}")

        print(f"\nTensors:")
        for name, info in sorted(reader.tensors.items()):
            dtype_name = GGML_TYPE_INFO.get(info.dtype, (0, 0, 'unknown'))[2]
            print(f"  {name}: {info.shape} ({dtype_name})")

    elif fmt == ModelFormat.SAFETENSOR:
        from safetensor_inference import SafeTensorReader, ModelConfig, find_model_files

        reader = SafeTensorReader(model_path)
        model_files = find_model_files(model_path)

        print(f"\nFile Information:")
        print(f"  Header size: {reader.header_size} bytes")
        print(f"  Data offset: {reader.data_offset} bytes")
        print(f"  Tensors: {len(reader.tensors)}")

        # Show related files found
        print(f"\nModel Directory: {Path(model_path).parent}")
        print(f"  Found files:")
        for file_type, file_path in model_files.items():
            if file_path:
                print(f"    {file_type}: {Path(file_path).name}")

        # Load and show config if available
        if model_files['config']:
            config = ModelConfig.from_json(model_files['config'])
            print(f"\nModel Config (from config.json):")
            print(f"  Architecture: {config.architecture}")
            print(f"  Hidden size: {config.hidden_size}")
            print(f"  Layers: {config.num_hidden_layers}")
            print(f"  Attention heads: {config.num_attention_heads}")
            print(f"  KV heads: {config.num_key_value_heads or config.num_attention_heads}")
            print(f"  Vocab size: {config.vocab_size}")

        if reader.metadata:
            print(f"\nSafeTensor Metadata:")
            for key, value in sorted(reader.metadata.items()):
                print(f"  {key}: {value}")

        print(f"\nTensors ({len(reader.tensors)}):")
        for name, info in sorted(reader.tensors.items()):
            print(f"  {name}: {info.shape} ({info.dtype.value})")

    else:
        print(f"Unknown model format: {model_path}")


def load_model_and_tokenizer(model_path: str, tokenizer_path: Optional[str] = None):
    """Load model and tokenizer based on format."""
    fmt = detect_format(model_path)

    if fmt == ModelFormat.GGUF:
        from gguf_inference import (
            GGUFReader, create_llm_model, SimpleTokenizer
        )

        reader = GGUFReader(model_path)
        print(f"\nLoading GGUF model: {model_path}")

        print("\nLoading tokenizer...")
        tokenizer = SimpleTokenizer.from_gguf(reader)
        print(f"Vocabulary size: {len(tokenizer.vocab)}")

        print("\nLoading model weights...")
        model = create_llm_model(reader)

        return model, tokenizer

    elif fmt == ModelFormat.SAFETENSOR:
        from safetensor_inference import (
            SafeTensorReader, create_llm_model, SimpleTokenizer,
            ModelConfig, find_model_files
        )

        reader = SafeTensorReader(model_path)
        print(f"\nLoading SafeTensor model: {model_path}")

        # Find related files in model directory
        model_files = find_model_files(model_path)
        model_dir = Path(model_path).parent

        # Load config from model directory
        config = None
        if model_files['config']:
            print(f"Loading config from: {model_files['config']}")
            config = ModelConfig.from_json(model_files['config'])

        print("\nLoading tokenizer...")
        if tokenizer_path and os.path.exists(tokenizer_path):
            tokenizer = SimpleTokenizer.from_tokenizer_json(
                tokenizer_path,
                model_files.get('tokenizer_config'),
                model_files.get('special_tokens_map')
            )
        else:
            tokenizer = SimpleTokenizer.from_model_directory(str(model_dir))
        print(f"Vocabulary size: {len(tokenizer.vocab)}")

        print("\nLoading model weights...")
        model = create_llm_model(reader, config)

        return model, tokenizer

    else:
        raise ValueError(f"Unsupported model format: {model_path}")


def run_inference(model, tokenizer, prompt: str, max_tokens: int = 50,
                  temperature: float = 0.8, top_p: float = 0.9, top_k: int = 40):
    """Run text generation inference."""
    # Import generate from appropriate module based on what's available
    try:
        from gguf_inference import generate
    except ImportError:
        from safetensor_inference import generate

    model.eval()
    print(f"\nGenerating text (max {max_tokens} tokens)...")
    print("-" * 50)

    output = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k
    )

    print("-" * 50)
    return output


def interactive_mode(model, tokenizer, max_tokens: int = 50,
                     temperature: float = 0.8, top_p: float = 0.9, top_k: int = 40):
    """Run in interactive chat mode."""
    try:
        from gguf_inference import generate
    except ImportError:
        from safetensor_inference import generate

    model.eval()
    print("\nInteractive mode. Type 'quit' or 'exit' to stop.")
    print("-" * 50)

    while True:
        try:
            prompt = input("\nYou: ").strip()
            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Goodbye!")
                break
            if not prompt:
                continue

            print("\nAssistant:", end='')
            generate(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k
            )

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except EOFError:
            print("\nGoodbye!")
            break


def main():
    parser = argparse.ArgumentParser(
        description='Unified LLM Inference (GGUF & SafeTensor)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --list                                  # List available models
  %(prog)s --info                                  # Show model info
  %(prog)s --prompt "Hello, world"                 # Generate text
  %(prog)s --model path/to/model.gguf --prompt "Hi"
  %(prog)s --interactive                           # Interactive chat mode
        """
    )

    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to model file (.gguf or .safetensors)')
    parser.add_argument('--models-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'models'),
                        help='Directory containing models')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer.json (for SafeTensor models)')
    parser.add_argument('--prompt', '-p', type=str, default=None,
                        help='Prompt for text generation')
    parser.add_argument('--max-tokens', '-n', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Sampling temperature (0 = greedy)')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available models')
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show model information (no PyTorch needed)')
    parser.add_argument('--interactive', action='store_true',
                        help='Run in interactive chat mode')

    args = parser.parse_args()

    # List models
    if args.list:
        models = find_models(args.models_dir)
        print("Available models:")
        if not models:
            print(f"  (No models found in {args.models_dir})")
        else:
            for m in models:
                fmt = detect_format(m)
                print(f"  [{fmt.name:10}] {m}")
        return

    # Find model path
    if args.model:
        model_path = args.model
    else:
        models = find_models(args.models_dir)
        if not models:
            print(f"No models found in {args.models_dir}")
            print("Use --model to specify a model path, or place models in the models/ directory")
            return
        model_path = models[0]
        print(f"Using model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Show info mode
    if args.info:
        show_info(model_path)
        return

    # Load model and tokenizer
    model, tokenizer = load_model_and_tokenizer(model_path, args.tokenizer)
    model.eval()

    # Interactive mode
    if args.interactive:
        interactive_mode(
            model=model,
            tokenizer=tokenizer,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k
        )
        return

    # Single prompt mode
    prompt = args.prompt or "Hello, I am"
    run_inference(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    print("\nGeneration complete!")


if __name__ == '__main__':
    main()
