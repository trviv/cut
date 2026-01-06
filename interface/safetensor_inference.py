#!/usr/bin/env python3
"""
SafeTensor Model Loader and Inference Script

This script reads SafeTensor model files from the models directory and runs
inference on them using PyTorch CPU backend.

Usage:
    python safetensor_inference.py [--model MODEL_PATH] [--prompt PROMPT] [--max-tokens N]

Example:
    python safetensor_inference.py --prompt "Hello, how are you?"
    python safetensor_inference.py --info  # Show model info without loading PyTorch
"""

import argparse
import json
import struct
import os
from pathlib import Path
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
import math


# =============================================================================
# SafeTensor Constants and Types
# =============================================================================

class DType(Enum):
    """SafeTensor data types."""
    BOOL = "BOOL"
    U8 = "U8"
    I8 = "I8"
    U16 = "U16"
    I16 = "I16"
    U32 = "U32"
    I32 = "I32"
    U64 = "U64"
    I64 = "I64"
    F16 = "F16"
    BF16 = "BF16"
    F32 = "F32"
    F64 = "F64"


DTYPE_SIZES = {
    DType.BOOL: 1,
    DType.U8: 1,
    DType.I8: 1,
    DType.U16: 2,
    DType.I16: 2,
    DType.U32: 4,
    DType.I32: 4,
    DType.U64: 8,
    DType.I64: 8,
    DType.F16: 2,
    DType.BF16: 2,
    DType.F32: 4,
    DType.F64: 8,
}


@dataclass
class TensorInfo:
    """Information about a tensor in the SafeTensor file."""
    name: str
    dtype: DType
    shape: List[int]
    data_offset: int
    data_size: int

    @property
    def n_elements(self) -> int:
        result = 1
        for dim in self.shape:
            result *= dim
        return result

    def nbytes(self) -> int:
        return self.n_elements * DTYPE_SIZES[self.dtype]


@dataclass
class ModelConfig:
    """Model configuration loaded from config.json."""
    hidden_size: int = 4096
    num_hidden_layers: int = 32
    num_attention_heads: int = 32
    num_key_value_heads: Optional[int] = None
    vocab_size: int = 32000
    intermediate_size: int = 11008
    rms_norm_eps: float = 1e-5
    max_position_embeddings: int = 2048
    rope_theta: float = 10000.0
    architecture: str = "LlamaForCausalLM"

    @classmethod
    def from_json(cls, path: str) -> 'ModelConfig':
        """Load config from config.json file."""
        with open(path, 'r') as f:
            data = json.load(f)

        return cls(
            hidden_size=data.get('hidden_size', 4096),
            num_hidden_layers=data.get('num_hidden_layers', 32),
            num_attention_heads=data.get('num_attention_heads', 32),
            num_key_value_heads=data.get('num_key_value_heads'),
            vocab_size=data.get('vocab_size', 32000),
            intermediate_size=data.get('intermediate_size', 11008),
            rms_norm_eps=data.get('rms_norm_eps', 1e-5),
            max_position_embeddings=data.get('max_position_embeddings', 2048),
            rope_theta=data.get('rope_theta', 10000.0),
            architecture=data.get('architectures', ['LlamaForCausalLM'])[0] if data.get('architectures') else 'LlamaForCausalLM'
        )

    @classmethod
    def from_directory(cls, model_dir: str) -> Optional['ModelConfig']:
        """Try to load config from model directory."""
        config_path = Path(model_dir) / 'config.json'
        if config_path.exists():
            return cls.from_json(str(config_path))
        return None


def find_model_files(model_path: str) -> Dict[str, Optional[str]]:
    """Find related model files in the same directory."""
    model_dir = Path(model_path).parent

    files = {
        'model': model_path,
        'config': None,
        'tokenizer': None,
        'tokenizer_config': None,
        'special_tokens_map': None,
        'generation_config': None,
    }

    # Check for common files
    if (model_dir / 'config.json').exists():
        files['config'] = str(model_dir / 'config.json')
    if (model_dir / 'tokenizer.json').exists():
        files['tokenizer'] = str(model_dir / 'tokenizer.json')
    if (model_dir / 'tokenizer_config.json').exists():
        files['tokenizer_config'] = str(model_dir / 'tokenizer_config.json')
    if (model_dir / 'special_tokens_map.json').exists():
        files['special_tokens_map'] = str(model_dir / 'special_tokens_map.json')
    if (model_dir / 'generation_config.json').exists():
        files['generation_config'] = str(model_dir / 'generation_config.json')

    return files


# =============================================================================
# SafeTensor Reader (No PyTorch dependency)
# =============================================================================

class SafeTensorReader:
    """Reader for SafeTensor model files."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, str] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.header_size = 0
        self.data_offset = 0

        with open(path, 'rb') as f:
            self._parse_header(f)

    def _parse_header(self, f):
        # Read header size (8 bytes, little-endian uint64)
        header_len_bytes = f.read(8)
        if len(header_len_bytes) < 8:
            raise ValueError("Failed to read header size")

        self.header_size = struct.unpack('<Q', header_len_bytes)[0]
        self.data_offset = 8 + self.header_size

        # Read JSON header
        header_json = f.read(self.header_size).decode('utf-8')
        header = json.loads(header_json)

        # Parse metadata and tensors
        for key, value in header.items():
            if key == "__metadata__":
                self.metadata = value
            else:
                # Parse tensor info
                dtype_str = value.get("dtype", "F32")
                try:
                    dtype = DType(dtype_str)
                except ValueError:
                    dtype = DType.F32

                shape = value.get("shape", [])
                data_offsets = value.get("data_offsets", [0, 0])

                self.tensors[key] = TensorInfo(
                    name=key,
                    dtype=dtype,
                    shape=shape,
                    data_offset=data_offsets[0],
                    data_size=data_offsets[1] - data_offsets[0]
                )

    def get_tensor_names(self) -> List[str]:
        return list(self.tensors.keys())

    def read_tensor_raw(self, name: str) -> bytes:
        """Read raw tensor data."""
        tensor = self.tensors[name]
        with open(self.path, 'rb') as f:
            f.seek(self.data_offset + tensor.data_offset)
            return f.read(tensor.data_size)


# =============================================================================
# PyTorch-dependent code (imported lazily)
# =============================================================================

def _import_torch():
    """Import PyTorch and return the modules."""
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        return torch, nn, F
    except ImportError:
        raise ImportError(
            "PyTorch is required for model inference. Install it with:\n"
            "  pip install torch\n"
            "or for CPU-only:\n"
            "  pip install torch --index-url https://download.pytorch.org/whl/cpu"
        )


class SafeTensorLoader:
    """Load SafeTensor tensors into PyTorch tensors."""

    def __init__(self, reader: SafeTensorReader):
        self.reader = reader
        self.torch, _, _ = _import_torch()

    def read_tensor(self, name: str):
        """Read tensor as PyTorch tensor with original dtype."""
        torch = self.torch
        tensor = self.reader.tensors[name]
        raw_data = self.reader.read_tensor_raw(name)

        dtype_map = {
            DType.F32: torch.float32,
            DType.F16: torch.float16,
            DType.BF16: torch.bfloat16,
            DType.F64: torch.float64,
            DType.I8: torch.int8,
            DType.U8: torch.uint8,
            DType.I16: torch.int16,
            DType.I32: torch.int32,
            DType.I64: torch.int64,
        }

        torch_dtype = dtype_map.get(tensor.dtype, torch.float32)
        data = torch.frombuffer(bytearray(raw_data), dtype=torch_dtype).clone()
        return data.reshape(tensor.shape)

    def read_tensor_f32(self, name: str):
        """Read tensor and convert to float32 PyTorch tensor."""
        tensor = self.read_tensor(name)
        return tensor.float()


def create_llm_model(reader: SafeTensorReader, config: Optional[ModelConfig] = None):
    """Create and load an LLM model from SafeTensor file."""
    torch, nn, F = _import_torch()

    # =========================================================================
    # Model Components
    # =========================================================================

    class RMSNorm(nn.Module):
        def __init__(self, dim: int, eps: float = 1e-6):
            super().__init__()
            self.eps = eps
            self.weight = nn.Parameter(torch.ones(dim))

        def forward(self, x):
            rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)
            return x / rms * self.weight

    class RotaryEmbedding(nn.Module):
        def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0):
            super().__init__()
            self.dim = dim
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            self.register_buffer('inv_freq', inv_freq)
            t = torch.arange(max_seq_len).float()
            freqs = torch.outer(t, self.inv_freq)
            emb = torch.cat([freqs, freqs], dim=-1)
            self.register_buffer('cos_cached', emb.cos())
            self.register_buffer('sin_cached', emb.sin())

        def forward(self, seq_len: int):
            return self.cos_cached[:seq_len], self.sin_cached[:seq_len]

    def apply_rotary_pos_emb(q, k, cos, sin):
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat([-x2, x1], dim=-1)
        q_embed = (q * cos) + (rotate_half(q) * sin)
        k_embed = (k * cos) + (rotate_half(k) * sin)
        return q_embed, k_embed

    class Attention(nn.Module):
        def __init__(self, dim: int, n_heads: int, n_kv_heads: int, max_seq_len: int):
            super().__init__()
            self.n_heads = n_heads
            self.n_kv_heads = n_kv_heads
            self.head_dim = dim // n_heads
            self.n_rep = n_heads // n_kv_heads

            self.wq = nn.Linear(dim, n_heads * self.head_dim, bias=False)
            self.wk = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
            self.wv = nn.Linear(dim, n_kv_heads * self.head_dim, bias=False)
            self.wo = nn.Linear(n_heads * self.head_dim, dim, bias=False)
            self.rotary_emb = RotaryEmbedding(self.head_dim, max_seq_len)
            self.cache_k = None
            self.cache_v = None

        def forward(self, x, start_pos: int = 0, mask=None):
            batch_size, seq_len, _ = x.shape
            q = self.wq(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
            k = self.wk(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)
            v = self.wv(x).view(batch_size, seq_len, self.n_kv_heads, self.head_dim)

            cos, sin = self.rotary_emb(start_pos + seq_len)
            cos = cos[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(2)
            sin = sin[start_pos:start_pos + seq_len].unsqueeze(0).unsqueeze(2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            if self.cache_k is not None and start_pos > 0:
                k = torch.cat([self.cache_k, k], dim=1)
                v = torch.cat([self.cache_v, v], dim=1)
            self.cache_k = k
            self.cache_v = v

            if self.n_rep > 1:
                k = k.repeat_interleave(self.n_rep, dim=2)
                v = v.repeat_interleave(self.n_rep, dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
            if mask is not None:
                scores = scores + mask
            attn = F.softmax(scores, dim=-1)
            out = torch.matmul(attn, v)
            out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
            return self.wo(out)

        def clear_cache(self):
            self.cache_k = None
            self.cache_v = None

    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden_dim: int):
            super().__init__()
            self.w1 = nn.Linear(dim, hidden_dim, bias=False)
            self.w2 = nn.Linear(hidden_dim, dim, bias=False)
            self.w3 = nn.Linear(dim, hidden_dim, bias=False)

        def forward(self, x):
            return self.w2(F.silu(self.w1(x)) * self.w3(x))

    class TransformerBlock(nn.Module):
        def __init__(self, dim: int, n_heads: int, n_kv_heads: int,
                     hidden_dim: int, norm_eps: float, max_seq_len: int):
            super().__init__()
            self.attention = Attention(dim, n_heads, n_kv_heads, max_seq_len)
            self.feed_forward = FeedForward(dim, hidden_dim)
            self.attention_norm = RMSNorm(dim, eps=norm_eps)
            self.ffn_norm = RMSNorm(dim, eps=norm_eps)

        def forward(self, x, start_pos: int = 0, mask=None):
            h = x + self.attention(self.attention_norm(x), start_pos, mask)
            return h + self.feed_forward(self.ffn_norm(h))

        def clear_cache(self):
            self.attention.clear_cache()

    class LLM(nn.Module):
        def __init__(self, dim, n_layers, n_heads, n_kv_heads, vocab_size,
                     hidden_dim, norm_eps, max_seq_len):
            super().__init__()
            self.tok_embeddings = nn.Embedding(vocab_size, dim)
            self.layers = nn.ModuleList([
                TransformerBlock(dim, n_heads, n_kv_heads, hidden_dim, norm_eps, max_seq_len)
                for _ in range(n_layers)
            ])
            self.norm = RMSNorm(dim, eps=norm_eps)
            self.output = nn.Linear(dim, vocab_size, bias=False)

        def forward(self, tokens, start_pos: int = 0):
            batch_size, seq_len = tokens.shape
            h = self.tok_embeddings(tokens)

            mask = None
            if seq_len > 1:
                mask = torch.full((seq_len, seq_len), float('-inf'), device=tokens.device)
                mask = torch.triu(mask, diagonal=1)
                if start_pos > 0:
                    mask = torch.cat([
                        torch.zeros((seq_len, start_pos), device=tokens.device),
                        mask
                    ], dim=1)
                mask = mask.unsqueeze(0).unsqueeze(0)

            for layer in self.layers:
                h = layer(h, start_pos, mask)

            return self.output(self.norm(h))

        def clear_cache(self):
            for layer in self.layers:
                layer.clear_cache()

    # =========================================================================
    # Get model config (from config.json or infer from tensor shapes)
    # =========================================================================

    tensor_names = reader.get_tensor_names()
    print(f"Found {len(tensor_names)} tensors")

    if config:
        # Use provided config from config.json
        print("Using config from config.json")
        dim = config.hidden_size
        n_layers = config.num_hidden_layers
        n_heads = config.num_attention_heads
        n_kv_heads = config.num_key_value_heads or n_heads
        vocab_size = config.vocab_size
        hidden_dim = config.intermediate_size
        norm_eps = config.rms_norm_eps
        max_seq_len = config.max_position_embeddings
    else:
        # Infer configuration from tensor shapes
        print("Inferring config from tensor shapes")
        dim = None
        n_layers = 0
        n_heads = None
        n_kv_heads = None
        vocab_size = None
        hidden_dim = None

        for name, info in reader.tensors.items():
            # Count layers
            if name.startswith('model.layers.') or name.startswith('layers.'):
                parts = name.split('.')
                for i, part in enumerate(parts):
                    if part == 'layers' and i + 1 < len(parts):
                        try:
                            layer_idx = int(parts[i + 1])
                            n_layers = max(n_layers, layer_idx + 1)
                        except ValueError:
                            pass

            # Get embedding dimension
            if 'embed_tokens' in name or 'tok_embeddings' in name or 'token_embd' in name:
                if len(info.shape) == 2:
                    vocab_size, dim = info.shape

            # Get hidden dimension from FFN
            if 'gate_proj' in name or 'ffn_gate' in name or 'w1' in name:
                if len(info.shape) == 2:
                    hidden_dim = info.shape[0]

            # Get attention heads from q_proj
            if 'q_proj' in name or 'wq' in name:
                if len(info.shape) == 2 and dim:
                    total_head_dim = info.shape[0]
                    # Assume head_dim = 128 for modern models
                    n_heads = total_head_dim // 128 if total_head_dim >= 128 else total_head_dim // 64

            # Get KV heads from k_proj
            if 'k_proj' in name or 'wk' in name:
                if len(info.shape) == 2 and dim:
                    total_kv_dim = info.shape[0]
                    n_kv_heads = total_kv_dim // 128 if total_kv_dim >= 128 else total_kv_dim // 64

        # Set defaults if not found
        dim = dim or 4096
        n_layers = n_layers or 32
        n_heads = n_heads or 32
        n_kv_heads = n_kv_heads or n_heads
        vocab_size = vocab_size or 32000
        hidden_dim = hidden_dim or (dim * 4)
        norm_eps = 1e-5
        max_seq_len = 2048

    print(f"Config: dim={dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}")
    print(f"        vocab_size={vocab_size}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")

    # Create model
    model = LLM(dim, n_layers, n_heads, n_kv_heads, vocab_size,
                hidden_dim, norm_eps, max_seq_len)

    # Load weights
    print("Loading weights...")
    loader = SafeTensorLoader(reader)
    loaded = 0

    # Map SafeTensor names to PyTorch state dict keys
    # Support multiple naming conventions
    name_patterns = [
        # LLaMA/Hugging Face style
        ('model.embed_tokens.weight', 'tok_embeddings.weight'),
        ('model.norm.weight', 'norm.weight'),
        ('lm_head.weight', 'output.weight'),
        # GGUF-converted style
        ('token_embd.weight', 'tok_embeddings.weight'),
        ('output_norm.weight', 'norm.weight'),
        ('output.weight', 'output.weight'),
    ]

    layer_patterns = [
        # LLaMA/Hugging Face style
        ('self_attn.q_proj.weight', 'attention.wq.weight'),
        ('self_attn.k_proj.weight', 'attention.wk.weight'),
        ('self_attn.v_proj.weight', 'attention.wv.weight'),
        ('self_attn.o_proj.weight', 'attention.wo.weight'),
        ('input_layernorm.weight', 'attention_norm.weight'),
        ('mlp.gate_proj.weight', 'feed_forward.w1.weight'),
        ('mlp.down_proj.weight', 'feed_forward.w2.weight'),
        ('mlp.up_proj.weight', 'feed_forward.w3.weight'),
        ('post_attention_layernorm.weight', 'ffn_norm.weight'),
        # Alternative naming
        ('attn_q.weight', 'attention.wq.weight'),
        ('attn_k.weight', 'attention.wk.weight'),
        ('attn_v.weight', 'attention.wv.weight'),
        ('attn_output.weight', 'attention.wo.weight'),
        ('attn_norm.weight', 'attention_norm.weight'),
        ('ffn_gate.weight', 'feed_forward.w1.weight'),
        ('ffn_down.weight', 'feed_forward.w2.weight'),
        ('ffn_up.weight', 'feed_forward.w3.weight'),
        ('ffn_norm.weight', 'ffn_norm.weight'),
    ]

    name_mapping = {}

    # Build name mapping
    for st_name, pt_name in name_patterns:
        if st_name in tensor_names:
            name_mapping[st_name] = pt_name

    # Map layer tensors
    for name in tensor_names:
        # Extract layer index
        layer_idx = None
        for prefix in ['model.layers.', 'layers.', 'blk.']:
            if name.startswith(prefix):
                rest = name[len(prefix):]
                try:
                    layer_idx = int(rest.split('.')[0])
                    rest_name = '.'.join(rest.split('.')[1:])
                    for st_pattern, pt_pattern in layer_patterns:
                        if rest_name == st_pattern:
                            name_mapping[name] = f'layers.{layer_idx}.{pt_pattern}'
                            break
                except ValueError:
                    pass
                break

    state_dict = model.state_dict()
    for st_name, pt_name in name_mapping.items():
        if st_name in tensor_names and pt_name in state_dict:
            try:
                tensor = loader.read_tensor_f32(st_name)
                if tensor.shape == state_dict[pt_name].shape:
                    state_dict[pt_name] = tensor
                    loaded += 1
                else:
                    print(f"  Shape mismatch for {pt_name}: "
                          f"expected {state_dict[pt_name].shape}, got {tensor.shape}")
            except Exception as e:
                print(f"  Failed to load {st_name}: {e}")

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {loaded} tensors")

    return model


# =============================================================================
# Simple Tokenizer
# =============================================================================

class SimpleTokenizer:
    """Simple tokenizer - loads from tokenizer.json and related files."""

    def __init__(self, vocab: Optional[List[str]] = None,
                 bos_token: str = '<s>', eos_token: str = '</s>',
                 pad_token: str = '<pad>', unk_token: str = '<unk>'):
        if vocab:
            self.vocab = vocab
            self.token_to_id = {token: i for i, token in enumerate(vocab)}
            self.id_to_token = {i: token for i, token in enumerate(vocab)}
        else:
            # Basic ASCII fallback
            self.vocab = [unk_token, bos_token, eos_token] + [chr(i) for i in range(256)]
            self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
            self.id_to_token = {i: token for i, token in enumerate(self.vocab)}

        # Store special tokens
        self.bos_token = bos_token
        self.eos_token = eos_token
        self.pad_token = pad_token
        self.unk_token = unk_token

        self.bos_token_id = self.token_to_id.get(bos_token, 1)
        self.eos_token_id = self.token_to_id.get(eos_token, 2)
        self.pad_token_id = self.token_to_id.get(pad_token, 0)
        self.unk_token_id = self.token_to_id.get(unk_token, 0)

    def encode(self, text: str, add_bos: bool = True) -> List[int]:
        """Encode text to token IDs."""
        tokens = []
        if add_bos:
            tokens.append(self.bos_token_id)

        i = 0
        while i < len(text):
            best_match = None
            best_len = 0

            for length in range(min(20, len(text) - i), 0, -1):
                substr = text[i:i+length]
                if substr in self.token_to_id:
                    best_match = substr
                    best_len = length
                    break

            if best_match:
                tokens.append(self.token_to_id[best_match])
                i += best_len
            else:
                byte_val = ord(text[i])
                byte_token = f'<0x{byte_val:02X}>'
                if byte_token in self.token_to_id:
                    tokens.append(self.token_to_id[byte_token])
                else:
                    # Fall back to character
                    char = text[i]
                    if char in self.token_to_id:
                        tokens.append(self.token_to_id[char])
                    else:
                        tokens.append(self.unk_token_id)
                i += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        text = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                # Skip special tokens
                if token in [self.bos_token, self.eos_token, self.pad_token, self.unk_token]:
                    continue
                if token.startswith('<0x') and token.endswith('>'):
                    try:
                        byte_val = int(token[3:-1], 16)
                        text.append(chr(byte_val))
                    except:
                        text.append(token)
                else:
                    # Handle sentencepiece-style space encoding
                    token = token.replace('▁', ' ')
                    token = token.replace('Ġ', ' ')  # GPT-style space
                    text.append(token)
        return ''.join(text)

    @classmethod
    def from_tokenizer_json(cls, path: str, tokenizer_config_path: Optional[str] = None,
                            special_tokens_path: Optional[str] = None) -> 'SimpleTokenizer':
        """Load tokenizer from tokenizer.json and related files."""
        with open(path, 'r') as f:
            data = json.load(f)

        # Load vocab
        vocab = []
        if 'model' in data and 'vocab' in data['model']:
            # Hugging Face tokenizers format
            vocab_dict = data['model']['vocab']
            vocab = [''] * len(vocab_dict)
            for token, idx in vocab_dict.items():
                if idx < len(vocab):
                    vocab[idx] = token
        elif 'added_tokens' in data:
            # Try to build vocab from added_tokens
            max_id = max(t['id'] for t in data['added_tokens']) if data['added_tokens'] else 0
            vocab = [''] * (max_id + 1)
            for token_info in data['added_tokens']:
                vocab[token_info['id']] = token_info['content']

        # Default special tokens
        bos_token = '<s>'
        eos_token = '</s>'
        pad_token = '<pad>'
        unk_token = '<unk>'

        # Load special tokens from tokenizer_config.json
        if tokenizer_config_path and os.path.exists(tokenizer_config_path):
            with open(tokenizer_config_path, 'r') as f:
                config = json.load(f)
                bos_token = config.get('bos_token', bos_token)
                eos_token = config.get('eos_token', eos_token)
                pad_token = config.get('pad_token', pad_token)
                unk_token = config.get('unk_token', unk_token)

                # Handle dict format for special tokens
                if isinstance(bos_token, dict):
                    bos_token = bos_token.get('content', '<s>')
                if isinstance(eos_token, dict):
                    eos_token = eos_token.get('content', '</s>')
                if isinstance(pad_token, dict):
                    pad_token = pad_token.get('content', '<pad>')
                if isinstance(unk_token, dict):
                    unk_token = unk_token.get('content', '<unk>')

        # Load special tokens from special_tokens_map.json
        if special_tokens_path and os.path.exists(special_tokens_path):
            with open(special_tokens_path, 'r') as f:
                special_map = json.load(f)
                if 'bos_token' in special_map:
                    bos_token = special_map['bos_token'] if isinstance(special_map['bos_token'], str) else special_map['bos_token'].get('content', bos_token)
                if 'eos_token' in special_map:
                    eos_token = special_map['eos_token'] if isinstance(special_map['eos_token'], str) else special_map['eos_token'].get('content', eos_token)
                if 'pad_token' in special_map:
                    pad_token = special_map['pad_token'] if isinstance(special_map['pad_token'], str) else special_map['pad_token'].get('content', pad_token)
                if 'unk_token' in special_map:
                    unk_token = special_map['unk_token'] if isinstance(special_map['unk_token'], str) else special_map['unk_token'].get('content', unk_token)

        return cls(vocab if vocab else None, bos_token, eos_token, pad_token, unk_token)

    @classmethod
    def from_model_directory(cls, model_dir: str) -> 'SimpleTokenizer':
        """Load tokenizer from model directory, looking for all related files."""
        model_dir = Path(model_dir)

        tokenizer_path = model_dir / 'tokenizer.json'
        tokenizer_config_path = model_dir / 'tokenizer_config.json'
        special_tokens_path = model_dir / 'special_tokens_map.json'

        if tokenizer_path.exists():
            return cls.from_tokenizer_json(
                str(tokenizer_path),
                str(tokenizer_config_path) if tokenizer_config_path.exists() else None,
                str(special_tokens_path) if special_tokens_path.exists() else None
            )
        else:
            # Fallback to basic tokenizer
            return cls()


# =============================================================================
# Text Generation
# =============================================================================

def generate(model, tokenizer: SimpleTokenizer, prompt: str,
             max_tokens: int = 100, temperature: float = 0.8,
             top_p: float = 0.9, top_k: int = 40) -> str:
    """Generate text continuation for a prompt."""
    torch, _, F = _import_torch()

    model.eval()
    model.clear_cache()

    tokens = tokenizer.encode(prompt)
    tokens = torch.tensor([tokens], dtype=torch.long)

    generated = []
    print(f"\n{prompt}", end='', flush=True)

    with torch.inference_mode():
        for i in range(max_tokens):
            if i == 0:
                logits = model(tokens, start_pos=0)
            else:
                logits = model(tokens[:, -1:], start_pos=tokens.shape[1] - 1)

            logits = logits[:, -1, :]

            if temperature > 0:
                logits = logits / temperature

            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')

            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=-1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = float('-inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            if next_token.item() == tokenizer.eos_token_id:
                break

            tokens = torch.cat([tokens, next_token], dim=1)
            generated.append(next_token.item())

            token_str = tokenizer.decode([next_token.item()])
            print(token_str, end='', flush=True)

    print()
    return tokenizer.decode(generated)


# =============================================================================
# Main Entry Point
# =============================================================================

def find_models(models_dir: str) -> List[str]:
    """Find all SafeTensor model files in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    return [str(p) for p in models_path.glob('*.safetensors')]


def main():
    parser = argparse.ArgumentParser(description='SafeTensor Model Inference')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to SafeTensor model file')
    parser.add_argument('--models-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'models'),
                        help='Directory containing SafeTensor models')
    parser.add_argument('--tokenizer', type=str, default=None,
                        help='Path to tokenizer.json file')
    parser.add_argument('--prompt', '-p', type=str, default='Hello, I am',
                        help='Prompt for text generation')
    parser.add_argument('--max-tokens', '-n', type=int, default=50,
                        help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', '-t', type=float, default=0.8,
                        help='Sampling temperature')
    parser.add_argument('--top-p', type=float, default=0.9,
                        help='Top-p (nucleus) sampling')
    parser.add_argument('--top-k', type=int, default=40,
                        help='Top-k sampling')
    parser.add_argument('--list', '-l', action='store_true',
                        help='List available models')
    parser.add_argument('--info', '-i', action='store_true',
                        help='Show model information only (no PyTorch needed)')

    args = parser.parse_args()

    # Find models
    if args.model:
        model_path = args.model
    else:
        models = find_models(args.models_dir)
        if args.list:
            print("Available SafeTensor models:")
            for m in models:
                print(f"  - {m}")
            return

        if not models:
            print(f"No SafeTensor models found in {args.models_dir}")
            return

        model_path = models[0]
        print(f"Using model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load SafeTensor (no PyTorch needed)
    print(f"\nLoading SafeTensor file: {model_path}")
    reader = SafeTensorReader(model_path)

    # Find related files in the model directory
    model_files = find_model_files(model_path)
    model_dir = Path(model_path).parent

    if args.info:
        print(f"\nFile Information:")
        print(f"  Header size: {reader.header_size} bytes")
        print(f"  Data offset: {reader.data_offset} bytes")
        print(f"  Tensors: {len(reader.tensors)}")

        # Show related files found
        print(f"\nModel Directory: {model_dir}")
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
            print(f"  Intermediate size: {config.intermediate_size}")
            print(f"  Max position embeddings: {config.max_position_embeddings}")
            print(f"  RMS norm eps: {config.rms_norm_eps}")
            print(f"  RoPE theta: {config.rope_theta}")

        if reader.metadata:
            print(f"\nSafeTensor Metadata:")
            for key, value in sorted(reader.metadata.items()):
                print(f"  {key}: {value}")

        print(f"\nTensors ({len(reader.tensors)}):")
        for name, info in sorted(reader.tensors.items()):
            print(f"  {name}: {info.shape} ({info.dtype.value})")
        return

    # From here, PyTorch is required

    # Load config from model directory if available
    config = None
    if model_files['config']:
        print(f"\nLoading config from: {model_files['config']}")
        config = ModelConfig.from_json(model_files['config'])

    # Load tokenizer from model directory
    print("\nLoading tokenizer...")
    if args.tokenizer and os.path.exists(args.tokenizer):
        tokenizer = SimpleTokenizer.from_tokenizer_json(
            args.tokenizer,
            model_files.get('tokenizer_config'),
            model_files.get('special_tokens_map')
        )
    else:
        tokenizer = SimpleTokenizer.from_model_directory(str(model_dir))
    print(f"Vocabulary size: {len(tokenizer.vocab)}")
    print(f"Special tokens: bos='{tokenizer.bos_token}', eos='{tokenizer.eos_token}'")

    print("\nLoading model (requires PyTorch)...")
    model = create_llm_model(reader, config)
    model.eval()

    print(f"\nGenerating text (max {args.max_tokens} tokens)...")
    print("-" * 50)

    generate(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k
    )

    print("-" * 50)
    print("Generation complete!")


if __name__ == '__main__':
    main()
