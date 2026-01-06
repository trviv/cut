#!/usr/bin/env python3
"""
GGUF Model Loader and Inference Script

This script reads GGUF model files from the models directory and runs
inference on them using PyTorch CPU backend.

Usage:
    python gguf_inference.py [--model MODEL_PATH] [--prompt PROMPT] [--max-tokens N]

Example:
    python gguf_inference.py --prompt "Hello, how are you?"
    python gguf_inference.py --info  # Show model info without loading PyTorch
"""

import argparse
import struct
import os
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Optional, Tuple, Any
import math


# =============================================================================
# GGUF Constants and Types
# =============================================================================

GGUF_MAGIC = 0x46554747  # "GGUF" in ASCII (little-endian)
GGUF_VERSION = 3
DEFAULT_ALIGNMENT = 32


class GGMLType(IntEnum):
    """GGML tensor data types (quantization formats)."""
    F32 = 0
    F16 = 1
    Q4_0 = 2
    Q4_1 = 3
    Q5_0 = 6
    Q5_1 = 7
    Q8_0 = 8
    Q8_1 = 9
    Q2_K = 10
    Q3_K = 11
    Q4_K = 12
    Q5_K = 13
    Q6_K = 14
    Q8_K = 15
    I8 = 24
    I16 = 25
    I32 = 26
    I64 = 27
    F64 = 28
    BF16 = 30


class GGUFValueType(IntEnum):
    """GGUF metadata value types."""
    UINT8 = 0
    INT8 = 1
    UINT16 = 2
    INT16 = 3
    UINT32 = 4
    INT32 = 5
    FLOAT32 = 6
    BOOL = 7
    STRING = 8
    ARRAY = 9
    UINT64 = 10
    INT64 = 11
    FLOAT64 = 12


# Type info: (block_size, type_size_bytes, name)
GGML_TYPE_INFO = {
    GGMLType.F32: (1, 4, "F32"),
    GGMLType.F16: (1, 2, "F16"),
    GGMLType.Q4_0: (32, 18, "Q4_0"),
    GGMLType.Q4_1: (32, 20, "Q4_1"),
    GGMLType.Q5_0: (32, 22, "Q5_0"),
    GGMLType.Q5_1: (32, 24, "Q5_1"),
    GGMLType.Q8_0: (32, 34, "Q8_0"),
    GGMLType.Q8_1: (32, 36, "Q8_1"),
    GGMLType.Q2_K: (256, 84, "Q2_K"),
    GGMLType.Q3_K: (256, 110, "Q3_K"),
    GGMLType.Q4_K: (256, 144, "Q4_K"),
    GGMLType.Q5_K: (256, 176, "Q5_K"),
    GGMLType.Q6_K: (256, 210, "Q6_K"),
    GGMLType.Q8_K: (256, 292, "Q8_K"),
    GGMLType.I8: (1, 1, "I8"),
    GGMLType.I16: (1, 2, "I16"),
    GGMLType.I32: (1, 4, "I32"),
    GGMLType.I64: (1, 8, "I64"),
    GGMLType.F64: (1, 8, "F64"),
    GGMLType.BF16: (1, 2, "BF16"),
}


@dataclass
class TensorInfo:
    """Information about a tensor in the GGUF file."""
    name: str
    dimensions: List[int]
    dtype: GGMLType
    offset: int  # Relative to tensor data section

    @property
    def n_elements(self) -> int:
        result = 1
        for dim in self.dimensions:
            result *= dim
        return result

    @property
    def shape(self) -> Tuple[int, ...]:
        return tuple(self.dimensions)

    def nbytes(self) -> int:
        info = GGML_TYPE_INFO.get(self.dtype)
        if info is None:
            return 0
        block_size, type_size, _ = info
        n_blocks = (self.n_elements + block_size - 1) // block_size
        return n_blocks * type_size


# =============================================================================
# GGUF Reader (No PyTorch dependency)
# =============================================================================

class GGUFReader:
    """Reader for GGUF model files."""

    def __init__(self, path: str):
        self.path = path
        self.metadata: Dict[str, Any] = {}
        self.tensors: Dict[str, TensorInfo] = {}
        self.tensor_data_offset = 0
        self._is_big_endian = False
        self._version = 0

        with open(path, 'rb') as f:
            self._parse_header(f)

    def _read_uint8(self, f) -> int:
        return struct.unpack('<B', f.read(1))[0]

    def _read_int8(self, f) -> int:
        return struct.unpack('<b', f.read(1))[0]

    def _read_uint16(self, f) -> int:
        fmt = '>H' if self._is_big_endian else '<H'
        return struct.unpack(fmt, f.read(2))[0]

    def _read_int16(self, f) -> int:
        fmt = '>h' if self._is_big_endian else '<h'
        return struct.unpack(fmt, f.read(2))[0]

    def _read_uint32(self, f) -> int:
        fmt = '>I' if self._is_big_endian else '<I'
        return struct.unpack(fmt, f.read(4))[0]

    def _read_int32(self, f) -> int:
        fmt = '>i' if self._is_big_endian else '<i'
        return struct.unpack(fmt, f.read(4))[0]

    def _read_uint64(self, f) -> int:
        fmt = '>Q' if self._is_big_endian else '<Q'
        return struct.unpack(fmt, f.read(8))[0]

    def _read_int64(self, f) -> int:
        fmt = '>q' if self._is_big_endian else '<q'
        return struct.unpack(fmt, f.read(8))[0]

    def _read_float32(self, f) -> float:
        fmt = '>f' if self._is_big_endian else '<f'
        return struct.unpack(fmt, f.read(4))[0]

    def _read_float64(self, f) -> float:
        fmt = '>d' if self._is_big_endian else '<d'
        return struct.unpack(fmt, f.read(8))[0]

    def _read_bool(self, f) -> bool:
        return self._read_uint8(f) != 0

    def _read_string(self, f) -> str:
        length = self._read_uint64(f)
        return f.read(length).decode('utf-8')

    def _read_value(self, f, value_type: GGUFValueType) -> Any:
        if value_type == GGUFValueType.UINT8:
            return self._read_uint8(f)
        elif value_type == GGUFValueType.INT8:
            return self._read_int8(f)
        elif value_type == GGUFValueType.UINT16:
            return self._read_uint16(f)
        elif value_type == GGUFValueType.INT16:
            return self._read_int16(f)
        elif value_type == GGUFValueType.UINT32:
            return self._read_uint32(f)
        elif value_type == GGUFValueType.INT32:
            return self._read_int32(f)
        elif value_type == GGUFValueType.FLOAT32:
            return self._read_float32(f)
        elif value_type == GGUFValueType.BOOL:
            return self._read_bool(f)
        elif value_type == GGUFValueType.STRING:
            return self._read_string(f)
        elif value_type == GGUFValueType.ARRAY:
            return self._read_array(f)
        elif value_type == GGUFValueType.UINT64:
            return self._read_uint64(f)
        elif value_type == GGUFValueType.INT64:
            return self._read_int64(f)
        elif value_type == GGUFValueType.FLOAT64:
            return self._read_float64(f)
        else:
            raise ValueError(f"Unknown value type: {value_type}")

    def _read_array(self, f) -> List[Any]:
        element_type = GGUFValueType(self._read_uint32(f))
        count = self._read_uint64(f)
        return [self._read_value(f, element_type) for _ in range(count)]

    def _parse_header(self, f):
        # Read and validate magic number
        magic = self._read_uint32(f)
        if magic == 0x47475546:  # Big-endian magic
            self._is_big_endian = True
            magic = GGUF_MAGIC

        if magic != GGUF_MAGIC:
            raise ValueError(f"Invalid GGUF magic: {magic:#x}")

        # Read version
        self._version = self._read_uint32(f)
        if self._version > GGUF_VERSION:
            raise ValueError(f"Unsupported GGUF version: {self._version}")

        # Read counts
        tensor_count = self._read_uint64(f)
        metadata_kv_count = self._read_uint64(f)

        # Read metadata
        for _ in range(metadata_kv_count):
            key = self._read_string(f)
            value_type = GGUFValueType(self._read_uint32(f))
            value = self._read_value(f, value_type)
            self.metadata[key] = value

        # Read tensor info
        for _ in range(tensor_count):
            name = self._read_string(f)
            n_dims = self._read_uint32(f)
            dims = [self._read_uint64(f) for _ in range(n_dims)]
            dtype = GGMLType(self._read_uint32(f))
            offset = self._read_uint64(f)

            self.tensors[name] = TensorInfo(
                name=name,
                dimensions=dims,
                dtype=dtype,
                offset=offset
            )

        # Calculate tensor data offset with alignment
        alignment = self.metadata.get('general.alignment', DEFAULT_ALIGNMENT)
        current_pos = f.tell()
        self.tensor_data_offset = ((current_pos + alignment - 1) // alignment) * alignment

    @property
    def architecture(self) -> str:
        return self.metadata.get('general.architecture', '')

    @property
    def name(self) -> str:
        return self.metadata.get('general.name', '')

    def get_tensor_names(self) -> List[str]:
        return list(self.tensors.keys())

    def read_tensor_raw(self, name: str) -> bytes:
        """Read raw tensor data."""
        tensor = self.tensors[name]
        with open(self.path, 'rb') as f:
            f.seek(self.tensor_data_offset + tensor.offset)
            return f.read(tensor.nbytes())


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


class GGUFTensorLoader:
    """Load GGUF tensors into PyTorch tensors."""

    def __init__(self, reader: GGUFReader):
        self.reader = reader
        self.torch, _, _ = _import_torch()

    def read_tensor_f32(self, name: str):
        """Read tensor and dequantize to float32 PyTorch tensor."""
        torch = self.torch
        tensor = self.reader.tensors[name]
        raw_data = self.reader.read_tensor_raw(name)

        if tensor.dtype == GGMLType.F32:
            data = torch.frombuffer(bytearray(raw_data), dtype=torch.float32).clone()
        elif tensor.dtype == GGMLType.F16:
            data = torch.frombuffer(bytearray(raw_data), dtype=torch.float16).float().clone()
        elif tensor.dtype == GGMLType.BF16:
            data = torch.frombuffer(bytearray(raw_data), dtype=torch.bfloat16).float().clone()
        elif tensor.dtype == GGMLType.Q4_0:
            data = self._dequantize_q4_0(raw_data, tensor.n_elements)
        elif tensor.dtype == GGMLType.Q8_0:
            data = self._dequantize_q8_0(raw_data, tensor.n_elements)
        elif tensor.dtype == GGMLType.Q4_K:
            data = self._dequantize_q4_k(raw_data, tensor.n_elements)
        elif tensor.dtype == GGMLType.Q6_K:
            data = self._dequantize_q6_k(raw_data, tensor.n_elements)
        else:
            raise NotImplementedError(f"Dequantization not implemented for {tensor.dtype.name}")

        # Reshape to original dimensions
        # GGUF stores dimensions in reverse order (column-major)
        shape = tuple(reversed(tensor.dimensions))
        return data.reshape(shape)

    def _dequantize_q4_0(self, data: bytes, n_elements: int):
        """Dequantize Q4_0 format to float32."""
        torch = self.torch
        block_size = 32
        n_blocks = (n_elements + block_size - 1) // block_size
        output = torch.zeros(n_elements, dtype=torch.float32)

        offset = 0
        out_idx = 0
        for _ in range(n_blocks):
            # Read scale (float16)
            scale = struct.unpack('<e', data[offset:offset+2])[0]
            offset += 2

            # Read 16 bytes (32 nibbles)
            for byte_idx in range(16):
                if out_idx >= n_elements:
                    break
                byte = data[offset + byte_idx]

                # Lower nibble
                q0 = (byte & 0x0F) - 8
                output[out_idx] = q0 * scale
                out_idx += 1

                if out_idx >= n_elements:
                    break

                # Upper nibble
                q1 = ((byte >> 4) & 0x0F) - 8
                output[out_idx] = q1 * scale
                out_idx += 1

            offset += 16

        return output

    def _dequantize_q8_0(self, data: bytes, n_elements: int):
        """Dequantize Q8_0 format to float32."""
        torch = self.torch
        block_size = 32
        n_blocks = (n_elements + block_size - 1) // block_size
        output = torch.zeros(n_elements, dtype=torch.float32)

        offset = 0
        out_idx = 0
        for _ in range(n_blocks):
            # Read scale (float16)
            scale = struct.unpack('<e', data[offset:offset+2])[0]
            offset += 2

            # Read 32 int8 values
            for i in range(32):
                if out_idx >= n_elements:
                    break
                q = struct.unpack('<b', data[offset+i:offset+i+1])[0]
                output[out_idx] = q * scale
                out_idx += 1

            offset += 32

        return output

    def _dequantize_q4_k(self, data: bytes, n_elements: int):
        """Dequantize Q4_K format to float32."""
        torch = self.torch
        block_size = 256
        n_blocks = (n_elements + block_size - 1) // block_size
        output = torch.zeros(n_elements, dtype=torch.float32)

        offset = 0
        out_idx = 0
        for _ in range(n_blocks):
            if out_idx >= n_elements:
                break

            # Q4_K block structure: scales, mins, quantized values
            d = struct.unpack('<e', data[offset:offset+2])[0]
            offset += 4  # Skip d and dmin

            # Skip scales/mins encoding (12 bytes)
            offset += 12

            # Read quantized values (128 bytes)
            qs = data[offset:offset+128]
            offset += 128

            # Decode (simplified)
            for i in range(128):
                if out_idx >= n_elements:
                    break
                byte = qs[i]

                # Lower nibble
                q0 = byte & 0x0F
                output[out_idx] = d * q0
                out_idx += 1

                if out_idx >= n_elements:
                    break

                # Upper nibble
                q1 = (byte >> 4) & 0x0F
                output[out_idx] = d * q1
                out_idx += 1

        return output

    def _dequantize_q6_k(self, data: bytes, n_elements: int):
        """Dequantize Q6_K format to float32."""
        torch = self.torch
        block_size = 256
        n_blocks = (n_elements + block_size - 1) // block_size
        output = torch.zeros(n_elements, dtype=torch.float32)

        offset = 0
        out_idx = 0
        bytes_per_block = 210

        for _ in range(n_blocks):
            if out_idx >= n_elements:
                break

            block = data[offset:offset+bytes_per_block]
            if len(block) < bytes_per_block:
                break

            # Q6_K structure: ql (128), qh (64), scales (16), d (2)
            ql = block[0:128]
            qh = block[128:192]
            scales = block[192:208]
            d = struct.unpack('<e', block[208:210])[0]

            # Dequantize (simplified)
            for i in range(256):
                if out_idx >= n_elements:
                    break

                il = i % 128
                q_low = ql[il] if il < len(ql) else 0
                q_high = qh[il // 2] if il // 2 < len(qh) else 0

                if i < 128:
                    q = (q_low & 0x0F) | (((q_high >> (2 * (il % 2))) & 0x03) << 4)
                else:
                    q = ((q_low >> 4) & 0x0F) | (((q_high >> (2 * (il % 2) + 4)) & 0x03) << 4)

                q = q - 32
                sc = scales[i // 16] if i // 16 < len(scales) else 1
                output[out_idx] = d * sc * q / 16.0
                out_idx += 1

            offset += bytes_per_block

        return output


def create_llm_model(reader: GGUFReader):
    """Create and load an LLM model from GGUF file."""
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
    # Load model from GGUF
    # =========================================================================

    arch = reader.architecture
    print(f"Model architecture: {arch}")
    print(f"Model name: {reader.name}")

    def get_meta(key: str, default=None):
        full_key = f"{arch}.{key}" if arch else key
        return reader.metadata.get(full_key, reader.metadata.get(key, default))

    # Extract configuration
    dim = get_meta('embedding_length', 2048)
    n_layers = get_meta('block_count', 22)
    n_heads = get_meta('attention.head_count', 32)
    n_kv_heads = get_meta('attention.head_count_kv', n_heads)
    vocab_size = len(reader.metadata.get('tokenizer.ggml.tokens', [])) or 32000
    hidden_dim = get_meta('feed_forward_length', 4 * dim)
    norm_eps = get_meta('attention.layer_norm_rms_epsilon', 1e-5)
    max_seq_len = get_meta('context_length', 2048)

    print(f"Config: dim={dim}, layers={n_layers}, heads={n_heads}, kv_heads={n_kv_heads}")
    print(f"        vocab_size={vocab_size}, hidden_dim={hidden_dim}, max_seq_len={max_seq_len}")

    # Create model
    model = LLM(dim, n_layers, n_heads, n_kv_heads, vocab_size,
                hidden_dim, norm_eps, max_seq_len)

    # Load weights
    print("Loading weights...")
    loader = GGUFTensorLoader(reader)
    tensor_names = reader.get_tensor_names()
    loaded = 0

    name_mapping = {
        'token_embd.weight': 'tok_embeddings.weight',
        'output_norm.weight': 'norm.weight',
        'output.weight': 'output.weight',
    }

    for name in tensor_names:
        if name.startswith('blk.'):
            parts = name.split('.')
            layer_idx = int(parts[1])
            rest = '.'.join(parts[2:])

            layer_mapping = {
                'attn_q.weight': f'layers.{layer_idx}.attention.wq.weight',
                'attn_k.weight': f'layers.{layer_idx}.attention.wk.weight',
                'attn_v.weight': f'layers.{layer_idx}.attention.wv.weight',
                'attn_output.weight': f'layers.{layer_idx}.attention.wo.weight',
                'attn_norm.weight': f'layers.{layer_idx}.attention_norm.weight',
                'ffn_gate.weight': f'layers.{layer_idx}.feed_forward.w1.weight',
                'ffn_down.weight': f'layers.{layer_idx}.feed_forward.w2.weight',
                'ffn_up.weight': f'layers.{layer_idx}.feed_forward.w3.weight',
                'ffn_norm.weight': f'layers.{layer_idx}.ffn_norm.weight',
            }
            if rest in layer_mapping:
                name_mapping[name] = layer_mapping[rest]

    state_dict = model.state_dict()
    for gguf_name, pytorch_name in name_mapping.items():
        if gguf_name in tensor_names and pytorch_name in state_dict:
            try:
                tensor = loader.read_tensor_f32(gguf_name)
                if tensor.shape == state_dict[pytorch_name].shape:
                    state_dict[pytorch_name] = tensor
                    loaded += 1
                else:
                    print(f"  Shape mismatch for {pytorch_name}: "
                          f"expected {state_dict[pytorch_name].shape}, got {tensor.shape}")
            except Exception as e:
                print(f"  Failed to load {gguf_name}: {e}")

    model.load_state_dict(state_dict, strict=False)
    print(f"Loaded {loaded} tensors")

    return model


# =============================================================================
# Simple Tokenizer
# =============================================================================

class SimpleTokenizer:
    """Simple tokenizer using vocabulary from GGUF metadata."""

    def __init__(self, vocab: List[str], merges: Optional[List[str]] = None):
        self.vocab = vocab
        self.token_to_id = {token: i for i, token in enumerate(vocab)}
        self.id_to_token = {i: token for i, token in enumerate(vocab)}
        self.merges = merges or []

        self.bos_token_id = self.token_to_id.get('<s>', 1)
        self.eos_token_id = self.token_to_id.get('</s>', 2)
        self.pad_token_id = self.token_to_id.get('<pad>', 0)

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
                i += 1

        return tokens

    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text."""
        text = []
        for tid in token_ids:
            if tid in self.id_to_token:
                token = self.id_to_token[tid]
                if token in ['<s>', '</s>', '<pad>', '<unk>']:
                    continue
                if token.startswith('<0x') and token.endswith('>'):
                    try:
                        byte_val = int(token[3:-1], 16)
                        text.append(chr(byte_val))
                    except:
                        text.append(token)
                else:
                    token = token.replace('▁', ' ')
                    text.append(token)
        return ''.join(text)

    @classmethod
    def from_gguf(cls, reader: GGUFReader) -> 'SimpleTokenizer':
        """Create tokenizer from GGUF metadata."""
        vocab = reader.metadata.get('tokenizer.ggml.tokens', [])
        merges = reader.metadata.get('tokenizer.ggml.merges', [])

        if not vocab:
            vocab = ['<unk>', '<s>', '</s>'] + [chr(i) for i in range(256)]

        return cls(vocab, merges)


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
    """Find all GGUF model files in the models directory."""
    models_path = Path(models_dir)
    if not models_path.exists():
        return []
    return [str(p) for p in models_path.glob('*.gguf')]


def main():
    parser = argparse.ArgumentParser(description='GGUF Model Inference')
    parser.add_argument('--model', '-m', type=str, default=None,
                        help='Path to GGUF model file')
    parser.add_argument('--models-dir', type=str,
                        default=str(Path(__file__).parent.parent / 'models'),
                        help='Directory containing GGUF models')
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
            print("Available models:")
            for m in models:
                print(f"  - {m}")
            return

        if not models:
            print(f"No GGUF models found in {args.models_dir}")
            return

        model_path = models[0]
        print(f"Using model: {model_path}")

    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        return

    # Load GGUF (no PyTorch needed)
    print(f"\nLoading GGUF file: {model_path}")
    reader = GGUFReader(model_path)

    if args.info:
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
        return

    # From here, PyTorch is required
    print("\nLoading tokenizer...")
    tokenizer = SimpleTokenizer.from_gguf(reader)
    print(f"Vocabulary size: {len(tokenizer.vocab)}")

    print("\nLoading model (requires PyTorch)...")
    model = create_llm_model(reader)
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
