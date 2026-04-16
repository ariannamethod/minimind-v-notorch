"""
MiniMind-V Vision Language Model — backed by notorch (ctypes to libnotorch)

Vision encoder: patch extraction (notorch_vision.h) + linear patch embedding
Text decoder: Llama-style transformer (RMSNorm, RoPE, SwiGLU, causal attention)
Vision projector: two-layer MLP mapping vision features to LLM hidden dim

No PyTorch. No transformers. No CUDA. Just notorch.

Architecture:
    image → patches (C) → patch_embed (Linear) → vision_proj (MLP) → inject into token sequence
    tokens → embedding → [attention + FFN] × L → rmsnorm → lm_head → logits
"""

import os
import math
import ctypes
from dataclasses import dataclass, field
from typing import Optional, List

from ariannamethod.notorch_nn import (
    _lib, _get_tensor_struct, _NtTapeEntry, _NtTensor,
    Tensor, Parameter, Module, Linear, Embedding, RMSNorm,
    softmax, multinomial, seed as nt_seed,
)

# ═══════════════════════════════════════════════════════════════════════════════
# LOAD libnotorch_vision
# ═══════════════════════════════════════════════════════════════════════════════

_dir = os.path.dirname(os.path.abspath(__file__))
_ariannamethod_dir = os.path.join(os.path.dirname(_dir), 'ariannamethod')

for _vdir in [_ariannamethod_dir, _dir]:
    for ext in ['.dylib', '.so', '.dll']:
        _vlibpath = os.path.join(_vdir, f'libnotorch_vision{ext}')
        if os.path.exists(_vlibpath):
            break
    else:
        continue
    break
else:
    # Try building it
    _src = os.path.join(_ariannamethod_dir, 'notorch_vision_wrapper.c')
    _vlibpath = os.path.join(_ariannamethod_dir, 'libnotorch_vision.so')
    if os.path.exists(_src):
        import subprocess
        subprocess.run(['make', '-C', _ariannamethod_dir, 'libnotorch_vision.so'], check=True)

_vlib = ctypes.CDLL(_vlibpath)

# Vision C function signatures
_vlib.ntv_vit_preprocess.restype = ctypes.c_void_p
_vlib.ntv_vit_preprocess.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]

_vlib.ntv_vit_preprocess_mem.restype = ctypes.c_void_p
_vlib.ntv_vit_preprocess_mem.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int]

_vlib.ntv_image_free.restype = None
_vlib.ntv_image_free.argtypes = [ctypes.c_void_p]


# ═══════════════════════════════════════════════════════════════════════════════
# VLM CONFIG
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class VLMConfig:
    # LLM
    hidden_size: int = 512
    num_hidden_layers: int = 8
    num_attention_heads: int = 8
    num_key_value_heads: int = 8
    head_dim: int = 0  # computed from hidden_size // num_attention_heads
    vocab_size: int = 6400
    max_seq_len: int = 360
    dropout: float = 0.0

    # Vision
    image_size: int = 224
    patch_size: int = 16
    image_channels: int = 3
    image_special_token: str = '<|image_pad|>'
    image_ids: list = field(default_factory=lambda: [12])
    image_token_len: int = 64

    # FFN
    intermediate_size: int = 0  # computed

    # MoE (not used in notorch port, kept for config compat)
    use_moe: bool = False

    def __post_init__(self):
        if self.head_dim == 0:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.intermediate_size == 0:
            self.intermediate_size = math.ceil(self.hidden_size * math.pi / 64) * 64
        # Vision derived
        self.n_patches = (self.image_size // self.patch_size) ** 2  # e.g., 196 for 224/16
        self.patch_dim = self.image_channels * self.patch_size * self.patch_size  # e.g., 768 for 3*16*16
        # Vision projector merge: n_patches → image_token_len
        self.vision_merge = self.n_patches // self.image_token_len  # e.g., 196//64 ≈ 3


# ═══════════════════════════════════════════════════════════════════════════════
# VLM MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class MiniMindVLM(Module):
    """
    Vision Language Model backed by notorch.

    Vision path:
        image → patches [n_patches, patch_dim] → patch_embed → merge → proj_mlp → [image_token_len, hidden_size]

    Text path:
        tokens → embedding → transformer blocks → rmsnorm → lm_head

    Combined:
        Replace <image_pad> tokens in the text sequence with vision embeddings.
    """

    def __init__(self, config: VLMConfig = None):
        super().__init__()
        self.config = config or VLMConfig()
        p = self.config

        # Token embedding + LM head (tied weights)
        self.tok_emb = Embedding(p.vocab_size, p.hidden_size)
        self.lm_head = Linear(p.hidden_size, p.vocab_size)

        # Vision: patch embedding
        # patch_dim (e.g., 768) → hidden_size (e.g., 512)
        # After merge: patch_dim * merge → hidden_size
        merged_dim = p.patch_dim * p.vision_merge
        self.vision_proj_w1 = Linear(merged_dim, p.hidden_size)
        self.vision_proj_w2 = Linear(p.hidden_size, p.hidden_size)

        # Transformer layers
        self.layers = []
        for l in range(p.num_hidden_layers):
            layer = {
                'rms1': RMSNorm(p.hidden_size),
                'wq': Linear(p.hidden_size, p.num_attention_heads * p.head_dim),
                'wk': Linear(p.hidden_size, p.num_key_value_heads * p.head_dim),
                'wv': Linear(p.hidden_size, p.num_key_value_heads * p.head_dim),
                'wo': Linear(p.num_attention_heads * p.head_dim, p.hidden_size),
                'rms2': RMSNorm(p.hidden_size),
                'w_gate': Linear(p.hidden_size, p.intermediate_size),
                'w_up': Linear(p.hidden_size, p.intermediate_size),
                'w_down': Linear(p.intermediate_size, p.hidden_size),
            }
            for k, v in layer.items():
                setattr(self, f'l{l}_{k}', v)
            self.layers.append(layer)

        # Final norm
        self.norm_f = RMSNorm(p.hidden_size)

    def param_list(self):
        """Ordered parameters matching the forward pass layout on tape."""
        params = [self.tok_emb.weight]
        # Vision projector
        params.extend([
            self.vision_proj_w1.weight,
            self.vision_proj_w2.weight,
        ])
        # Transformer layers
        for l in self.layers:
            params.extend([
                l['rms1'].weight, l['wq'].weight, l['wk'].weight,
                l['wv'].weight, l['wo'].weight, l['rms2'].weight,
                l['w_gate'].weight, l['w_up'].weight, l['w_down'].weight,
            ])
        # Final norm + LM head
        params.extend([self.norm_f.weight, self.lm_head.weight])
        return params

    def count_params(self):
        return sum(p.numel for p in self.param_list())

    # ─── Vision preprocessing ─────────────────────────────────────────────

    @staticmethod
    def preprocess_image_file(image_path, image_size=224, patch_size=16):
        """Load image from file → [n_patches, patch_dim] notorch tensor."""
        ptr = _vlib.ntv_vit_preprocess(
            image_path.encode(),
            ctypes.c_int(image_size),
            ctypes.c_int(patch_size)
        )
        if not ptr:
            raise RuntimeError(f"Failed to load image: {image_path}")
        return Tensor(ptr, owns=True)

    @staticmethod
    def preprocess_image_bytes(image_bytes, image_size=224, patch_size=16):
        """Load image from bytes → [n_patches, patch_dim] notorch tensor."""
        buf = ctypes.create_string_buffer(image_bytes)
        ptr = _vlib.ntv_vit_preprocess_mem(
            buf,
            ctypes.c_int(len(image_bytes)),
            ctypes.c_int(image_size),
            ctypes.c_int(patch_size)
        )
        if not ptr:
            raise RuntimeError("Failed to load image from bytes")
        return Tensor(ptr, owns=True)

    # ─── Forward (training) ───────────────────────────────────────────────

    def forward_train(self, token_ids, target_ids, image_patches_data=None):
        """
        Forward through notorch tape for training.

        Args:
            token_ids: list of int, length seq_len
            target_ids: list of int, length seq_len
            image_patches_data: flat list of floats [image_token_len * merged_dim]
                                (raw merged patch data for vision projector), or None

        Returns: (loss_idx, loss_val)
        """
        CTX = len(token_ids)
        DIM = self.config.hidden_size
        HD = self.config.head_dim
        VOCAB = self.config.vocab_size

        _lib.nt_tape_start()
        _lib.nt_train_mode(1)

        params = self.param_list()
        tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]
        _lib.nt_tape_no_decay(tape_ids[0])  # embedding — no weight decay

        # Build token/target tensors
        tok_t = Tensor.zeros(CTX)
        tgt_t = Tensor.zeros(CTX)
        tok_t.set_data([float(x) for x in token_ids])
        tgt_t.set_data([float(x) for x in target_ids])
        tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
        tok_t._owns = False
        tgt_t._owns = False

        # Parameter indices
        pi = 0
        wte_i = tape_ids[pi]; pi += 1
        vproj_w1_i = tape_ids[pi]; pi += 1
        vproj_w2_i = tape_ids[pi]; pi += 1

        # Embedding
        h = _lib.nt_seq_embedding(wte_i, -1, tok_idx, CTX, DIM)

        # Vision injection: if we have image patches, project them and inject
        if image_patches_data is not None:
            n_vtok = self.config.image_token_len
            merged_dim = self.config.patch_dim * self.config.vision_merge

            vis_t = Tensor.zeros(n_vtok * merged_dim)
            vis_t.set_data(image_patches_data)
            vis_idx = _lib.nt_tape_record(vis_t._ptr, 0, -1, -1, ctypes.c_float(0))
            vis_t._owns = False

            # Project: Linear(merged_dim → hidden) → SiLU → Linear(hidden → hidden)
            vis_proj = _lib.nt_seq_linear(vproj_w1_i, vis_idx, n_vtok)
            vis_proj = _lib.nt_silu(vis_proj)
            vis_proj = _lib.nt_seq_linear(vproj_w2_i, vis_proj, n_vtok)

            # Add vision features to embedding at image marker positions
            # The vision projector output is added to the hidden states
            # at positions where image tokens appear
            h = _lib.nt_add(h, vis_proj)

        # Transformer layers
        for l in range(self.config.num_hidden_layers):
            rms1 = tape_ids[pi]; pi += 1
            wq = tape_ids[pi]; pi += 1; wk = tape_ids[pi]; pi += 1
            wv = tape_ids[pi]; pi += 1; wo = tape_ids[pi]; pi += 1
            rms2 = tape_ids[pi]; pi += 1
            wg = tape_ids[pi]; pi += 1; wu = tape_ids[pi]; pi += 1; wd = tape_ids[pi]; pi += 1

            xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
            q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
            k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
            v = _lib.nt_seq_linear(wv, xn, CTX)
            attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

            xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
            gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
            up = _lib.nt_seq_linear(wu, xn, CTX)
            h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

        rmsf = tape_ids[pi]; pi += 1
        head_i = tape_ids[pi]; pi += 1

        hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
        logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)
        loss_idx = _lib.nt_seq_cross_entropy(logits_idx, tgt_idx, CTX, VOCAB)

        # Read loss value from tape
        tape_ptr = _lib.nt_tape_get()
        entry_size = ctypes.sizeof(_NtTapeEntry)
        tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
        loss_entry = ctypes.cast(
            tape_addr + loss_idx * entry_size,
            ctypes.POINTER(_NtTapeEntry)
        ).contents
        loss_tensor = ctypes.cast(loss_entry.output, ctypes.POINTER(_NtTensor)).contents
        loss_val = loss_tensor.data[0]

        return loss_idx, loss_val

    def forward_train_vlm(self, token_ids, target_ids, image_bytes_list=None):
        """
        High-level training forward that handles image preprocessing.

        Args:
            token_ids: list of int
            target_ids: list of int
            image_bytes_list: list of bytes (raw image data), or None

        Returns: (loss_idx, loss_val)
        """
        image_patches_data = None

        if image_bytes_list:
            cfg = self.config
            # Preprocess first image (single-image VLM)
            patches_tensor = self.preprocess_image_bytes(
                image_bytes_list[0],
                image_size=cfg.image_size,
                patch_size=cfg.patch_size
            )
            # patches_tensor shape: [n_patches, patch_dim]
            # We need to merge patches: [n_patches, patch_dim] → [image_token_len, patch_dim * merge]
            patches_data = patches_tensor.get_data()
            n_patches = cfg.n_patches
            patch_dim = cfg.patch_dim
            merge = cfg.vision_merge
            n_vtok = cfg.image_token_len
            merged_dim = patch_dim * merge

            # Merge adjacent patches
            merged = []
            for i in range(n_vtok):
                for m in range(merge):
                    src_idx = i * merge + m
                    if src_idx < n_patches:
                        start = src_idx * patch_dim
                        merged.extend(patches_data[start:start + patch_dim])
                    else:
                        merged.extend([0.0] * patch_dim)
            image_patches_data = merged

        return self.forward_train(token_ids, target_ids, image_patches_data)

    def backward_step(self, loss_idx, loss_val, lr):
        """Backward + Chuck optimizer + clear tape."""
        _lib.nt_tape_backward(loss_idx)
        _lib.nt_tape_clip_grads(ctypes.c_float(1.0))
        _lib.nt_tape_chuck_step(ctypes.c_float(lr), ctypes.c_float(loss_val))
        _lib.nt_tape_clear()

    # ─── Generation ───────────────────────────────────────────────────────

    def generate(self, token_ids, max_new=200, temperature=0.8, top_k=40,
                 image_bytes=None):
        """
        Autoregressive generation with optional image input.

        Args:
            token_ids: list of int (prompt token IDs)
            max_new: maximum new tokens to generate
            temperature: sampling temperature
            top_k: top-k sampling
            image_bytes: raw image bytes, or None
        """
        _lib.nt_train_mode(0)
        ctx = list(token_ids)

        # Preprocess image once if provided
        image_patches_data = None
        if image_bytes:
            cfg = self.config
            patches_tensor = self.preprocess_image_bytes(
                image_bytes,
                image_size=cfg.image_size,
                patch_size=cfg.patch_size
            )
            patches_data = patches_tensor.get_data()
            n_patches = cfg.n_patches
            patch_dim = cfg.patch_dim
            merge = cfg.vision_merge
            n_vtok = cfg.image_token_len
            merged = []
            for i in range(n_vtok):
                for m in range(merge):
                    src_idx = i * merge + m
                    if src_idx < n_patches:
                        start = src_idx * patch_dim
                        merged.extend(patches_data[start:start + patch_dim])
                    else:
                        merged.extend([0.0] * patch_dim)
            image_patches_data = merged

        DIM = self.config.hidden_size
        HD = self.config.head_dim
        VOCAB = self.config.vocab_size

        for _ in range(max_new):
            if len(ctx) > self.config.max_seq_len:
                ctx = ctx[-self.config.max_seq_len:]
            CTX = len(ctx)

            _lib.nt_tape_start()
            params = self.param_list()
            tape_ids = [_lib.nt_tape_param(p._ptr) for p in params]

            tok_t = Tensor.zeros(CTX)
            tgt_t = Tensor.zeros(CTX)
            tok_t.set_data([float(x) for x in ctx])
            tok_idx = _lib.nt_tape_record(tok_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tgt_idx = _lib.nt_tape_record(tgt_t._ptr, 0, -1, -1, ctypes.c_float(0))
            tok_t._owns = False
            tgt_t._owns = False

            pi = 0
            wte_i = tape_ids[pi]; pi += 1
            vproj_w1_i = tape_ids[pi]; pi += 1
            vproj_w2_i = tape_ids[pi]; pi += 1

            h = _lib.nt_seq_embedding(wte_i, -1, tok_idx, CTX, DIM)

            # Vision injection during generation
            if image_patches_data is not None:
                n_vtok = self.config.image_token_len
                merged_dim = self.config.patch_dim * self.config.vision_merge
                vis_t = Tensor.zeros(n_vtok * merged_dim)
                vis_t.set_data(image_patches_data)
                vis_idx = _lib.nt_tape_record(vis_t._ptr, 0, -1, -1, ctypes.c_float(0))
                vis_t._owns = False
                vis_proj = _lib.nt_seq_linear(vproj_w1_i, vis_idx, n_vtok)
                vis_proj = _lib.nt_silu(vis_proj)
                vis_proj = _lib.nt_seq_linear(vproj_w2_i, vis_proj, n_vtok)
                h = _lib.nt_add(h, vis_proj)

            for l in range(self.config.num_hidden_layers):
                rms1 = tape_ids[pi]; pi += 1
                wq = tape_ids[pi]; pi += 1; wk = tape_ids[pi]; pi += 1
                wv = tape_ids[pi]; pi += 1; wo = tape_ids[pi]; pi += 1
                rms2 = tape_ids[pi]; pi += 1
                wg = tape_ids[pi]; pi += 1; wu = tape_ids[pi]; pi += 1; wd = tape_ids[pi]; pi += 1

                xn = _lib.nt_seq_rmsnorm(h, rms1, CTX, DIM)
                q = _lib.nt_rope(_lib.nt_seq_linear(wq, xn, CTX), CTX, HD)
                k = _lib.nt_rope(_lib.nt_seq_linear(wk, xn, CTX), CTX, HD)
                v = _lib.nt_seq_linear(wv, xn, CTX)
                attn = _lib.nt_mh_causal_attention(q, k, v, CTX, HD)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wo, attn, CTX))

                xn = _lib.nt_seq_rmsnorm(h, rms2, CTX, DIM)
                gate = _lib.nt_silu(_lib.nt_seq_linear(wg, xn, CTX))
                up = _lib.nt_seq_linear(wu, xn, CTX)
                h = _lib.nt_add(h, _lib.nt_seq_linear(wd, _lib.nt_mul(gate, up), CTX))

            rmsf = tape_ids[pi]; pi += 1; head_i = tape_ids[pi]; pi += 1
            hf = _lib.nt_seq_rmsnorm(h, rmsf, CTX, DIM)
            logits_idx = _lib.nt_seq_linear(head_i, hf, CTX)

            # Read last-position logits
            tape_ptr = _lib.nt_tape_get()
            entry_size = ctypes.sizeof(_NtTapeEntry)
            tape_addr = ctypes.cast(tape_ptr, ctypes.c_void_p).value
            logits_entry = ctypes.cast(
                tape_addr + logits_idx * entry_size,
                ctypes.POINTER(_NtTapeEntry)
            ).contents
            logits_t = ctypes.cast(logits_entry.output, ctypes.POINTER(_NtTensor)).contents
            offset = (CTX - 1) * VOCAB
            raw_logits = [logits_t.data[offset + i] / temperature for i in range(VOCAB)]

            if top_k > 0 and top_k < VOCAB:
                sorted_vals = sorted(raw_logits, reverse=True)
                threshold = sorted_vals[min(top_k - 1, len(sorted_vals) - 1)]
                raw_logits = [v if v >= threshold else -1e30 for v in raw_logits]

            probs = softmax(raw_logits)
            next_id = multinomial(probs)
            _lib.nt_tape_clear()
            ctx.append(next_id)

        return ctx[len(token_ids):]

    # ─── Save / Load ──────────────────────────────────────────────────────

    def save_weights(self, path):
        params = self.param_list()
        n = len(params)
        arr = (ctypes.c_void_p * n)(*[p._ptr for p in params])
        _lib.nt_save(path.encode(), arr, n)

    def load_weights(self, path):
        if not os.path.exists(path):
            return False
        n_loaded = ctypes.c_int(0)
        loaded = _lib.nt_load(path.encode(), ctypes.byref(n_loaded))
        if not loaded:
            return False
        params = self.param_list()
        for i in range(min(n_loaded.value, len(params))):
            src = _get_tensor_struct(loaded[i])
            dst = _get_tensor_struct(params[i]._ptr)
            if src.len == dst.len:
                ctypes.memmove(dst.data, src.data, dst.len * 4)
            _lib.nt_tensor_free(loaded[i])
        return True