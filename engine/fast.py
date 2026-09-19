"""Hand-rolled Qwen3 forward: static KV cache, no Transformers dispatch, CUDA-graphed decode.

Uses the loaded Transformers modules only as weight holders (and for the exact
torch ops of the MLP and RMSNorm), so tied weights and dtypes are preserved.

Layout: per-layer K and V are ``[B, 8, cap, 128]``, preallocated once per
(batch, prompt length, output length). Decode attends over the full capacity
with a device-computed mask, so every decode step has identical shapes and the
whole step is one CUDA graph. The four query heads that share a KV head are
folded into the query-length dimension, so K/V are never expanded 4x.
"""

import torch
import torch.nn.functional as F

try:
    from kernels.rmsnorm import rms_norm as _triton_rms_norm
except Exception:  # no triton (CPU test box) or import failure
    _triton_rms_norm = None


def _rotate_half(x):
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class FastDecoder:
    def __init__(self, model, use_triton: bool = False) -> None:
        cfg = model.config
        self.model = model
        self.base = model.model
        self.lm_head = model.lm_head
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        self.n_heads = cfg.num_attention_heads
        self.n_kv = cfg.num_key_value_heads
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
        self.group = self.n_heads // self.n_kv
        self.use_triton = bool(use_triton and _triton_rms_norm is not None and self.device.type == "cuda")
        self.use_graph = self.device.type == "cuda"
        self.shape = None
        self.graph = None
        self._merge_projections()

    def _merge_projections(self):
        """One weight per fused GEMM (qkv, gate+up), as views over the originals.

        The Transformers modules keep working (their weights alias the merged
        buffer), so the native fallback needs no second copy of the weights.
        """
        self.w_qkv, self.w_gu = [], []
        for layer in self.base.layers:
            attn, mlp = layer.self_attn, layer.mlp
            for owner, names, out in (
                (attn, ("q_proj", "k_proj", "v_proj"), self.w_qkv),
                (mlp, ("gate_proj", "up_proj"), self.w_gu),
            ):
                mods = [getattr(owner, n) for n in names]
                merged = torch.cat([m.weight.data for m in mods], dim=0).contiguous()
                start = 0
                for m in mods:
                    rows = m.weight.shape[0]
                    m.weight.data = merged[start : start + rows]
                    start += rows
                out.append(merged)
        self.q_size = self.n_heads * self.head_dim
        self.kv_size = self.n_kv * self.head_dim

    # ------------------------------------------------------------------ ops
    def _norm(self, module, x):
        if self.use_triton:
            return _triton_rms_norm(x, module.weight, module.variance_epsilon)
        return module(x)

    def _layer(self, i, layer, x, cos, sin, prefill):
        B, T, _ = x.shape
        attn = layer.self_attn
        h = self._norm(layer.input_layernorm, x)
        q, k, v = F.linear(h, self.w_qkv[i]).split((self.q_size, self.kv_size, self.kv_size), dim=-1)
        q = self._norm(attn.q_norm, q.view(B, T, self.n_heads, self.head_dim)).transpose(1, 2)
        k = self._norm(attn.k_norm, k.view(B, T, self.n_kv, self.head_dim)).transpose(1, 2)
        v = v.reshape(B, T, self.n_kv, self.head_dim).transpose(1, 2)
        q = q * cos + _rotate_half(q) * sin
        k = k * cos + _rotate_half(k) * sin

        kc, vc = self.k_cache[i], self.v_cache[i]
        if prefill:
            kc[:, :, :T].copy_(k)
            vc[:, :, :T].copy_(v)
            kf = k[:, :, None].expand(B, self.n_kv, self.group, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
            vf = v[:, :, None].expand(B, self.n_kv, self.group, T, self.head_dim).reshape(B, self.n_heads, T, self.head_dim)
            o = F.scaled_dot_product_attention(q, kf, vf, is_causal=True)
            o = o.transpose(1, 2).reshape(B, T, self.n_heads * self.head_dim)
        else:
            kc.index_copy_(2, self.pos, k)
            vc.index_copy_(2, self.pos, v)
            q4 = q.reshape(B, self.n_kv, self.group, self.head_dim)
            o = F.scaled_dot_product_attention(q4, kc, vc, attn_mask=self.mask)
            o = o.reshape(B, 1, self.n_heads * self.head_dim)

        x = x + attn.o_proj(o)
        gate, up = F.linear(self._norm(layer.post_attention_layernorm, x), self.w_gu[i]).chunk(2, dim=-1)
        return x + layer.mlp.down_proj(layer.mlp.act_fn(gate) * up)

    def _forward(self, x, cos, sin, prefill):
        for i, layer in enumerate(self.base.layers):
            x = self._layer(i, layer, x, cos, sin, prefill)
        x = self._norm(self.base.norm, x[:, -1])
        return self.lm_head(x).argmax(dim=-1)

    def _step(self):
        """One decode step from ``tok`` at position ``pos``; all state in place."""
        x = self.base.embed_tokens(self.tok[:, None])
        cos = self.cos_tab.index_select(0, self.pos).view(1, 1, 1, self.head_dim)
        sin = self.sin_tab.index_select(0, self.pos).view(1, 1, 1, self.head_dim)
        self.mask.copy_(((self.arange > self.pos).to(self.dtype) * self.neg).view(1, 1, 1, -1))
        nxt = self._forward(x, cos, sin, prefill=False)
        self.tok.copy_(nxt)
        self.pos += 1

    # ---------------------------------------------------------------- setup
    def _setup(self, batch, prompt_len, max_new):
        self.release()
        dev, cap = self.device, ((prompt_len + max_new + 63) // 64) * 64
        self.shape = (batch, prompt_len, max_new)
        shape = (batch, self.n_kv, cap, self.head_dim)
        n = len(self.base.layers)
        self.k_cache = [torch.zeros(shape, dtype=self.dtype, device=dev) for _ in range(n)]
        self.v_cache = [torch.zeros(shape, dtype=self.dtype, device=dev) for _ in range(n)]
        with torch.inference_mode():
            positions = torch.arange(cap, device=dev)[None]
            dummy = torch.zeros(1, cap, self.head_dim, dtype=self.dtype, device=dev)
            cos, sin = self.base.rotary_emb(dummy, positions)
        self.cos_tab, self.sin_tab = cos[0].clone(), sin[0].clone()
        self.arange = torch.arange(cap, device=dev)
        self.neg = torch.finfo(self.dtype).min
        self.pos = torch.zeros(1, dtype=torch.int64, device=dev)
        self.tok = torch.zeros(batch, dtype=torch.int64, device=dev)
        self.mask = torch.zeros(batch, 1, 1, cap, dtype=self.dtype, device=dev)
        self.out_dev = torch.zeros(max_new, batch, dtype=torch.int64, device=dev)
        if self.use_graph:
            self.out_host = torch.zeros(max_new, batch, dtype=torch.int64).pin_memory()
            self.events = [torch.cuda.Event() for _ in range(max_new)]
            self._capture()

    def _capture(self):
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                self._step()
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self._step()
        torch.cuda.synchronize()

    def release(self):
        self.graph = None
        self.shape = None
        for name in ("k_cache", "v_cache", "out_dev", "out_host", "events", "mask"):
            if hasattr(self, name):
                delattr(self, name)
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    # ------------------------------------------------------------- generate
    @torch.inference_mode()
    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        batch, prompt_len = len(input_ids), len(input_ids[0])
        if self.shape != (batch, prompt_len, max_new_tokens):
            self._setup(batch, prompt_len, max_new_tokens)

        ids = torch.tensor(input_ids, dtype=torch.int64, device=self.device)
        x = self.base.embed_tokens(ids)
        cos = self.cos_tab[:prompt_len][None, None]
        sin = self.sin_tab[:prompt_len][None, None]
        self.tok.copy_(self._forward(x, cos, sin, prefill=True))
        self.pos.fill_(prompt_len)
        self.out_dev[0].copy_(self.tok)

        for i in range(max_new_tokens):
            if self.use_graph:
                self.out_host[i].copy_(self.out_dev[i], non_blocking=True)
                self.events[i].record()
            if i + 1 < max_new_tokens:
                if self.graph is not None:
                    self.graph.replay()
                else:
                    self._step()
                self.out_dev[i + 1].copy_(self.tok)
            if self.use_graph:
                self.events[i].synchronize()
                yield self.out_host[i].tolist()
            else:
                yield self.out_dev[i].tolist()
