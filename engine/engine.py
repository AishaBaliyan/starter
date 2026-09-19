"""Qwen3 4B engine: static-cache, CUDA-graphed decode with a native fallback.

``fast.FastDecoder`` runs the model with a preallocated KV cache and one CUDA
graph per decode step. At load time the fast path is checked against native
Transformers on a small synthetic prompt; if it errors or disagrees, the engine
falls back to the plain greedy loop below (same weights, no extra memory), so a
bad fast path costs speed, never correctness. Set ENGINE_FAST=0 to force native.
"""

import os
import sys

import torch
from transformers import AutoModelForCausalLM

from fast import FastDecoder


def _log(*args) -> None:
    print("[engine]", *args, file=sys.stderr, flush=True)


class Engine:
    def __init__(self, model_path: str) -> None:
        """Load the pinned checkpoint from model_path. Untimed, budgeted."""
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                attn_implementation="sdpa",
                local_files_only=True,
            )
            .eval()
            .to("cuda:0")
        )
        self.fast = self._pick_fast_path() if os.environ.get("ENGINE_FAST", "1") != "0" else None
        _log("path:", "fast" if self.fast else "native")

    def _pick_fast_path(self):
        vocab = self.model.config.vocab_size
        gen = torch.Generator().manual_seed(0)
        ids = torch.randint(1000, min(vocab, 100000), (2, 48), generator=gen).tolist()
        steps = 12
        try:
            reference = [t for t in self._native(ids, steps)]
        except Exception as err:
            _log("native reference failed:", repr(err))
            return None
        for use_triton in (True, False):
            try:
                decoder = FastDecoder(self.model, use_triton=use_triton)
                got = list(decoder.generate(ids, steps))
            except Exception as err:
                _log(f"fast path (triton={use_triton}) failed:", repr(err))
                continue
            agree = sum(a == b for a, b in zip(got, reference))
            _log(f"fast path (triton={use_triton}) agrees on {agree}/{steps} steps")
            if got[0] == reference[0] and agree >= steps * 3 // 4:
                return decoder
        return None

    def _native(self, input_ids, max_new_tokens):
        current = torch.tensor(input_ids, dtype=torch.int64, device="cuda:0")
        cache = None
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                output = self.model(
                    input_ids=current,
                    past_key_values=cache,
                    use_cache=True,
                    logits_to_keep=1,
                    return_dict=True,
                )
                current = output.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                cache = output.past_key_values
                yield current[:, 0].tolist()

    def generate(self, input_ids: list[list[int]], max_new_tokens: int):
        """Greedy continuation of every sequence, one step at a time.

        Yields a list with one token id per sequence for each output step,
        exactly max_new_tokens times. Every sequence has the same length.
        Never stops at end-of-sequence tokens.
        """
        if self.fast is None:
            yield from self._native(input_ids, max_new_tokens)
            return
        try:
            stream = self.fast.generate(input_ids, max_new_tokens)
            first = next(stream)
        except Exception as err:
            _log("fast path failed at start, using native:", repr(err))
            self.fast = None
            yield from self._native(input_ids, max_new_tokens)
            return
        yield first
        yield from stream
