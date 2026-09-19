"""FastDecoder vs native Transformers on a tiny random Qwen3 (runs on CPU)."""
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "engine"))

from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

from fast import FastDecoder  # noqa: E402


def tiny_model(dtype):
    torch.manual_seed(0)
    cfg = Qwen3Config(
        vocab_size=512, hidden_size=64, intermediate_size=128, num_hidden_layers=3,
        num_attention_heads=8, num_key_value_heads=2, head_dim=16,
        max_position_embeddings=512, rope_theta=5_000_000.0, tie_word_embeddings=True,
    )
    return Qwen3ForCausalLM(cfg).to(dtype).eval()


def native(model, ids, n):
    cur, cache, out = torch.tensor(ids), None, []
    with torch.inference_mode():
        for _ in range(n):
            o = model(input_ids=cur, past_key_values=cache, use_cache=True, logits_to_keep=1)
            cur, cache = o.logits[:, -1].argmax(-1, keepdim=True), o.past_key_values
            out.append(cur[:, 0].tolist())
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_matches_native(dtype):
    model = tiny_model(dtype)
    fast = FastDecoder(model)
    for seed, (b, s, n) in enumerate([(1, 40, 9), (3, 40, 9), (3, 40, 9), (2, 17, 5), (1, 1, 1)]):
        g = torch.Generator().manual_seed(seed)
        ids = torch.randint(0, 512, (b, s), generator=g).tolist()
        got, want = list(fast.generate(ids, n)), native(model, ids, n)
        assert len(got) == n
        if dtype == torch.float32:
            assert got == want, (b, s, n)
        else:
            assert got[0] == want[0]
            assert sum(x == y for x, y in zip(got, want)) >= n - 1
