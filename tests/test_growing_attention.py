"""Tests for attention growth: forward equivalence pre-grow.

Grow-step tests come in Phase 3c.
"""

from __future__ import annotations

import pytest
import torch
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from distill_nli.models.growing import make_student_growable


STUDENT_NAME = "FacebookAI/roberta-base"


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine.",
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    return torch.device("mps")


@pytest.fixture
def student(device):
    # function-scope: attention surgery is destructive.
    config = AutoConfig.from_pretrained(STUDENT_NAME, num_labels=3)
    # Force eager attention so per-head fp32 softmax matches HF's eager_attention_forward
    # bit-for-bit. (SDPA on MPS may reorder reductions and break a tight allclose.)
    config._attn_implementation = "eager"
    model = AutoModelForSequenceClassification.from_pretrained(STUDENT_NAME, config=config)
    model.to(device).eval()
    return model


@pytest.fixture(scope="module")
def tokenizer():
    return AutoTokenizer.from_pretrained(STUDENT_NAME)


@pytest.fixture(scope="module")
def batch(tokenizer, device):
    enc = tokenizer(
        ["A man eats a sandwich.", "The cat sleeps on the mat."],
        ["A person consumes food.", "An animal rests."],
        return_tensors="pt", padding=True, truncation=True, max_length=32,
    )
    return {k: v.to(device) for k, v in enc.items()}


def _forward(model, batch) -> torch.Tensor:
    with torch.no_grad():
        out = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"])
    return out.logits


@requires_mps
def test_attention_wrap_preserves_forward(student, batch):
    """After wrapping attention.self in all 12 layers, logits match pre-surgery to fp32 tolerance."""
    logits_before = _forward(student, batch).clone()

    grow_cfg = {"attention": {"enabled": True, "layers": "all"}}
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["attention"]) == 12

    logits_after = _forward(student, batch)
    max_diff = (logits_before - logits_after).abs().max().item()
    assert torch.allclose(logits_before, logits_after, atol=1e-4, rtol=1e-4), (
        f"max diff = {max_diff:.3e}"
    )


@requires_mps
def test_attention_and_ffn_wrap_together_preserves_forward(student, batch):
    """Wrapping BOTH attention and FFN in all 12 layers preserves forward output."""
    logits_before = _forward(student, batch).clone()

    grow_cfg = {
        "ffn": {"enabled": True, "layers": "all"},
        "attention": {"enabled": True, "layers": "all"},
    }
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["ffn"]) == 12
    assert len(registry["attention"]) == 12

    logits_after = _forward(student, batch)
    max_diff = (logits_before - logits_after).abs().max().item()
    assert torch.allclose(logits_before, logits_after, atol=1e-4, rtol=1e-4), (
        f"max diff = {max_diff:.3e}"
    )
