"""Tests for FFN growth: forward equivalence pre-grow + one grow step."""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from distill_nli.growing.ffn import grow_step_ffn
from distill_nli.models.growing import make_student_growable


STUDENT_NAME = "FacebookAI/roberta-base"


requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine.",
)


@pytest.fixture(scope="module")
def device() -> torch.device:
    # Tests run on MPS to match the device the project actually trains on.
    return torch.device("mps")


@pytest.fixture
def student(device):
    # function-scope: surgery is destructive, so each test starts from a fresh model.
    config = AutoConfig.from_pretrained(STUDENT_NAME, num_labels=3)
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
def test_ffn_wrap_preserves_forward(student, batch):
    """After FFN surgery on all 12 layers, forward output is unchanged to fp32 tolerance."""
    logits_before = _forward(student, batch).clone()

    grow_cfg = {"ffn": {"enabled": True, "layers": "all"}}
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["ffn"]) == 12

    logits_after = _forward(student, batch)
    assert torch.allclose(logits_before, logits_after, atol=1e-5, rtol=1e-5), (
        f"max diff = {(logits_before - logits_after).abs().max().item():.3e}"
    )


@requires_mps
def test_ffn_grow_step_increases_intermediate_dim(student, batch, device):
    """One TINY-style grow step should add neurons to the bottleneck of each FFN."""
    grow_cfg = {"ffn": {"enabled": True, "layers": [0, 6]}}   # just two layers to keep test fast
    registry = make_student_growable(student, grow_cfg)
    ffns = registry["ffn"]
    assert all(ffn.intermediate_size == 3072 for ffn in ffns)

    def loss_fn(model, b):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        # Hard-label CE against arbitrary fake targets — what matters is having
        # a real grad signal, not what task the labels encode.
        targets = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
        return F.cross_entropy(out, targets)

    added = grow_step_ffn(
        student=student,
        growable_ffns=ffns,
        probe_batches=[batch, batch],   # 2 fake probe batches
        loss_fn=loss_fn,
        neurons_per_grow=8,
        alpha=1e-3,
        max_intermediate=6144,
    )
    # Each FFN should have added at least 1 neuron (gromo may add fewer than the cap
    # if statistical_threshold filters near-zero eigenvalues).
    assert set(added.keys()) == {0, 1}
    for i, n in added.items():
        assert n > 0, f"FFN {i} added {n} neurons (expected > 0)"
        assert ffns[i].intermediate_size == 3072 + n
