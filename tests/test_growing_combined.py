"""End-to-end test: one combined grow_step that grows BOTH attention and FFN.

Covers the integration surface that the training scripts will hit in Phase 4.
"""

from __future__ import annotations

import pytest
import torch
from torch.nn import functional as F
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

from distill_nli.growing import grow_step
from distill_nli.models.growing import make_student_growable


STUDENT_NAME = "FacebookAI/roberta-base"

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(),
    reason="MPS not available on this machine.",
)


@pytest.fixture(scope="module")
def device():
    return torch.device("mps")


@pytest.fixture
def student(device):
    config = AutoConfig.from_pretrained(STUDENT_NAME, num_labels=3)
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


@requires_mps
def test_grow_step_runs_both_growers(student, batch):
    """grow_step orchestrates attention + FFN in one call and reports both."""
    grow_cfg = {
        "ffn": {
            "enabled": True,
            "layers": [3],          # one layer is enough to test orchestration
            "neurons_per_grow": 4,
            "alpha": 1e-3,
            "max_intermediate": 6144,
        },
        "attention": {
            "enabled": True,
            "layers": [3],
            "p_per_grow": 2,
            "max_k_per_head": 128,
            "top_k": 3,
            "lambda_reg": 1e-3,
            "alpha": 1e-3,
            "adaptive_tau": 1e-4,
            "cg_tol": 1e-4,
            "cg_max_iter": 15,
            "precision": "float64",
        },
    }
    registry = make_student_growable(student, grow_cfg)
    assert len(registry["ffn"]) == 1
    assert len(registry["attention"]) == 1

    # Snapshot sizes before grow.
    pre_intermediate = registry["ffn"][0].intermediate_size
    pre_k_dims = [h.k_dim for h in registry["attention"][0].heads]

    def loss_fn(model, b):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        targets = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
        return F.cross_entropy(out, targets)

    report = grow_step(
        student=student,
        registry=registry,
        probe_batches=[batch, batch],
        loss_fn=loss_fn,
        grow_cfg=grow_cfg,
    )

    # FFN report: dict mapping ffn index -> neurons added.
    assert report["ffn"] is not None and report["ffn"].get(0, 0) > 0, (
        f"expected FFN 0 to grow, got {report['ffn']}"
    )
    assert registry["ffn"][0].intermediate_size > pre_intermediate

    # Attention report: top_k heads applied with k_added == p_per_grow.
    assert report["attention"] is not None
    applied = [k for k, v in report["attention"].items() if v["applied"]]
    assert len(applied) == grow_cfg["attention"]["top_k"]
    for (li, hi) in applied:
        assert registry["attention"][0].heads[hi].k_dim == pre_k_dims[hi] + grow_cfg["attention"]["p_per_grow"]


@requires_mps
def test_grow_step_respects_disabled_flags(student, batch):
    """If only attention is enabled, FFN side of the report stays None."""
    grow_cfg = {
        "ffn":       {"enabled": False, "layers": "all"},
        "attention": {
            "enabled": True, "layers": [3], "p_per_grow": 2, "top_k": 1,
            "lambda_reg": 1e-3, "alpha": 1e-3, "adaptive_tau": 1e-4,
            "cg_tol": 1e-4, "cg_max_iter": 10, "precision": "float64",
            "max_k_per_head": 128,
        },
    }
    registry = make_student_growable(student, grow_cfg)
    assert registry["ffn"] == []
    assert len(registry["attention"]) == 1

    def loss_fn(model, b):
        out = model(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits
        targets = torch.zeros(out.size(0), dtype=torch.long, device=out.device)
        return F.cross_entropy(out, targets)

    report = grow_step(
        student=student,
        registry=registry,
        probe_batches=[batch],
        loss_fn=loss_fn,
        grow_cfg=grow_cfg,
    )
    assert report["ffn"] is None
    assert report["attention"] is not None
