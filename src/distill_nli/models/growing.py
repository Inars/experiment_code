"""Surgery: in-place replacement of HF RoBERTa modules with growable variants.

This is the ONLY place in the project that touches HF's RobertaLayer internals.
Two surgeries:
- FFN: layer.intermediate + layer.output.dense -> distill_nli.growing.ffn.wrap_ffn(...)
- Attention: layer.attention.self -> distill_nli.growing.attention.wrap_attention(...)

Each surgery is independently toggleable via configs/grow.yaml (ffn.enabled,
attention.enabled). Both must preserve forward-pass output to floating-point
tolerance before the first growth step — verified by tests/test_growing_wrap.py.

gromo (../gromo) provides the FFN growth primitives and stays read-only.
The attention port lives entirely in distill_nli.growing.attention.

Public API (to be implemented across Phases 2-3):
- make_student_growable(student, grow_cfg) -> dict
    Walks student.roberta.encoder.layer, applies the requested surgeries per
    grow_cfg, and returns a registry of growable modules grouped by kind:
        {"ffn": [GrowableRobertaFFN, ...], "attention": [GrowableRobertaSelfAttention, ...]}
"""

from __future__ import annotations
