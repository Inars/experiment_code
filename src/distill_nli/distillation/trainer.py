"""Training loop: distillation with optional TINY-style growing steps.

Owns:
- per-step student fwd/bwd against the mixed distillation loss
- periodic eval on MNLI validation matched/mismatched
- grow-step hook called at a schedule from configs/distill.yaml (delegates to
  growing/tiny.py to compute the natural-gradient expansion and to models/growing.py
  to apply it).
"""
