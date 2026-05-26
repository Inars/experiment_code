"""TINY-style growing step, ported from MNIST_TINY_vs_Big.ipynb.

Reference:
- Verbockhaven et al., "Growing Tiny Networks: Spotting Expressivity Bottlenecks and
  Fixing Them Optimally" — https://openreview.net/pdf?id=hbtG6s6e7r
- Code: https://gitlab.inria.fr/mverbock/tinypub/-/blob/main/Paper/MNIST/MNIST_TINY_vs_Big.ipynb

Owns:
- accumulation of the per-layer natural-gradient statistics on a probe batch
- selection of where (which FFN layer) and how many neurons to add
- computation of the new weights for added neurons (init that locally
  optimizes the loss, per the paper)
- handing the expansion off to models/growing.py for in-place application.
"""
