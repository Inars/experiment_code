"""Distillation losses.

- soft target KL/CE between (teacher_logits / T) and (student_logits / T), scaled by T^2
- hard label cross-entropy against gold MNLI labels
- convex combination: L = alpha * L_soft + (1 - alpha) * L_hard
"""
