"""Bridge between the student RoBERTa-base and gromo growing modules.

This is the ONLY module that touches gromo internals. It:
- wraps target FFN blocks of the student with gromo growing containers
- exposes a snapshot/restore API so a grow step is atomic
- preserves forward-pass equivalence pre-grow (verified by tests/test_growing_wrap.py).

gromo lives at ../gromo as an editable install and must not be modified.
"""
