"""Student: FacebookAI/roberta-base with a 3-way NLI classification head.

Owns:
- backbone load from `FacebookAI/roberta-base`
- a fresh classification head matching the teacher's label order
- a clean public interface for `growing.py` to swap FFN sub-modules in/out.
"""
