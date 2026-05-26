"""SetFit/mnli loading, tokenization, and dataloader construction.

Owns:
- HF dataset load (SetFit/mnli, splits: train, validation, test)
- premise/hypothesis tokenization with the student tokenizer
- collation and DataLoader factories
- label-id mapping that aligns with the teacher (roberta-large-mnli uses
  0=contradiction, 1=neutral, 2=entailment).
"""
