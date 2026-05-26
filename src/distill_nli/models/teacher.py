"""Teacher: FacebookAI/roberta-large-mnli wrapper.

Owns:
- frozen-load of the pretrained classifier (no fine-tuning)
- emitting logits in the same label order as the student head
  (HF roberta-large-mnli label2id is {contradiction:0, neutral:1, entailment:2})
- soft-target generation with temperature for distillation.
"""
