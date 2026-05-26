"""pytest configuration for distill_nli.

Sets PYTORCH_ENABLE_MPS_FALLBACK=1 BEFORE torch is imported anywhere. Required
because torch 2.12's MPS backend lacks `torch.linalg.eigh`, which gromo's growth
math calls inside compute_optimal_updates. The fallback routes the unsupported
op (only) to CPU while forward/backward stay on MPS.
"""

import os


os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
