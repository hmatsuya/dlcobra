"""Thin wrapper that fixes mlp_expansion_dim=1536 (25% pruned from 2048).

The ptl.py factory calls PolicyValueNetwork() with no arguments for
user-defined (dotted) network paths.  This wrapper lets us pass the pruned
expansion dim without modifying the factory or ptl.py.

If you want a different prune ratio, change PRUNED_MLP_DIM here and re-run
prune.py with the matching --prune-ratio.

    prune_ratio=0.10 → PRUNED_MLP_DIM = 1840  (77.8M params)
    prune_ratio=0.25 → PRUNED_MLP_DIM = 1536  (65.7M params)  ← default
    prune_ratio=0.40 → PRUNED_MLP_DIM = 1224  (53.2M params)
    prune_ratio=0.50 → PRUNED_MLP_DIM = 1024  (45.2M params)
"""
from dlshogi.experiments.exp032_structured_pruning.model import PolicyValueNetwork

# Must match the n_keep value printed by prune.py
PRUNED_MLP_DIM = 1536


class PrunedPolicyValueNetwork(PolicyValueNetwork):
    """PolicyValueNetwork with mlp_expansion_dim fixed to PRUNED_MLP_DIM."""

    def __init__(self):
        super().__init__(mlp_expansion_dim=PRUNED_MLP_DIM)
