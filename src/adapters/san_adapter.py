"""
san_adapter.py
==============
Maps the shared PE cache into SAN's (Kreuzer et al., 2021) input format.

SAN natively expects a Learned Positional Encoding (LPE): raw Laplacian eigenvectors +
eigenvalues, passed through a small Transformer encoder, then added to node features. That
is exactly our `lap_pe`/`lap_eigvals` cache fields, so LapPE is a near-faithful drop-in.

- No-PE: disable SAN's LPE module entirely (feed zeros / skip the add).
- LapPE: native fit, SAN's own LPE encoder consumes `lap_pe` + `lap_eigvals` unchanged.
- RWSE: not part of SAN's original design. We concatenate `rwse` to the node input features
  alongside (or instead of, per config flag) the LPE output -- a straightforward feature-
  level extension, no architecture change needed.
- SignNet-PE: replaces SAN's own LPE encoder with the SignNetEncoder from
  `src/pe/compute_pe.py` (both operate on the same raw eigenvectors, so this is a clean
  swap of "how do we make eigenvectors sign-invariant", not a structural change).
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build_san_config(pe_name: str, cache_dir: str) -> dict:
    base = {
        "lpe_enable": False,
        "extra_node_feat": None,     # e.g. "rwse" to concat
        "signnet_replaces_lpe": False,
        "cache_dir": cache_dir,
    }
    if pe_name == "none":
        return base
    if pe_name == "lappe":
        return {**base, "lpe_enable": True}
    if pe_name == "rwse":
        return {**base, "lpe_enable": True, "extra_node_feat": "rwse"}
    if pe_name == "signnet":
        return {**base, "lpe_enable": True, "signnet_replaces_lpe": True}
    raise ValueError(f"Unknown pe_name: {pe_name}")

