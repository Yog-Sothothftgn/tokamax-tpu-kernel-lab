"""WP-KV5 prep (2026-09-14): closes a gap flagged earlier in this project --
`kimi_k3_latent_moe_reference.py`'s `kimi_k3_config()` is a hand-typed
`LatentMoEConfig`, kept in sync with the verified official snapshot
(`official_kimi_k3/config.json`, pinned/hashed by `validate_official_config.py`)
only by manual review, not by an automated check. A full-dimension smoke test
built on top of `kimi_k3_config()` is only as trustworthy as that config
actually matching the real model -- this script is the missing link, mirroring
`test_jax_reference_against_pytorch_golden.py`'s existing
`_assert_config_matches_toy_config` pattern (golden bundle's config.json vs.
`toy_config()`) for the real, full-dimension case (official snapshot vs.
`kimi_k3_config()`) that was never actually built.

CPU-only, no tokamax/TPU needed -- pure Python/json plus importing
`kimi_k3_config()` (which itself has no tokamax import at module scope beyond
what kimi_k3_latent_moe_reference.py already needs). Safe to run anywhere.

Usage:
  python verify_kimi_k3_config_matches_official.py
"""

import json
import pathlib
import sys

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "05_ragged_dot_on_tpu"))

from kimi_k3_latent_moe_reference import kimi_k3_config  # noqa: E402
from validate_official_config import _find_field  # noqa: E402

OFFICIAL_CONFIG_PATH = _HERE / "official_kimi_k3" / "config.json"

# Same field-name mapping as test_jax_reference_against_pytorch_golden.py's
# _assert_config_matches_toy_config -- the official snapshot's field names
# (left) don't all match LatentMoEConfig's attribute names (right) 1:1.
FIELD_MAPPING = {
    "hidden_size": "hidden_size",
    "routed_expert_hidden_size": "latent_size",
    "moe_intermediate_size": "intermediate_size",
    "num_experts": "num_experts",
    "num_experts_per_token": "top_k",
    "num_shared_experts": "num_shared_experts",
    "moe_renormalize": "moe_renormalize",
    "routed_scaling_factor": "routed_scaling_factor",
    "rms_norm_eps": "rms_norm_eps",
    "activation_situ_beta": "activation_situ_beta",
    "activation_situ_linear_beta": "activation_situ_linear_beta",
}


def main() -> bool:
  if not OFFICIAL_CONFIG_PATH.exists():
    raise FileNotFoundError(
        f"{OFFICIAL_CONFIG_PATH} not found -- run validate_official_config.py first "
        "to download and pin the official snapshot."
    )
  with OFFICIAL_CONFIG_PATH.open(encoding="utf-8") as f:
    official = json.load(f)

  config = kimi_k3_config()

  print("[verify-kimi-k3-config] checking kimi_k3_config() against the official snapshot:")
  all_ok = True
  for official_field, config_attr in FIELD_MAPPING.items():
    expected = _find_field(official, official_field)
    actual = getattr(config, config_attr)
    ok = actual == expected
    print(
        f"  {config_attr} (official field {official_field!r}): "
        f"expected={expected!r} actual={actual!r} [{'OK' if ok else 'FAIL'}]"
    )
    all_ok = all_ok and ok

  print(f"[verify-kimi-k3-config] {'ALL FIELDS MATCH' if all_ok else 'MISMATCH -- fix before trusting kimi_k3_config()'}")
  return all_ok


if __name__ == "__main__":
  ok = main()
  sys.exit(0 if ok else 1)
