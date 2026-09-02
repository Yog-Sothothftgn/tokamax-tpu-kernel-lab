"""WP-KV1 (KIMI_K3_LATENT_MOE_VALIDATION_PLAN_2026-08-26.md): lock the
official Kimi K3 config + modeling source, hash them, and validate the
LatentMoE-relevant fields against known values -- so the model structure
this project's JAX/TPU prototype implements is checked automatically, not
by re-reading the source by hand (which has needed correction multiple
times already in this project's history).

Pins a specific HuggingFace commit SHA (confirmed via the HF API
2026-08-26: a590ce090cb049c93a33dfe8c208ec652aa20503, moonshotai/Kimi-K3,
lastModified 2026-08-20) rather than `main`, so this doesn't silently drift
if the upstream repo changes. Re-run with a different --commit to check
against a newer revision deliberately.

This script has NO tokamax/TPU dependency -- plain Python stdlib
(urllib/hashlib/json) plus jax only for an optional environment fingerprint
(best-effort, doesn't fail the check if jax isn't importable). Safe to run
anywhere: this machine, the v6e TPU VM, or any other host with network
access to huggingface.co.

Usage:
  python validate_official_config.py
  python validate_official_config.py --commit <other-sha>
"""

import argparse
import hashlib
import json
import pathlib
import subprocess
import sys
import urllib.request

REPO = "moonshotai/Kimi-K3"
DEFAULT_COMMIT = "a590ce090cb049c93a33dfe8c208ec652aa20503"
OUT_DIR = pathlib.Path(__file__).parent / "official_kimi_k3"

FILES = ("config.json", "modeling_kimi_linear.py", "configuration_kimi_k3.py")

# Confirmed by direct fetch of config.json's `text_config` on 2026-08-26 (see
# project memory / this session's feasibility pre-check) -- update this dict
# (with a dated comment explaining why) if the pinned commit ever changes and
# a field legitimately differs.
REQUIRED_FIELDS = {
    "hidden_size": 7168,
    "moe_intermediate_size": 3072,
    "num_experts": 896,
    "num_experts_per_token": 16,
    "num_shared_experts": 2,
    "moe_renormalize": True,
    "moe_router_activation_func": "sigmoid",
    "activation_situ_beta": 4.0,
    "activation_situ_linear_beta": 25.0,
    "dtype": "bfloat16",
    "routed_expert_hidden_size": 3584,
    "rms_norm_eps": 1e-5,
    "routed_scaling_factor": 1.0,
    "hidden_act": "situ",
    # Added 2026-08-26 per user request, confirmed against the real
    # modeling_kimi_linear.py source (KimiMoEGate's grouped-topk branch is
    # gated on num_expert_group>1 and num_expert_group>topk_group -- both
    # being 1 confirms that branch is dead code for Kimi K3's actual config,
    # not just unused-by-convention).
    "latent_moe_use_norm": True,
    "num_expert_group": 1,
    "topk_group": 1,
    "first_k_dense_replace": 1,
    "moe_layer_freq": 1,
    "num_hidden_layers": 93,
}


def _download(url: str, dest: pathlib.Path) -> bytes:
  with urllib.request.urlopen(url) as resp:
    data = resp.read()
  dest.write_bytes(data)
  return data


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _find_field(config: dict, key: str):
  """The LatentMoE-relevant fields live under `text_config`, not top-level
  -- but a few (like `dtype`) are duplicated at both levels. Check
  `text_config` first since that's the authoritative location for anything
  MoE-specific, falling back to top-level.
  """
  text_config = config.get("text_config", {})
  if key in text_config:
    return text_config[key]
  if key in config:
    return config[key]
  return None


def _env_fingerprint() -> dict:
  info = {"python_version": sys.version}
  try:
    import jax  # noqa: PLC0415 -- optional, best-effort only

    info["jax_version"] = jax.__version__
    info["jax_devices"] = [str(d) for d in jax.devices()]
  except Exception as e:  # noqa: BLE001 - environment fingerprint is best-effort
    info["jax_version"] = None
    info["jax_error"] = str(e)
  try:
    result = subprocess.run(
        ["git", "-C", str(pathlib.Path(__file__).parent.parent.parent / "tokamax"), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    info["tokamax_commit"] = result.stdout.strip() or None
  except Exception as e:  # noqa: BLE001
    info["tokamax_commit"] = None
    info["tokamax_error"] = str(e)
  return info


def main(commit: str) -> bool:
  OUT_DIR.mkdir(parents=True, exist_ok=True)
  hashes = {}
  raw_bytes = {}

  print(f"[validate-official-config] pinned commit: {commit}")
  for filename in FILES:
    url = f"https://huggingface.co/{REPO}/raw/{commit}/{filename}"
    dest = OUT_DIR / filename
    print(f"[validate-official-config] downloading {filename} ...")
    data = _download(url, dest)
    digest = _sha256(data)
    hashes[filename] = digest
    raw_bytes[filename] = data
    print(f"  saved to {dest}, sha256={digest}")

  config = json.loads(raw_bytes["config.json"])

  print("\n[validate-official-config] checking required fields:")
  all_ok = True
  field_results = {}
  for field, expected in REQUIRED_FIELDS.items():
    actual = _find_field(config, field)
    ok = actual == expected
    field_results[field] = {"expected": expected, "actual": actual, "ok": ok}
    print(f"  {field}: expected={expected!r} actual={actual!r} [{'OK' if ok else 'FAIL'}]")
    all_ok = all_ok and ok

  quant_cfg = _find_field(config, "quantization_config")
  print(f"\n[validate-official-config] quantization_config present: {quant_cfg is not None}")
  if quant_cfg is not None:
    print(f"  quant_method={quant_cfg.get('quant_method')!r} format={quant_cfg.get('format')!r}")
    print(
        "  NOTE: per this scheme's `ignore` list, only routed-expert weights "
        "(block_sparse_moe.experts.*.w1/w2/w3) are quantized -- router/shared_experts/"
        "norms remain plain bf16. WP-KV6 needs a dequantization step (the "
        "`compressed-tensors` library, or manual MXFP4 unpacking) for the routed-expert "
        "weights specifically before treating them as bf16."
    )

  env_info = _env_fingerprint()

  manifest_lines = [
      "# Kimi K3 official config/source manifest (WP-KV1)",
      "",
      f"Pinned commit: `{commit}`",
      f"Repo: `{REPO}`",
      "",
      "## File hashes (SHA256)",
      "",
  ]
  for filename, digest in hashes.items():
    manifest_lines.append(f"- `{filename}`: `{digest}`")
  manifest_lines += ["", "## Required-field validation", ""]
  manifest_lines.append("| field | expected | actual | status |")
  manifest_lines.append("|---|---|---|---|")
  for field, result in field_results.items():
    status = "OK" if result["ok"] else "FAIL"
    manifest_lines.append(f"| {field} | {result['expected']!r} | {result['actual']!r} | {status} |")
  manifest_lines += [
      "",
      "## Quantization config",
      "",
      "```json",
      json.dumps(quant_cfg, indent=2) if quant_cfg is not None else "null",
      "```",
      "",
      "## Environment fingerprint",
      "",
      "```json",
      json.dumps(env_info, indent=2, default=str),
      "```",
      "",
      f"## Overall: {'ALL FIELDS OK' if all_ok else 'SOME FIELDS FAILED'}",
      "",
  ]
  manifest_path = OUT_DIR / "MANIFEST.md"
  manifest_path.write_text("\n".join(manifest_lines), encoding="utf-8")
  print(f"\n[validate-official-config] manifest written to {manifest_path}")
  print(f"[validate-official-config] {'ALL FIELDS OK' if all_ok else 'SOME FIELDS FAILED'}")
  return all_ok


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--commit", default=DEFAULT_COMMIT, help="HF commit SHA to pin (default: the commit confirmed 2026-08-26)")
  args = parser.parse_args()

  ok = main(args.commit)
  sys.exit(0 if ok else 1)
