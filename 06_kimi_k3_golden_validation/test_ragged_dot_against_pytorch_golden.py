"""WP-KV4 (KIMI_K3_LATENT_MOE_VALIDATION_PLAN_2026-08-26.md): regression-test
xla/mosaic-v1/mosaic-v2 (via `latent_moe_forward_ragged_dot` in
kimi_k3_latent_moe_ragged_dot.py) against the SAME golden bundles WP-KV3
already validated this project's naive JAX reference against -- so a kernel
swap (xla -> mosaic) is checked against real official-model ground truth,
not just against this project's own reference implementation.

**Has a real tokamax dependency and therefore CANNOT be verified locally**
(same constraint as kimi_k3_latent_moe_ragged_dot.py itself -- Windows
long-path pip install blocks a local tokamax install, see project memory).
Must run on the v6e TPU VM. Do not trust its output until it has actually
executed on hardware.

Reuses WP-KV3's helper functions directly (`_load_bundle`,
`_pt_weights_to_jax`, `_assert_config_matches_toy_config`, `_canonical_order`,
`_compare`) -- none of those have a tokamax dependency themselves, only this
file's own instrumented forward pass does (it calls `tokamax.ragged_dot`).

Unlike the naive per-expert Python loop (kimi_k3_latent_moe_reference.py),
`latent_moe_forward_ragged_dot` dispatches ALL tokens through THREE
`tokamax.ragged_dot` calls at once (gate, up, down) rather than looping
per-expert -- so `gate`/`up`/`activated`/`outs` are ALREADY the full
dispatch-order arrays WP-KV3's `expert_gate_output` etc. correspond to, no
per-expert loop/concatenation needed to capture them.

**Three bundle sets** (the 64/32/48 "small" bundle is below Mosaic's
confirmed hard 128-tiling floor, so it can only ever validate xla; the
256/128/128 "mosaic" bundle satisfies the K/N part of that floor but at its
original num_tokens=20, dispatch rows M=40 are still below the 128-row
floor, so mosaic (v1) has only ever been confirmed to SKIP there, never to
actually execute a kernel -- see "mosaic_wide" below, added 2026-08-28):
  - `--bundle-set small` -- the original 64/32/48 bundles (toy_config()),
    xla only by default.
  - `--bundle-set mosaic` -- the 256/128/128 bundles at num_tokens=20
    (mosaic_correctness_config()), xla + mosaic + mosaic_tpu_v2 by default
    -- mosaic (v1) is expected to SKIP here (M=40<128), not a bug.
  - `--bundle-set mosaic_wide` -- same 256/128/128 dims, num_tokens=64 ->
    M=128 (generated via `generate_pytorch_golden.py --config mosaic
    --num-tokens 64`), the first bundle where mosaic (v1) can actually run
    and get diffed against real official-model ground truth.
  - `--bundle-set both` (default) -- small + mosaic + mosaic_wide, with each
    set's own default implementations.

A `NotImplementedError` from any implementation counts as FAIL, never a
silent pass -- see run_one()'s try/except.

Usage (on the TPU VM, tokamax installed):
  python test_ragged_dot_against_pytorch_golden.py --bundle-set small --variant fp32 --implementation xla
  python test_ragged_dot_against_pytorch_golden.py --bundle-set mosaic_wide --variant bf16 --implementation mosaic
  python test_ragged_dot_against_pytorch_golden.py --bundle-set both --variant both  # full sweep, this project's recommended default
"""

import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "05_ragged_dot_on_tpu"))
sys.path.insert(0, str(_HERE))

# These four are pure-numpy/JAX helpers with no tokamax dependency --
# reused directly from WP-KV3 rather than duplicated.
from test_jax_reference_against_pytorch_golden import (  # noqa: E402
    BUNDLE_DIR_BF16,
    BUNDLE_DIR_FP32,
    TOLERANCES,
    _assert_config_matches_toy_config,
    _canonical_order,
    _compare,
    _load_bundle,
    _pt_weights_to_jax,
)

from kimi_k3_latent_moe_reference import _rms_norm, _situ_and_mul, toy_config  # noqa: E402
from kimi_k3_latent_moe_ragged_dot import _situ_glu_mlp, mosaic_correctness_config  # noqa: E402
import tokamax  # noqa: E402


def _run_instrumented_ragged_dot(hidden_states, weights, config, implementation: str):
  """Same structural instrumentation as WP-KV3's _run_instrumented_jax, but
  step 4 goes through tokamax.ragged_dot instead of a per-expert Python
  loop -- so expert_gate_output/expert_up_output/situ_output/
  expert_down_output are the direct ragged_dot outputs (already in
  dispatch order), not something assembled from a per-expert loop.

  Caller (run_one) wraps this whole function in
  `jax.default_matmul_precision("highest")`: TPU's MXU defaults every
  matmul-like op (plain `@`, `jnp.dot`, `tokamax.ragged_dot`) to a
  reduced-precision algorithm for float32 inputs unless told otherwise --
  invisible on CPU (where WP-KV3's naive-loop reference was validated), but
  producing real, non-negligible error on actual TPU hardware. The first
  hardware run of this file only set `precision=` on the 3 explicit
  `tokamax.ragged_dot` calls below and still saw ~1e-2-level mismatches in
  normalized_output/up_projection_output/final_output -- because the
  router logits, down_proj, up_proj, and shared-expert matmuls are all
  plain `@` and were still running at reduced precision. The context
  manager at the call site covers everything in one place instead of
  threading `precision=` through each call individually; the explicit
  `precision=` kwarg below is kept too (belt-and-suspenders, in case
  tokamax.ragged_dot doesn't read the ambient default the same way plain
  `@` does).
  """
  intermediates = {}
  identity = hidden_states
  compute_dtype = hidden_states.dtype

  logits = hidden_states.astype(jnp.float32) @ weights.router_weight.astype(jnp.float32)
  intermediates["router_logits"] = logits
  scores = jax.nn.sigmoid(logits)
  intermediates["router_scores_before_bias"] = scores
  scores_for_choice = scores + weights.e_score_correction_bias.astype(jnp.float32)[None, :]
  intermediates["router_scores_for_choice"] = scores_for_choice

  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)
  topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
    topk_weight = topk_weight / denom
  topk_weight = topk_weight * config.routed_scaling_factor
  # topk_weight stays float32 through the combine sum below -- see
  # kimi_k3_latent_moe_ragged_dot.py's matching comment / this project's
  # P1&2 precision fix. Do NOT cast it down here (WP-KV3 caught exactly
  # this mistake once already in the naive-reference instrumented copy).
  intermediates["topk_indices"] = topk_idx
  intermediates["topk_weights"] = topk_weight

  x = hidden_states @ weights.down_proj
  intermediates["latent_projection"] = x

  num_tokens = hidden_states.shape[0]
  flat_expert_ids = topk_idx.reshape(-1)
  order = jnp.argsort(flat_expert_ids)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  sorted_token_idx = token_of_slot[order]
  sorted_tokens = x[sorted_token_idx]
  group_sizes = jnp.bincount(flat_expert_ids, length=config.num_experts)
  intermediates["sorted_token_indices"] = order
  intermediates["group_sizes"] = group_sizes

  # precision=HIGHEST is load-bearing on TPU for fp32, not defensive:
  # tokamax.ragged_dot's own default (precision=None -> "DEFAULT") uses a
  # reduced-precision matmul algorithm for float32 inputs on TPU's MXU (a
  # single bf16-ish pass, not true IEEE float32) -- confirmed via the
  # autotuning-cache-miss log line printing `"precision":["DEFAULT","DEFAULT"]`,
  # and confirmed FIXED on real hardware once this (plus wrapping every
  # OTHER matmul in the function in `jax.default_matmul_precision("highest")`,
  # see run_one) was applied: fp32/xla and fp32/mosaic_tpu_v2 both went from
  # ~1e-2-level mismatches to ALL STAGES MATCH (~1e-6/1e-7, ordinary
  # cross-backend float noise).
  #
  # For bf16 inputs, HIGHEST is not just unnecessary (bf16 IS the MXU's
  # native precision -- there's no lower-precision hardware path to emulate
  # a "higher" one over) but actively BREAKS mosaic_tpu_v2 on real hardware:
  # passing precision=HIGHEST with bf16 lhs/rhs crashes Mosaic's compiler
  # with `Bad lhs type` (`tpu.matmul` requesting
  # `precision=#tpu.contract_precision<fp32>` against a `vector<...xbf16>`
  # operand -- confirmed on hardware, not a hypothetical). Gate this on the
  # actual compute dtype rather than hardcoding HIGHEST unconditionally.
  precision = jax.lax.Precision.HIGHEST if compute_dtype == jnp.float32 else None
  gate = tokamax.ragged_dot(
      sorted_tokens, weights.expert_gate, group_sizes, precision=precision, implementation=implementation
  )
  up = tokamax.ragged_dot(
      sorted_tokens, weights.expert_up, group_sizes, precision=precision, implementation=implementation
  )
  activated = _situ_and_mul(gate, up, config.activation_situ_beta, config.activation_situ_linear_beta)
  outs = tokamax.ragged_dot(
      activated, weights.expert_down, group_sizes, precision=precision, implementation=implementation
  )

  intermediates["expert_gate_output"] = gate
  intermediates["expert_up_output"] = up
  intermediates["situ_output"] = activated
  intermediates["expert_down_output"] = outs

  unsorted = jnp.zeros_like(outs).at[order].set(outs)
  intermediates["routed_output_before_combine"] = unsorted

  unsorted_reshaped = unsorted.reshape(num_tokens, config.top_k, config.latent_size)
  routed_out = jnp.sum(unsorted_reshaped * topk_weight[..., None], axis=1)
  routed_out = routed_out.astype(compute_dtype)
  intermediates["routed_output_after_combine"] = routed_out

  normed = _rms_norm(routed_out, weights.norm_scale, config.rms_norm_eps)
  intermediates["normalized_output"] = normed
  up_proj_out = normed @ weights.up_proj
  intermediates["up_projection_output"] = up_proj_out

  shared_out = _situ_glu_mlp(
      identity, weights.shared_gate, weights.shared_up, weights.shared_down,
      config.activation_situ_beta, config.activation_situ_linear_beta,
  )
  intermediates["shared_expert_output"] = shared_out

  final_output = up_proj_out + shared_out
  intermediates["final_output"] = final_output

  return final_output, intermediates


def run_one(bundle_dir: pathlib.Path, config, variant: str, implementation: str) -> bool:
  jax_dtype = jnp.float32 if variant == "fp32" else jnp.bfloat16
  atol, rtol = TOLERANCES[variant]

  inputs, weights_npz, outputs, config_kwargs, metadata = _load_bundle(bundle_dir)
  print(f"\n[kv4] variant={variant} implementation={implementation!r} bundle={bundle_dir.name}")
  print(f"[kv4] golden bundle source commit: {metadata['source_provenance']['pinned_commit']}")

  _assert_config_matches_toy_config(config_kwargs, config)

  weights = _pt_weights_to_jax(weights_npz, config, dtype=jax_dtype)
  hidden_states = jnp.array(inputs["hidden_states"][0], dtype=jax_dtype)

  try:
    # See _run_instrumented_ragged_dot's docstring: TPU defaults every
    # matmul-like op to reduced precision for float32 inputs unless told
    # otherwise -- this context manager covers ALL of them (router, ragged_dot,
    # down_proj/up_proj, shared-expert), not just the 3 explicit ragged_dot
    # calls (an earlier version only fixed those and still saw ~1e-2-level
    # mismatches downstream from the un-fixed plain `@` matmuls).
    # "highest" only for fp32: confirmed on hardware that forcing it for
    # bf16 crashes mosaic_tpu_v2's compiler (bf16 is already the MXU's
    # native precision -- there's no lower-precision path "highest" needs to
    # emulate past, and the Pallas kernel doesn't handle that combination).
    matmul_precision = "highest" if variant == "fp32" else "default"
    with jax.default_matmul_precision(matmul_precision):
      final_output, jax_intermediates = _run_instrumented_ragged_dot(
          hidden_states, weights, config, implementation=implementation
      )
  except NotImplementedError as e:
    print(f"[kv4] implementation={implementation!r}: SKIPPED ({e}) -- not counted as a pass")
    return False

  def compare(name: str, golden: np.ndarray, ours: np.ndarray, exact: bool = False) -> bool:
    return _compare(name, golden, ours, exact=exact, atol=atol, rtol=rtol)

  all_ok = True

  print("[kv4] exact comparisons:")
  all_ok &= compare("group_sizes", outputs["group_sizes"], np.asarray(jax_intermediates["group_sizes"]), exact=True)

  print("[kv4] per-token canonicalized comparisons:")
  g_idx, g_w = outputs["topk_indices"], outputs["topk_weights"]
  j_idx, j_w = np.asarray(jax_intermediates["topk_indices"]), np.asarray(jax_intermediates["topk_weights"])
  g_order = np.argsort(g_idx, axis=-1)
  j_order = np.argsort(j_idx, axis=-1)
  g_idx_sorted = np.take_along_axis(g_idx, g_order, axis=-1)
  j_idx_sorted = np.take_along_axis(j_idx, j_order, axis=-1)
  g_w_sorted = np.take_along_axis(g_w, g_order, axis=-1)
  j_w_sorted = np.take_along_axis(j_w, j_order, axis=-1)
  all_ok &= compare("topk_indices (canonicalized)", g_idx_sorted, j_idx_sorted, exact=True)
  all_ok &= compare("topk_weights (canonicalized)", g_w_sorted, j_w_sorted, exact=False)

  print("[kv4] per-dispatch-slot canonicalized comparisons:")
  top_k = config.top_k
  g_dispatch_order = outputs["sorted_token_indices"]
  j_dispatch_order = np.asarray(jax_intermediates["sorted_token_indices"])
  g_token_ids = g_dispatch_order // top_k
  j_token_ids = j_dispatch_order // top_k
  g_canon = _canonical_order(g_token_ids, outputs["group_sizes"])
  j_canon = _canonical_order(j_token_ids, np.asarray(jax_intermediates["group_sizes"]))

  for key in ("expert_gate_output", "expert_up_output", "situ_output", "expert_down_output"):
    all_ok &= compare(key, outputs[key][g_canon], np.asarray(jax_intermediates[key])[j_canon])

  golden_before_combine = outputs["routed_output_before_combine"].reshape(-1, top_k, config.latent_size)
  jax_before_combine = np.asarray(jax_intermediates["routed_output_before_combine"]).reshape(-1, top_k, config.latent_size)
  golden_before_combine_sorted = np.take_along_axis(golden_before_combine, g_order[:, :, None], axis=1)
  jax_before_combine_sorted = np.take_along_axis(jax_before_combine, j_order[:, :, None], axis=1)
  all_ok &= compare(
      "routed_output_before_combine (per-token slot-canonicalized)",
      golden_before_combine_sorted, jax_before_combine_sorted,
  )
  all_ok &= compare(
      "sorted_token_indices (token ids, canonicalized)", g_token_ids[g_canon], j_token_ids[j_canon], exact=True
  )

  print("[kv4] per-token comparisons (post-combine):")
  for key in ("routed_output_after_combine", "normalized_output", "up_projection_output"):
    all_ok &= compare(key, outputs[key], np.asarray(jax_intermediates[key]))

  golden_shared = outputs["shared_expert_output"].reshape(-1, config.hidden_size)
  golden_final = outputs["final_output"].reshape(-1, config.hidden_size)
  all_ok &= compare("shared_expert_output", golden_shared, np.asarray(jax_intermediates["shared_expert_output"]))
  all_ok &= compare("final_output", golden_final, np.asarray(final_output))

  print(f"[kv4] variant={variant} implementation={implementation!r}: {'ALL STAGES MATCH' if all_ok else 'MISMATCH DETECTED'}")
  return all_ok


BUNDLE_DIR_MOSAIC_FP32 = _HERE / "golden_bundle_mosaic_fp32"
BUNDLE_DIR_MOSAIC_BF16 = _HERE / "golden_bundle_mosaic_bf16"
BUNDLE_DIR_MOSAIC_WIDE_FP32 = _HERE / "golden_bundle_mosaic_fp32_n64"
BUNDLE_DIR_MOSAIC_WIDE_BF16 = _HERE / "golden_bundle_mosaic_bf16_n64"

# (bundle_dir, config, default implementations) per bundle-set. "small"
# (64/32/48) is below Mosaic's hard 128-tiling floor -- confirmed on
# hardware elsewhere in this project (`NotImplementedError: RaggedDot
# inputs must be >= 128`) -- so it can only ever validate xla; a
# NotImplementedError from mosaic there is expected, not a bug, but still
# only run xla by default to avoid a confusing default-FAIL. "mosaic"
# (256/128/128, generated via generate_pytorch_golden.py --config mosaic)
# satisfies the K/N tiling floor, but at the original num_tokens=20,
# dispatch rows M = num_tokens*top_k = 40 -- still below the 128-row floor,
# so mosaic (v1) has only ever been confirmed to SKIP there, never to
# actually run. "mosaic_wide" (same 256/128/128 dims, num_tokens=64 ->
# M=128, generated via `generate_pytorch_golden.py --config mosaic
# --num-tokens 64`) is the first bundle where mosaic (v1) can actually
# execute a kernel and get diffed against real official-model ground truth.
BUNDLE_SETS = {
    "small": {
        "fp32": (BUNDLE_DIR_FP32, toy_config()),
        "bf16": (BUNDLE_DIR_BF16, toy_config()),
        "default_implementations": ("xla",),
    },
    "mosaic": {
        "fp32": (BUNDLE_DIR_MOSAIC_FP32, mosaic_correctness_config()),
        "bf16": (BUNDLE_DIR_MOSAIC_BF16, mosaic_correctness_config()),
        "default_implementations": ("xla", "mosaic", "mosaic_tpu_v2"),
    },
    "mosaic_wide": {
        "fp32": (BUNDLE_DIR_MOSAIC_WIDE_FP32, mosaic_correctness_config()),
        "bf16": (BUNDLE_DIR_MOSAIC_WIDE_BF16, mosaic_correctness_config()),
        "default_implementations": ("xla", "mosaic", "mosaic_tpu_v2"),
    },
}


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--bundle-set", choices=["small", "mosaic", "mosaic_wide", "both"], default="both"
  )
  parser.add_argument("--variant", choices=["fp32", "bf16", "both"], default="both")
  parser.add_argument(
      "--implementation",
      choices=["xla", "mosaic", "mosaic_tpu_v2", "all"],
      default=None,
      help="defaults to xla-only for --bundle-set small, all three for --bundle-set mosaic "
      "(mosaic/mosaic_tpu_v2 can't run at all below the 128-tiling floor, so testing them "
      "against the small bundle would just report an expected NotImplementedError as FAIL)",
  )
  args = parser.parse_args()

  bundle_sets = ["small", "mosaic", "mosaic_wide"] if args.bundle_set == "both" else [args.bundle_set]
  variants = ["fp32", "bf16"] if args.variant == "both" else [args.variant]

  all_ok = True
  for bset in bundle_sets:
    spec = BUNDLE_SETS[bset]
    implementations = (
        ("xla", "mosaic", "mosaic_tpu_v2") if args.implementation == "all"
        else (args.implementation,) if args.implementation is not None
        else spec["default_implementations"]
    )
    for v in variants:
      bundle_dir, config = spec[v]
      for impl in implementations:
        all_ok = run_one(bundle_dir, config, v, impl) and all_ok

  print(f"\n[kv4] {'ALL PASS' if all_ok else 'AT LEAST ONE MISMATCH/SKIP'}")
  sys.exit(0 if all_ok else 1)
