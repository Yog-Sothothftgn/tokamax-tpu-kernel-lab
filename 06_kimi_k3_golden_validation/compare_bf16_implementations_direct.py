"""Direct, same-device, element-wise comparison between tokamax.ragged_dot's
`xla`/`mosaic` (v1)/`mosaic_tpu_v2` bf16 outputs on the SAME inputs, in the
SAME run -- ALL 18 staged intermediates, not just a couple, and saved to
disk so the comparison can be re-examined later without rerunning on
hardware.

Closes a gap flagged repeatedly in this project's README/memory: every
prior bf16 finding only compared each implementation's `max_abs_diff`
AGAINST the golden PyTorch reference separately, then noted that the
resulting numbers happened to match across implementations -- concluding
"this suggests a shared bf16/backend precision effect" from matching
SUMMARY STATISTICS computed in different runs, never from a direct
tensor-level diff between the implementations' own raw outputs. This
script computes every available implementation in one process on the same
device, saves each one's full output bundle, and diffs them against EACH
OTHER directly across all 18 intermediates.

**Scope of what this actually proves (tightened 2026-09-02 per reviewer
feedback)**: `xla`/`mosaic`/`mosaic_tpu_v2` are NOT three fully independent
end-to-end implementations -- they share routing, dispatch, combine,
RMSNorm, projection, and this test's own instrumentation wrapper; the only
thing that differs between them is the `tokamax.ragged_dot` backend
itself. So if all three report BIT-IDENTICAL values at the 4
previously-flagged stages (`expert_up_output`, `up_projection_output`,
`normalized_output`, `final_output`), that RULES OUT a difference among
the three ragged_dot backends at those stages, and SUPPORTS -- but does
not by itself PROVE -- a shared PyTorch-vs-TPU precision effect (a bug in
the shared surrounding code, not just in one backend, would also produce
bit-identical results across all three). If the three implementations
DISAGREE with each other (not just with golden), that points to a
backend-specific difference and needs further digging.

Not a pass/fail tolerance check like `test_ragged_dot_against_pytorch_golden.py`'s
`run_one` -- this is a diagnostic comparison. It reports the actual direct
diffs; interpreting them is informed by this output plus the existing
vs-golden numbers, not automated into a verdict beyond the bit-identical
check above.

Saves `<implementation>_outputs.npz` per computed implementation (all 18
intermediates + final_output) under `--output-dir`, so a later session (or
a person, not just this script) can re-load and re-examine the raw tensors
without needing a TPU again.

Has a real tokamax dependency (via `_run_instrumented_ragged_dot`) -- CANNOT
be verified locally, must run on the v6e TPU VM.

Usage:
  python compare_bf16_implementations_direct.py --bundle-set mosaic_wide --output-dir results/bf16_direct_compare
"""

import argparse
import itertools
import json
import pathlib
import sys

import jax  # noqa: E402
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "05_ragged_dot_on_tpu"))
sys.path.insert(0, str(_HERE))

from test_jax_reference_against_pytorch_golden import _canonical_order, _load_bundle, _pt_weights_to_jax  # noqa: E402
from test_ragged_dot_against_pytorch_golden import (  # noqa: E402
    BUNDLE_SETS,
    _run_instrumented_ragged_dot,
)

# Maps tokamax's `implementation=` string to the output-file name the user
# asked for -- "mosaic" is v1, "mosaic_tpu_v2" is v2.
_OUTPUT_FILE_NAME = {
    "xla": "xla_outputs",
    "mosaic": "mosaic_v1_outputs",
    "mosaic_tpu_v2": "mosaic_v2_outputs",
}

# Every staged intermediate captured by _run_instrumented_ragged_dot, plus
# final_output -- the FULL set, not just the 2 stages an earlier version of
# this script compared.
_ALL_STAGES = (
    "router_logits", "router_scores_before_bias", "router_scores_for_choice",
    "topk_indices", "topk_weights", "latent_projection", "group_sizes",
    "sorted_token_indices", "expert_gate_output", "expert_up_output",
    "situ_output", "expert_down_output", "routed_output_before_combine",
    "routed_output_after_combine", "normalized_output", "up_projection_output",
    "shared_expert_output", "final_output",
)

# The 4 stages every prior bf16 run in this project has flagged as
# exceeding tolerance vs. the golden reference -- singled out in the final
# verdict section below.
_FLAGGED_STAGES = ("expert_up_output", "up_projection_output", "normalized_output", "final_output")


def _to_np(x) -> np.ndarray:
  return np.asarray(x)


def compute_and_save_implementation(
    impl: str,
    hidden_states: jnp.ndarray,
    weights,
    config,
    output_dir: pathlib.Path,
) -> dict[str, np.ndarray] | None:
  """Runs one implementation, returns its full {stage: array} dict (numpy,
  all 18 intermediates + final_output), and saves it to
  `<name>_outputs.npz`. Returns None (and prints why) if the implementation
  raises NotImplementedError (e.g. mosaic (v1) below its tiling floor)."""
  try:
    with jax.default_matmul_precision("default"):  # bf16 -- "default" matches run_one's established pattern
      final_output, intermediates = _run_instrumented_ragged_dot(
          hidden_states, weights, config, implementation=impl
      )
  except NotImplementedError as e:
    print(f"[bf16-direct-compare] implementation={impl!r}: SKIPPED ({e})")
    return None

  bundle = {stage: _to_np(intermediates[stage]) for stage in _ALL_STAGES if stage != "final_output"}
  bundle["final_output"] = _to_np(final_output)

  output_dir.mkdir(parents=True, exist_ok=True)
  out_path = output_dir / f"{_OUTPUT_FILE_NAME[impl]}.npz"
  np.savez(out_path, **bundle)
  print(f"[bf16-direct-compare] implementation={impl!r}: computed OK, saved to {out_path}")
  return bundle


def _compare_stage(name: str, a: np.ndarray, b: np.ndarray, exact: bool = False) -> float:
  """Returns max_abs_diff (float('nan') if shapes mismatch), prints the result."""
  if a.shape != b.shape:
    print(f"    {name}: FAIL -- shape mismatch {a.shape} vs {b.shape}")
    return float("nan")
  if exact:
    ok = bool(np.array_equal(a, b))
    print(f"    {name}: {'bit-identical' if ok else 'DIFFERS (exact compare)'}")
    return 0.0 if ok else float("inf")
  diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
  max_abs = float(diff.max()) if diff.size else 0.0
  print(f"    {name}: max_abs_diff={max_abs:.3e}{' (bit-identical)' if max_abs == 0.0 else ''}")
  return max_abs


def compare_two_implementations(
    name_a: str, bundle_a: dict[str, np.ndarray],
    name_b: str, bundle_b: dict[str, np.ndarray],
    config,
) -> dict[str, float]:
  """Full 18-stage canonicalized comparison between two implementations'
  saved output bundles -- same canonicalization scheme as
  test_ragged_dot_against_pytorch_golden.py's run_one (dispatch order isn't
  guaranteed to match between implementations, even though both are
  correct, so raw-order comparison would spuriously fail), just applied
  between two JAX-computed bundles instead of one PyTorch + one JAX bundle.

  Returns {stage_name: max_abs_diff} for every stage compared.
  """
  print(f"\n  --- {name_a!r} vs {name_b!r} ---")
  top_k = config.top_k
  results: dict[str, float] = {}

  # Stages with no dispatch-order dependence -- direct comparison, no
  # canonicalization needed.
  for key in ("router_logits", "router_scores_before_bias", "router_scores_for_choice", "latent_projection"):
    results[key] = _compare_stage(key, bundle_a[key], bundle_b[key])

  results["group_sizes"] = _compare_stage("group_sizes", bundle_a["group_sizes"], bundle_b["group_sizes"], exact=True)

  a_idx, a_w = bundle_a["topk_indices"], bundle_a["topk_weights"]
  b_idx, b_w = bundle_b["topk_indices"], bundle_b["topk_weights"]
  a_order = np.argsort(a_idx, axis=-1)
  b_order = np.argsort(b_idx, axis=-1)
  a_idx_sorted = np.take_along_axis(a_idx, a_order, axis=-1)
  b_idx_sorted = np.take_along_axis(b_idx, b_order, axis=-1)
  a_w_sorted = np.take_along_axis(a_w, a_order, axis=-1)
  b_w_sorted = np.take_along_axis(b_w, b_order, axis=-1)
  results["topk_indices"] = _compare_stage("topk_indices (canonicalized)", a_idx_sorted, b_idx_sorted, exact=True)
  results["topk_weights"] = _compare_stage("topk_weights (canonicalized)", a_w_sorted, b_w_sorted)

  a_dispatch_order = bundle_a["sorted_token_indices"]
  b_dispatch_order = bundle_b["sorted_token_indices"]
  a_token_ids = a_dispatch_order // top_k
  b_token_ids = b_dispatch_order // top_k
  a_canon = _canonical_order(a_token_ids, bundle_a["group_sizes"])
  b_canon = _canonical_order(b_token_ids, bundle_b["group_sizes"])

  for key in ("expert_gate_output", "expert_up_output", "situ_output", "expert_down_output"):
    results[key] = _compare_stage(key, bundle_a[key][a_canon], bundle_b[key][b_canon])

  a_before_combine = bundle_a["routed_output_before_combine"].reshape(-1, top_k, config.latent_size)
  b_before_combine = bundle_b["routed_output_before_combine"].reshape(-1, top_k, config.latent_size)
  a_before_combine_sorted = np.take_along_axis(a_before_combine, a_order[:, :, None], axis=1)
  b_before_combine_sorted = np.take_along_axis(b_before_combine, b_order[:, :, None], axis=1)
  results["routed_output_before_combine"] = _compare_stage(
      "routed_output_before_combine (per-token slot-canonicalized)",
      a_before_combine_sorted, b_before_combine_sorted,
  )
  results["sorted_token_indices"] = _compare_stage(
      "sorted_token_indices (token ids, canonicalized)", a_token_ids[a_canon], b_token_ids[b_canon], exact=True
  )

  for key in ("routed_output_after_combine", "normalized_output", "up_projection_output"):
    results[key] = _compare_stage(key, bundle_a[key], bundle_b[key])

  results["shared_expert_output"] = _compare_stage(
      "shared_expert_output",
      bundle_a["shared_expert_output"].reshape(-1, config.hidden_size),
      bundle_b["shared_expert_output"].reshape(-1, config.hidden_size),
  )
  results["final_output"] = _compare_stage(
      "final_output",
      bundle_a["final_output"].reshape(-1, config.hidden_size),
      bundle_b["final_output"].reshape(-1, config.hidden_size),
  )
  return results


def compare_bf16_implementations(
    bundle_set: str = "mosaic_wide",
    output_dir: pathlib.Path = pathlib.Path("bf16_direct_compare_outputs"),
) -> bool:
  spec = BUNDLE_SETS[bundle_set]
  bundle_dir, config = spec["bf16"]
  implementations = spec["default_implementations"]

  inputs, weights_npz, _outputs, _config_kwargs, metadata = _load_bundle(bundle_dir)
  weights = _pt_weights_to_jax(weights_npz, config, dtype=jnp.bfloat16)
  hidden_states = jnp.array(inputs["hidden_states"][0], dtype=jnp.bfloat16)

  print(f"[bf16-direct-compare] bundle={bundle_dir.name} commit={metadata['source_provenance']['pinned_commit']}")
  print(f"[bf16-direct-compare] output_dir={output_dir}")

  bundles: dict[str, dict[str, np.ndarray]] = {}
  for impl in implementations:
    result = compute_and_save_implementation(impl, hidden_states, weights, config, output_dir)
    if result is not None:
      bundles[impl] = result

  computed = list(bundles.keys())
  if len(computed) < 2:
    print(
        f"[bf16-direct-compare] only {len(computed)} implementation(s) computed successfully "
        "-- need at least 2 to compare directly"
    )
    return False

  print(f"\n[bf16-direct-compare] direct pairwise comparisons across all {len(_ALL_STAGES)} stages "
        f"({len(computed)} implementations computed):")
  per_pair_flagged: dict[tuple[str, str], dict[str, float]] = {}
  all_stage_results: dict[str, dict[str, float]] = {}
  for impl_a, impl_b in itertools.combinations(computed, 2):
    stage_results = compare_two_implementations(impl_a, bundles[impl_a], impl_b, bundles[impl_b], config)
    per_pair_flagged[(impl_a, impl_b)] = {k: stage_results[k] for k in _FLAGGED_STAGES}
    all_stage_results[f"{impl_a}_vs_{impl_b}"] = stage_results

  print(f"\n[bf16-direct-compare] VERDICT on the {len(_FLAGGED_STAGES)} previously-flagged stages "
        f"({', '.join(_FLAGGED_STAGES)}):")
  all_bit_identical = True
  for (impl_a, impl_b), flagged in per_pair_flagged.items():
    for stage, diff in flagged.items():
      if diff != 0.0:
        all_bit_identical = False
      print(f"    {impl_a!r} vs {impl_b!r} / {stage}: max_abs_diff={diff:.3e}")

  # NOTE (reviewer, 2026-09-02): xla/mosaic/mosaic_tpu_v2 are NOT three fully
  # independent end-to-end implementations -- they share routing, dispatch,
  # combine, RMSNorm, projection, and this test's own instrumentation
  # wrapper; the only thing that actually differs between them is the
  # ragged_dot backend. So bit-identical results across all three pairs
  # rule out a difference AMONG THE RAGGED_DOT BACKENDS specifically at
  # these stages -- they do NOT, by themselves, rule out a problem in the
  # shared surrounding code (wrapper, projections, precision handling) that
  # all three would reproduce identically. The verdict below is phrased to
  # reflect exactly that scope, not a stronger claim.
  if all_bit_identical:
    print(
        "\n[bf16-direct-compare] All computed implementation pairs are BIT-IDENTICAL at every "
        "previously-flagged stage. This rules out an implementation-specific difference among "
        "the three ragged-dot backends at the compared stages, and supports -- but does not by "
        "itself prove -- a shared PyTorch-vs-TPU precision effect (the three backends share "
        "everything except the ragged_dot call itself: routing, dispatch, combine, RMSNorm, "
        "projection, and this test's own wrapper code are identical across all three, so a bug "
        "in any of THOSE shared parts would also show up as 'bit-identical across backends')."
    )
  else:
    print(
        "\n[bf16-direct-compare] Computed implementations DISAGREE with each other on at least "
        "one previously-flagged stage -- this points AWAY from 'shared PyTorch-vs-TPU precision "
        "effect' and toward a difference specific to one of the ragged_dot backends that needs "
        "further investigation."
    )

  correctness_path = pathlib.Path(output_dir) / "correctness.json"
  correctness_path.parent.mkdir(parents=True, exist_ok=True)
  correctness_path.write_text(
      json.dumps(
          {
              "bundle_set": bundle_set,
              "implementations_computed": computed,
              "all_stage_results": all_stage_results,
              "flagged_stages_all_bit_identical": all_bit_identical,
          },
          indent=2,
      ),
      encoding="utf-8",
  )
  print(f"\n[bf16-direct-compare] structured per-stage results written to {correctness_path}")

  return True


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--bundle-set",
      choices=["mosaic", "mosaic_wide"],
      default="mosaic_wide",
      help="'mosaic' (n=20, M=40) can only compute xla (mosaic/mosaic_tpu_v2 below the tiling "
      "floor there); 'mosaic_wide' (n=64, M=128, default) can compute all three, giving the "
      "most complete pairwise comparison",
  )
  parser.add_argument(
      "--output-dir",
      type=pathlib.Path,
      default=pathlib.Path("bf16_direct_compare_outputs"),
      help="directory to save <impl>_outputs.npz files to (created if it doesn't exist)",
  )
  args = parser.parse_args()
  ok = compare_bf16_implementations(bundle_set=args.bundle_set, output_dir=args.output_dir)
  sys.exit(0 if ok else 1)
