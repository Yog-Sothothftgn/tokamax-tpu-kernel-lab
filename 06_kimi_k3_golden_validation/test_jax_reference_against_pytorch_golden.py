"""WP-KV3 (KIMI_K3_LATENT_MOE_VALIDATION_PLAN_2026-08-26.md): diff this
project's JAX reference (kimi_k3_latent_moe_reference.py) against the real
official-PyTorch golden bundle from WP-KV2, stage by stage -- not just the
final output.

Uses `toy_config()` (kimi_k3_latent_moe_reference.py), which is dimension-
matched to `generate_pytorch_golden.py`'s SMALL_CONFIG_KWARGS on purpose
(hidden=64, latent=32, intermediate=48, experts=8, top_k=2, shared=1) so no
separate config needs constructing here -- the golden bundle's own
config.json is loaded and asserted to match toy_config(), rather than a
second hand-typed config existing side by side with drift risk.

**Ordering subtlety, handled explicitly below (do not skip this)**: PyTorch's
`torch.topk(..., sorted=False)` and JAX's `jax.lax.top_k` (always descending-
sorted) are not guaranteed to return a token's own top-k expert ids in the
same order, even when they select the exact same SET of experts (no ties
possible with continuous random scores, so the set itself is unambiguous --
only the within-token order can differ). This means:
  - `group_sizes` (a pure per-expert count, order-independent) can be
    compared directly, exactly.
  - `topk_indices`/`topk_weights` per token must be canonicalized (sorted by
    ascending expert id within each token row) before comparing, not
    compared as raw arrays.
  - Every per-DISPATCH-SLOT array (sorted_token_indices, expert_gate_output,
    expert_up_output, situ_output, expert_down_output) depends on the
    flattened (token,slot) order, which inherits the same ambiguity through
    the argsort-by-expert-id dispatch step -- these are canonicalized by
    re-sorting each implementation's own dispatch slots by (expert_id,
    token_id), a well-defined key pair (a token can't be assigned to the
    same expert twice), before comparing.
  - `routed_output_before_combine` ("new_x"/"unsorted") is DIFFERENT again:
    it's already been scattered BACK to (token,slot)-flat order by this
    point, not dispatch order -- but which "slot" (0..top_k-1) a given
    token's expert assignment lands in can still differ between
    implementations, so flat position `token*top_k+slot` isn't directly
    comparable either. Reshape to (num_tokens, top_k, latent) and reuse the
    SAME per-token sort-by-ascending-expert-id order already computed for
    topk_indices/topk_weights (see main()) -- caught this the hard way on
    the first run: expert_down_output (dispatch-order) and
    routed_output_after_combine (post-combine) both matched exactly while
    routed_output_before_combine alone showed a large diff, which is exactly
    the signature of comparing the right values in the wrong order rather
    than a real numerical bug.
  - Everything indexed by TOKEN post-combine (routed_output_after_combine,
    normalized_output, up_projection_output, shared_expert_output,
    final_output) needs no reordering -- token order is preserved throughout
    on both sides.

Usage:
  python test_jax_reference_against_pytorch_golden.py
"""

import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "05_ragged_dot_on_tpu"))

from kimi_k3_latent_moe_reference import (  # noqa: E402
    LatentMoEConfig,
    LatentMoEWeights,
    _rms_norm,
    _situ_and_mul,
    toy_config,
)

BUNDLE_DIR_FP32 = _HERE / "golden_bundle_small"
BUNDLE_DIR_BF16 = _HERE / "golden_bundle_small_bf16"

# Exact comparisons: indices and counts, no tolerance -- a mismatch here
# means the routing/dispatch logic itself disagrees, not a numerical
# precision difference.
EXACT_KEYS = ("group_sizes",)

# Floating-point comparisons: explicit atol/rtol, chosen generously enough
# to absorb JAX-vs-PyTorch differences in the underlying sigmoid/tanh/rsqrt
# implementations (different backends, not a correctness difference) but
# tight enough to catch a real formula/ordering bug. Actual observed max
# diffs are printed regardless, so these thresholds are auditable, not a
# black box. Both set by inspecting actual observed diffs from a real run,
# not guessed a priori -- bf16's was originally 5e-2/5e-2, set to paper
# over a real semantic bug in _run_instrumented_jax (a premature
# topk_weight downcast -- see main()'s replay-check comment); after that
# bug was fixed, every bf16 stage's real diff dropped to 0.0 except
# topk_weights (5.96e-08, ordinary float32 sigmoid backend noise from the
# router, which always computes in float32 regardless of compute_dtype),
# so this is now tight enough to actually catch a regression instead of
# hiding one.
TOLERANCES = {
    "fp32": (1e-4, 1e-3),
    "bf16": (1e-3, 1e-3),
}


def _load_bundle(bundle_dir: pathlib.Path):
  inputs = np.load(bundle_dir / "inputs.npz")
  weights_npz = np.load(bundle_dir / "weights.npz")
  outputs = np.load(bundle_dir / "golden_outputs.npz")
  with open(bundle_dir / "config.json", encoding="utf-8") as f:
    config_kwargs = json.load(f)
  with open(bundle_dir / "metadata.json", encoding="utf-8") as f:
    metadata = json.load(f)
  return inputs, weights_npz, outputs, config_kwargs, metadata


def _assert_config_matches_toy_config(config_kwargs: dict, config: LatentMoEConfig) -> None:
  """The golden bundle's config.json is the source of truth for what was
  actually run -- assert it matches toy_config() rather than silently
  trusting they're the same, since a future edit to either could drift."""
  mapping = {
      "hidden_size": config.hidden_size,
      "routed_expert_hidden_size": config.latent_size,
      "moe_intermediate_size": config.intermediate_size,
      "num_experts": config.num_experts,
      "num_experts_per_token": config.top_k,
      "num_shared_experts": config.num_shared_experts,
      "moe_renormalize": config.moe_renormalize,
      "routed_scaling_factor": config.routed_scaling_factor,
      "rms_norm_eps": config.rms_norm_eps,
      "activation_situ_beta": config.activation_situ_beta,
      "activation_situ_linear_beta": config.activation_situ_linear_beta,
  }
  mismatches = [
      (field, config_kwargs[field], jax_value)
      for field, jax_value in mapping.items()
      if config_kwargs[field] != jax_value
  ]
  assert not mismatches, (
      f"golden bundle's config.json disagrees with toy_config() on: {mismatches} -- "
      "fix the drift before trusting any comparison below"
  )


def _pt_weights_to_jax(w: dict, config: LatentMoEConfig, dtype: jnp.dtype) -> LatentMoEWeights:
  """PyTorch's nn.Linear stores weight as (out_features, in_features) and
  computes `x @ weight.T`; this project's JAX code stores weight as
  (in_features, out_features) and computes `x @ weight` -- so every 2D
  weight below needs a transpose, and every per-expert weight needs
  stacking across the leading expert axis.

  `weights.npz` was saved via generate_pytorch_golden.py's `_to_np`, which
  always upcasts to float32 before writing (numpy has no native bf16) --
  so `dtype` must be applied HERE to actually run this project's JAX
  reference at the same compute dtype the golden bundle was generated at
  (fp32 or bf16), not just inherit float32 from the saved array's on-disk
  dtype regardless of which bundle this is.
  """
  num_experts = config.num_experts
  expert_gate = np.stack([w[f"experts.{i}.w1.weight"].T for i in range(num_experts)])
  expert_up = np.stack([w[f"experts.{i}.w3.weight"].T for i in range(num_experts)])
  expert_down = np.stack([w[f"experts.{i}.w2.weight"].T for i in range(num_experts)])

  return LatentMoEWeights(
      router_weight=jnp.array(w["gate.weight"].T, dtype=dtype),
      e_score_correction_bias=jnp.array(w["gate.e_score_correction_bias"], dtype=dtype),
      down_proj=jnp.array(w["routed_expert_down_proj.weight"].T, dtype=dtype),
      up_proj=jnp.array(w["routed_expert_up_proj.weight"].T, dtype=dtype),
      expert_gate=jnp.array(expert_gate, dtype=dtype),
      expert_up=jnp.array(expert_up, dtype=dtype),
      expert_down=jnp.array(expert_down, dtype=dtype),
      norm_scale=jnp.array(w["routed_expert_norm.weight"], dtype=dtype),
      shared_gate=jnp.array(w["shared_experts.gate_proj.weight"].T, dtype=dtype),
      shared_up=jnp.array(w["shared_experts.up_proj.weight"].T, dtype=dtype),
      shared_down=jnp.array(w["shared_experts.down_proj.weight"].T, dtype=dtype),
  )


def _run_instrumented_jax(hidden_states: jax.Array, weights: LatentMoEWeights, config: LatentMoEConfig):
  """Line-by-line copy of latent_moe_forward, exposing the same 18
  intermediates generate_pytorch_golden.py's _run_instrumented captures on
  the PyTorch side, using the exact same variable-naming correspondence.
  Not a reimplementation -- this IS latent_moe_forward's body, just with
  intermediates stashed along the way (verified against calling
  latent_moe_forward directly in main() below, same cross-check discipline
  as the PyTorch side).
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
  intermediates["topk_indices"] = topk_idx
  intermediates["topk_weights"] = topk_weight
  # topk_weight stays float32 all the way through the combine sum below --
  # matching latent_moe_forward's real behavior (this project's own P1&2
  # precision fix from earlier this session: casting down here, before the
  # combine, would accumulate the top_k-way reduction in compute_dtype
  # instead of float32). A previous version of this line cast topk_weight
  # to compute_dtype right here, contradicting its own comment and
  # reintroducing the P1&2 bug in this instrumented copy specifically --
  # caught via a reviewer's read of this file, not by the cross-check below
  # (which had been loosened to 0.02 to paper over the resulting diff
  # instead of finding the actual cause).

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

  group_sizes_concrete = [int(s) for s in group_sizes]
  gate_outs, up_outs, situ_outs, down_outs = [], [], [], []
  start = 0
  for e in range(config.num_experts):
    size = group_sizes_concrete[e]
    chunk = sorted_tokens[start : start + size]
    if size > 0:
      gate_out = chunk @ weights.expert_gate[e]
      up_out = chunk @ weights.expert_up[e]
      situ_out = _situ_and_mul(gate_out, up_out, config.activation_situ_beta, config.activation_situ_linear_beta)
      down_out = situ_out @ weights.expert_down[e]
    else:
      gate_out = jnp.zeros((0, config.intermediate_size), dtype=compute_dtype)
      up_out = jnp.zeros((0, config.intermediate_size), dtype=compute_dtype)
      situ_out = jnp.zeros((0, config.intermediate_size), dtype=compute_dtype)
      down_out = jnp.zeros((0, config.latent_size), dtype=compute_dtype)
    gate_outs.append(gate_out)
    up_outs.append(up_out)
    situ_outs.append(situ_out)
    down_outs.append(down_out)
    start += size

  intermediates["expert_gate_output"] = jnp.concatenate(gate_outs, axis=0)
  intermediates["expert_up_output"] = jnp.concatenate(up_outs, axis=0)
  intermediates["situ_output"] = jnp.concatenate(situ_outs, axis=0)
  intermediates["expert_down_output"] = jnp.concatenate(down_outs, axis=0)

  outs = jnp.concatenate(down_outs, axis=0)
  unsorted = jnp.zeros_like(outs).at[order].set(outs)
  intermediates["routed_output_before_combine"] = unsorted

  unsorted_reshaped = unsorted.reshape(num_tokens, config.top_k, config.latent_size)
  routed_out = jnp.sum(unsorted_reshaped * topk_weight[..., None], axis=1)
  routed_out = routed_out.astype(compute_dtype)
  intermediates["routed_output_after_combine"] = routed_out

  normed = _rms_norm(routed_out, weights.norm_scale, config.rms_norm_eps)
  intermediates["normalized_output"] = normed
  up = normed @ weights.up_proj
  intermediates["up_projection_output"] = up

  shared_out = _situ_and_mul(
      identity @ weights.shared_gate, identity @ weights.shared_up,
      config.activation_situ_beta, config.activation_situ_linear_beta,
  ) @ weights.shared_down
  intermediates["shared_expert_output"] = shared_out

  final_output = up + shared_out
  intermediates["final_output"] = final_output

  return final_output, intermediates


def _canonical_order(dispatch_order: np.ndarray, group_sizes: np.ndarray) -> np.ndarray:
  """Re-sort key for per-dispatch-slot arrays: (expert_id, token_id),
  well-defined and implementation-order-independent since a token is never
  assigned to the same expert twice within its own top-k. `dispatch_order`
  is the flat (token,slot)-space permutation array (PyTorch's `idxs` /
  JAX's `order`); recovering token_id needs floor-dividing by top_k, done by
  the caller before this function since top_k isn't known here -- so this
  takes already-recovered per-slot token ids directly, not the raw
  permutation. See callers below.
  """
  expert_id_per_slot = np.repeat(np.arange(len(group_sizes)), group_sizes)
  return np.lexsort((dispatch_order, expert_id_per_slot))  # primary key: expert_id_per_slot


def _compare(name: str, golden: np.ndarray, ours: np.ndarray, exact: bool, atol: float, rtol: float) -> bool:
  if golden.shape != ours.shape:
    print(f"  {name}: FAIL -- shape mismatch golden={golden.shape} ours={ours.shape}")
    return False
  if exact:
    ok = bool(np.array_equal(golden, ours))
    print(f"  {name}: {'OK (exact)' if ok else 'FAIL (exact)'}")
    return ok
  diff = np.abs(golden.astype(np.float64) - ours.astype(np.float64))
  max_abs = float(diff.max()) if diff.size else 0.0
  ok = bool(np.allclose(golden, ours, atol=atol, rtol=rtol))
  print(f"  {name}: max_abs_diff={max_abs:.2e} {'OK' if ok else 'FAIL'} (atol={atol}, rtol={rtol})")
  return ok


def main(variant: str = "fp32") -> bool:
  bundle_dir = BUNDLE_DIR_FP32 if variant == "fp32" else BUNDLE_DIR_BF16
  jax_dtype = jnp.float32 if variant == "fp32" else jnp.bfloat16
  atol, rtol = TOLERANCES[variant]

  inputs, weights_npz, outputs, config_kwargs, metadata = _load_bundle(bundle_dir)
  print(f"[kv3] variant={variant} bundle={bundle_dir.name}")
  print(f"[kv3] golden bundle source commit: {metadata['source_provenance']['pinned_commit']}")
  assert metadata["dtype"].endswith("bfloat16" if variant == "bf16" else "float32"), (
      f"bundle's recorded dtype {metadata['dtype']!r} doesn't match the requested variant {variant!r}"
  )

  config = toy_config()
  _assert_config_matches_toy_config(config_kwargs, config)

  weights = _pt_weights_to_jax(weights_npz, config, dtype=jax_dtype)
  hidden_states = jnp.array(inputs["hidden_states"][0], dtype=jax_dtype)  # golden bundle has a leading batch=1 dim

  final_output, jax_intermediates = _run_instrumented_jax(hidden_states, weights, config)

  # Cross-check the instrumented replay against latent_moe_forward directly,
  # same discipline as generate_pytorch_golden.py's own cross-check.
  from kimi_k3_latent_moe_reference import latent_moe_forward

  direct_output = latent_moe_forward(hidden_states, weights, config)
  replay_max_err = float(jnp.max(jnp.abs(final_output.astype(jnp.float32) - direct_output.astype(jnp.float32))))
  print(f"[kv3] instrumented replay vs. latent_moe_forward directly: max_err={replay_max_err:.2e}")
  # Bit-exact for BOTH fp32 and bf16 -- and actually achieved. An earlier
  # version of _run_instrumented_jax cast topk_weight down to compute_dtype
  # right after computing it, instead of keeping it float32 through the
  # combine sum like latent_moe_forward actually does (this project's own
  # P1&2 precision fix, reintroduced here by mistake) -- that produced a
  # real ~7.8e-3 bf16 diff, which got misdiagnosed as "expected bf16
  # non-associativity from step-by-step vs. fused matmul ordering" and
  # papered over with a loose 0.02 tolerance instead of being fixed. Once
  # the premature cast was removed, the diff dropped to exactly 0.0 for
  # both dtypes -- proving it was 100% the semantic bug, not floating-point
  # noise. Keep this at 0.0; if it's ever nonzero again, that's a real
  # transcription bug to find and fix, not a tolerance to widen.
  assert replay_max_err == 0.0, (
      f"instrumented JAX replay diverged from latent_moe_forward by {replay_max_err:.2e} -- "
      "fix the actual cause, don't loosen this tolerance (see comment above)"
  )

  def compare(name: str, golden: np.ndarray, ours: np.ndarray, exact: bool = False) -> bool:
    return _compare(name, golden, ours, exact=exact, atol=atol, rtol=rtol)

  all_ok = True

  # --- group_sizes: exact, order-independent ---
  print("\n[kv3] exact comparisons:")
  all_ok &= compare(
      "group_sizes", outputs["group_sizes"], np.asarray(jax_intermediates["group_sizes"]), exact=True
  )

  # --- topk_indices / topk_weights: canonicalize by ascending expert id per token row ---
  print("\n[kv3] per-token canonicalized comparisons (sorted by expert id within each token):")
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

  # --- per-dispatch-slot arrays: canonicalize by (expert_id, token_id) ---
  print("\n[kv3] per-dispatch-slot canonicalized comparisons (sorted by expert id, then token id):")
  top_k = config.top_k
  g_dispatch_order = outputs["sorted_token_indices"]
  j_dispatch_order = np.asarray(jax_intermediates["sorted_token_indices"])
  g_token_ids = g_dispatch_order // top_k
  j_token_ids = j_dispatch_order // top_k
  g_canon = _canonical_order(g_token_ids, outputs["group_sizes"])
  j_canon = _canonical_order(j_token_ids, np.asarray(jax_intermediates["group_sizes"]))

  for key, exact in (
      ("expert_gate_output", False),
      ("expert_up_output", False),
      ("situ_output", False),
      ("expert_down_output", False),
  ):
    golden_arr = outputs[key][g_canon]
    ours_arr = np.asarray(jax_intermediates[key])[j_canon]
    all_ok &= compare(key, golden_arr, ours_arr, exact=exact)

  # routed_output_before_combine ("new_x"/"unsorted") is ALREADY scattered
  # back to (token, slot)-flat order by this point -- NOT dispatch order --
  # so the (expert_id, token_id) canonicalization above doesn't apply to it.
  # But a different ambiguity remains: which "slot" (0..top_k-1) a given
  # token's expert assignment landed in can differ between PyTorch's
  # sorted=False topk and JAX's always-descending top_k, so flat position
  # `token*top_k + slot` isn't directly comparable either. Reshape to
  # (num_tokens, top_k, latent) and reuse the SAME per-token
  # sort-by-ascending-expert-id order (g_order/j_order) already computed
  # for topk_indices/topk_weights above -- the correct, consistent fix.
  golden_before_combine = outputs["routed_output_before_combine"].reshape(-1, top_k, config.latent_size)
  jax_before_combine = np.asarray(jax_intermediates["routed_output_before_combine"]).reshape(-1, top_k, config.latent_size)
  golden_before_combine_sorted = np.take_along_axis(
      golden_before_combine, g_order[:, :, None], axis=1
  )
  jax_before_combine_sorted = np.take_along_axis(
      jax_before_combine, j_order[:, :, None], axis=1
  )
  all_ok &= compare(
      "routed_output_before_combine (per-token slot-canonicalized)",
      golden_before_combine_sorted,
      jax_before_combine_sorted,
      exact=False,
  )

  # sorted_token_indices itself: compare the recovered token IDs after
  # canonicalization (not the raw permutation, which is order-ambiguous by
  # construction -- see module docstring)
  all_ok &= compare(
      "sorted_token_indices (token ids, canonicalized)",
      g_token_ids[g_canon],
      j_token_ids[j_canon],
      exact=True,
  )

  # --- per-token arrays after combine: no reordering needed ---
  print("\n[kv3] per-token comparisons (post-combine, no reordering needed):")
  for key in (
      "routed_output_after_combine",
      "normalized_output",
      "up_projection_output",
  ):
    all_ok &= compare(key, outputs[key], np.asarray(jax_intermediates[key]), exact=False)

  golden_shared = outputs["shared_expert_output"].reshape(-1, config.hidden_size)
  golden_final = outputs["final_output"].reshape(-1, config.hidden_size)
  all_ok &= compare("shared_expert_output", golden_shared, np.asarray(jax_intermediates["shared_expert_output"]), exact=False)
  all_ok &= compare("final_output", golden_final, np.asarray(final_output), exact=False)

  print(f"\n[kv3] {'ALL STAGES MATCH' if all_ok else 'MISMATCH DETECTED'}")
  return all_ok


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--variant", choices=["fp32", "bf16", "both"], default="both")
  args = parser.parse_args()

  variants = ["fp32", "bf16"] if args.variant == "both" else [args.variant]
  all_ok = True
  for v in variants:
    print(f"\n{'=' * 20} variant={v} {'=' * 20}")
    all_ok = main(v) and all_ok

  sys.exit(0 if all_ok else 1)
