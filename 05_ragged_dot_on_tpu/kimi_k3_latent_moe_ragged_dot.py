"""WP-Kimi step 2: swap the naive per-expert loop for tokamax.ragged_dot.

Builds directly on kimi_k3_latent_moe_reference.py (WP-Kimi step 1). Steps
1-3 and 5-8 are unchanged from the reference; step 4 (the per-expert FFN) is
replaced with three tokamax.ragged_dot calls (gate, up, down -- the routed
expert FFN is a 3-matrix FFN with Kimi's custom SiTU-GLU activation, NOT
SiLU/SwiGLU, confirmed from source, see the reference file's module
docstring and `_situ_and_mul` for the full correction history):

  gate = ragged_dot(sorted_tokens, expert_gate, group_sizes)  # (M, latent) x (G, latent, inter) -> (M, inter)
  up   = ragged_dot(sorted_tokens, expert_up,   group_sizes)  # (M, latent) x (G, latent, inter) -> (M, inter)
  outs = ragged_dot(situ_and_mul(gate, up, beta, linear_beta), expert_down, group_sizes)  # (M, inter) x (G, inter, latent) -> (M, latent)

where M = num_tokens * top_k and G = num_experts -- the (lhs, rhs,
group_sizes) shape convention used throughout benchmark_harness.py's WP1/WP3
work.

**This file was rewritten on 2026-08-24 following an external architecture
review (KIMI_K3_KERNEL_REVIEW_2026-08-24.md), independently re-confirmed
against source before applying any fix.** The previous version had 2 fixed
bugs (OOM at 896 experts, LatentMoEWeights not a registered pytree) but was
still built on the WRONG router order inherited from the reference file --
see that file's docstring for the full list of what changed. The three
tokamax.ragged_dot calls below (not two) and the corrected weight field
names (expert_gate/expert_up/expert_down, router now taking hidden_states
directly) reflect the fix.

**2026-08-25, round-2 fix (review's P1-A item):** the activation between
the gate/up ragged_dot calls and the down ragged_dot call was
`jax.nn.silu(gate) * up` -- plain SiLU/SwiGLU, architecturally wrong for
Kimi K3. Replaced with `_situ_and_mul` (shared with the reference file),
using `config.activation_situ_beta`/`config.activation_situ_linear_beta`
(confirmed 4.0/25.0 from config.json).

**2026-08-25, round-2 fix (review's P1-B + P2-A items):** the router gate
and RMSNorm (`_rms_norm`, shared with the reference file) now both compute
in float32 regardless of compute dtype, casting back only at the end --
matters most at real benchmark scale where hidden_states/weights are bf16
(`run_benchmark`/`run_fair_baseline` below): a bf16 gate can flip which
experts land in the top-16 selection near the boundary, not just lose
precision. Verified locally (bf16, real per-expert dims hidden=7168/
latent=3584/intermediate=3072, reduced num_experts=4 to fit in CPU memory)
that output dtype stays bf16 and is NaN/Inf-free post-fix.

Unlike the naive Python loop in the reference (which needs concrete Python
ints for its slice sizes and therefore can't be jitted), ragged_dot handles
a *traced* group_sizes internally via Mosaic's tile-to-group metadata (see
wp3_notes.md / make_group_metadata) -- so this whole forward pass is
jit-compatible. That's the practical payoff of this step, not just speed.

This file has a real tokamax dependency and therefore CANNOT be verified
locally (Windows long-path pip install blocks a local tokamax install --
see project memory). It must be run on the v6e TPU VM. Do not trust its
output until it has actually executed on hardware.

Usage (on the TPU VM, tokamax installed -- except --route-filter-correctness,
which is pure JAX and needs neither tokamax nor a TPU):
  python kimi_k3_latent_moe_ragged_dot.py --correctness             # toy scale (xla) + Mosaic-compatible scale (xla/mosaic/mosaic_tpu_v2), both vs. the naive reference
  python kimi_k3_latent_moe_ragged_dot.py --benchmark               # single-chip-shard Kimi K3 scale, xla vs mosaic heuristic
  python kimi_k3_latent_moe_ragged_dot.py --fair-baseline           # same scale, + mosaic v2 + autotune-tuned comparison
  python kimi_k3_latent_moe_ragged_dot.py --shard-workload          # correctness check (valid-rows-only) + isolated expert-kernel benchmark on a realistic, fixed-total-padded 16-of-896-filtered workload (WP-Kimi step 2b part 1)
  python kimi_k3_latent_moe_ragged_dot.py --route-filter-correctness  # standalone (no tokamax/TPU) correctness + overflow test for REAL 896-expert routing filtered to a local shard (WP-Kimi step 2b part 2)
  python kimi_k3_latent_moe_ragged_dot.py --latency-sweep           # latency across multiple batch_size/seq_len pairs, one table (Zifan's 2026-08-28 standing request, see run_latency_sweep)

**2026-08-26, WP-Kimi step 2b (review's P1-C item, two-step plan per user
direction):** `single_chip_kimi_k3_config`'s num_experts=64/top_k=16 setup
is a dense, uniformly-distributed 16-of-64 workload, NOT what a real
16-of-896-then-filtered-to-64 shard would see (~14x lower average count,
more skew-prone -- see that function's docstring). Full fix needs real
global routing + local-id filtering -- a bigger change, still not done.
**Step 1 (done here)**: `generate_local_shard_workload` generates an
isolated expert-kernel benchmark input with the REALISTIC per-expert count
statistics directly (no actual global-routing simulation yet), PLUS a
fixed-total, tile-aligned padding scheme so xla/mosaic-v1/mosaic-v2 all
benchmark the identical shape and values (`run_shard_workload_benchmark`)
rather than each padding a data-dependent M by its own convention.

A first version of this padding (same day) forced every expert's
`group_sizes` entry to an identical constant `capacity` -- a review caught
that this silently erased the real skew the whole benchmark exists to
exercise (Mosaic's kernel reacts to per-group sizes, so uniform
group_sizes quietly turns this back into a too-regular workload). Fixed:
per-expert `group_sizes` are now the REAL, unmodified, genuinely-skewed
counts; the fixed-shape requirement is instead met by ONE trailing
padding-bucket group (an extra dummy expert row) that absorbs whatever's
needed to reach a fixed, tile-aligned total. A `valid_mask` tracks which
rows are real vs padding so correctness checks
(`check_shard_workload_correctness`) can exclude padded rows rather than
silently diffing against meaningless zero-input output. **Step 2 (done,
2026-08-26)**: `route_and_filter_to_local_shard` -- real 896-expert top-k +
local-id filtering (same padding scheme, now over real routing output
instead of a synthetic draw), tested standalone via
`check_route_and_filter_correctness`.

**Step 3 (done, 2026-08-28): wired into an actual end-to-end forward pass
and checked against a ground truth for the first time --
`check_sharded_forward_correctness` in `kimi_k3_latent_moe_reference.py`.**
Also on 2026-08-28, `generate_local_shard_workload`,
`route_and_filter_to_local_shard`, and `check_route_and_filter_correctness`
were MOVED to `kimi_k3_latent_moe_reference.py` and are imported back here
-- they never had a tokamax dependency of their own, but this file imports
tokamax unconditionally at module level, which blocked them from being
run/tested on a machine without a working tokamax install (this Windows dev
machine). They're re-exported here via the import above for backward
compatibility with `run_shard_workload_benchmark`/the CLI below.
"""

import argparse
import dataclasses

# Same environment workaround as benchmark_harness.py -- must run before
# `import tokamax` on this jax/flax/qwix version combination.
import jax.experimental.hijax as _hijax  # noqa: E402

if not hasattr(_hijax, "MutableHiType"):

  class _MutableHiTypeStub:
    pass

  _hijax.MutableHiType = _MutableHiTypeStub

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import tokamax  # noqa: E402

from kimi_k3_latent_moe_reference import (  # noqa: E402
    LatentMoEConfig,
    LatentMoEWeights,
    _rms_norm,
    _situ_and_mul,
    _situ_glu_mlp,
    check_route_and_filter_correctness,
    generate_local_shard_workload,
    init_weights,
    kimi_k3_config,
    latent_moe_forward,
    route_and_filter_to_local_shard,
    toy_config,
)


def latent_moe_forward_ragged_dot(
    hidden_states: jax.Array,
    weights: LatentMoEWeights,
    config: LatentMoEConfig,
    implementation: str | None = None,
) -> jax.Array:
  """Same 8-step forward pass as latent_moe_forward, step 4 via ragged_dot."""
  identity = hidden_states

  # Step 1: router, on the ORIGINAL hidden_states. Gate math in float32
  # regardless of compute dtype (review's P1-B item) -- see the reference
  # file's matching comment for why this isn't just a precision nicety.
  compute_dtype = hidden_states.dtype
  logits = hidden_states.astype(jnp.float32) @ weights.router_weight.astype(jnp.float32)
  scores = jax.nn.sigmoid(logits)
  scores_for_choice = scores + weights.e_score_correction_bias.astype(jnp.float32)[None, :]
  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)
  topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
    topk_weight = topk_weight / denom
  topk_weight = topk_weight * config.routed_scaling_factor
  # topk_weight stays float32 through the weighted-sum combine below --
  # casting down here would accumulate the top_k=16-way reduction in bf16,
  # losing exactly the precision the float32 gate fix above was for. Cast
  # back to compute_dtype happens after the combine instead.

  # Step 2: shared down-projection, applied to all tokens.
  x = hidden_states @ weights.down_proj

  num_tokens = hidden_states.shape[0]
  flat_expert_ids = topk_idx.reshape(-1)
  order = jnp.argsort(flat_expert_ids)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  sorted_token_idx = token_of_slot[order]
  sorted_tokens = x[sorted_token_idx]
  group_sizes = jnp.bincount(flat_expert_ids, length=config.num_experts)

  # Step 4, ragged_dot version -- replaces the reference's Python loop.
  # Routed expert FFN is 3-matrix, Kimi's SiTU-GLU activation (NOT
  # SiLU/SwiGLU): situ_and_mul(x @ gate, x @ up) @ down.
  gate = tokamax.ragged_dot(
      sorted_tokens, weights.expert_gate, group_sizes, implementation=implementation
  )
  up = tokamax.ragged_dot(
      sorted_tokens, weights.expert_up, group_sizes, implementation=implementation
  )
  activated = _situ_and_mul(
      gate, up, config.activation_situ_beta, config.activation_situ_linear_beta
  )
  outs = tokamax.ragged_dot(
      activated, weights.expert_down, group_sizes, implementation=implementation
  )

  unsorted = jnp.zeros_like(outs).at[order].set(outs)
  unsorted = unsorted.reshape(num_tokens, config.top_k, config.latent_size)
  # topk_weight (float32) promotes this reduction to float32; cast back to
  # compute_dtype right after, not before (see step 1's comment).
  routed_out = jnp.sum(unsorted * topk_weight[..., None], axis=1)
  routed_out = routed_out.astype(compute_dtype)

  normed = _rms_norm(routed_out, weights.norm_scale, config.rms_norm_eps)
  up_proj_out = normed @ weights.up_proj

  # Step 8: shared experts (dense, not routed -- no ragged_dot needed here).
  shared_out = _situ_glu_mlp(
      identity,
      weights.shared_gate,
      weights.shared_up,
      weights.shared_down,
      config.activation_situ_beta,
      config.activation_situ_linear_beta,
  )

  return up_proj_out + shared_out


def _check_correctness_for_config(
    config: LatentMoEConfig,
    num_tokens: int,
    implementations: tuple[str, ...],
    seed: int = 0,
) -> bool:
  """Shared driver: naive Python-loop reference vs. one or more ragged_dot
  implementations, at whatever config/dims the caller passes in. The
  reference is always the ground truth here, not xla-vs-mosaic.
  """
  key = jax.random.key(seed)
  key_w, key_x = jax.random.split(key)
  weights = init_weights(config, key_w)
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))

  reference_out = latent_moe_forward(hidden_states, weights, config)

  all_ok = True
  for impl in implementations:
    try:
      out = latent_moe_forward_ragged_dot(hidden_states, weights, config, implementation=impl)
    except NotImplementedError as e:
      # A skip is NOT a pass: the caller only requests implementations it
      # expects to actually run at this config's dims, so an unexpected
      # NotImplementedError here means something needs investigating, not
      # silent continuation. Previously this `continue`d without touching
      # `all_ok`, so a run where every implementation got skipped still
      # returned True -- a no-op test reporting success.
      print(f"[correctness] implementation={impl!r}: SKIPPED ({e}) -- counted as FAIL")
      all_ok = False
      continue
    max_err = float(jnp.max(jnp.abs(out - reference_out)))
    ok = max_err < 1e-3  # ragged_dot's own tiling can introduce small reduction-order differences
    print(f"[correctness] implementation={impl!r} max_err={max_err:.2e} {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok
  return all_ok


def check_correctness(seed: int = 0) -> bool:
  """Toy-scale: ragged_dot (xla only) vs. the naive Python-loop reference.

  Mosaic's TPU kernel enforces a hard minimum of 128 on the lhs/rhs matmul
  dims (confirmed on hardware: `NotImplementedError: RaggedDot inputs must
  be >= 128` when run against toy_config()'s latent_size=32/
  intermediate_size=48). That's a real Mosaic-kernel tiling constraint, not a
  bug in this dispatch/gather/scatter code -- xla has no such floor and
  matches the reference exactly (max_err=0.0) at this scale, which already
  confirms the ragged_dot wiring (group_sizes, shapes, argument order) is
  correct.

  **Mosaic's own correctness is NOT confirmed here** -- this only ever runs
  xla, by construction (toy_config's dims are below Mosaic's floor). Prior
  versions of this docstring claimed Mosaic correctness was "confirmed for
  free" by run_benchmark() at real Kimi K3 scale -- that was wrong
  (review's P1-D item): run_benchmark/run_fair_baseline only time Mosaic,
  they never diff its output against anything. See
  check_mosaic_correctness() for the actual Mosaic-output check, added
  2026-08-25 to close this gap.
  """
  config = toy_config()
  print(
      f"[correctness] toy scale: latent={config.latent_size}, "
      f"intermediate={config.intermediate_size} < Mosaic's 128 minimum -- xla only "
      "(see check_mosaic_correctness for the Mosaic-compatible check)"
  )
  return _check_correctness_for_config(config, num_tokens=64, implementations=("xla",), seed=seed)


def mosaic_correctness_config() -> LatentMoEConfig:
  """Smallest config with every matmul dim >=128, so Mosaic's kernel actually
  runs (not just XLA) and its output can be diffed against the naive
  reference -- review's P1-D item, dims per the review's suggestion
  (KIMI_K3_KERNEL_REVIEW_2026-08-24.md). Not Kimi K3's real per-expert
  shape (that's kimi_k3_config()/single_chip_kimi_k3_config()) -- this
  config exists purely to satisfy Mosaic's tiling floor cheaply.
  """
  return LatentMoEConfig(
      hidden_size=256,
      latent_size=128,
      intermediate_size=128,
      num_experts=8,
      top_k=2,
      num_shared_experts=1,
      moe_renormalize=True,
      routed_scaling_factor=1.0,
      rms_norm_eps=1e-5,
      activation_situ_beta=4.0,
      activation_situ_linear_beta=25.0,
  )


def check_mosaic_correctness(seed: int = 0) -> bool:
  """Mosaic-compatible correctness check (review's P1-D item, added
  2026-08-25): dims all >=128 so xla, mosaic (v1), and mosaic_tpu_v2 all
  actually run and get diffed against the naive reference. Until this
  function existed, Mosaic's *output* had never been checked for this
  architecture at all -- run_benchmark/run_fair_baseline only ever timed
  it. A NotImplementedError from a given implementation at this config is
  reported as SKIPPED, not a failure (mirrors run_fair_baseline's handling).
  """
  config = mosaic_correctness_config()
  return _check_correctness_for_config(
      config, num_tokens=64, implementations=("xla", "mosaic", "mosaic_tpu_v2"), seed=seed
  )


def single_chip_kimi_k3_config(num_experts: int = 64) -> LatentMoEConfig:
  """Kimi K3's real per-expert shapes, num_experts scoped to fit one v6e chip.

  kimi_k3_config()'s full 896 experts is NOT reproducible on a single v6e
  chip: confirmed on hardware (RESOURCE_EXHAUSTED at float32; even bf16 with
  the (wrong, 2-matrix) earlier expert design needed ~39.5GB, and the
  corrected 3-matrix design needs ~59GB -- see review doc). This isn't a bug
  to fix: a 2.8T-param model's routed-expert weights are necessarily sharded
  across many chips in real deployment (expert parallelism) -- no single
  chip ever holds all 896.

  This function keeps the real per-expert matmul shape (latent_size=3584,
  intermediate_size=3072 -- what ragged_dot actually computes against, and
  the thing WP-Kimi step 2 is testing) but reduces num_experts to a size one
  chip's HBM can hold, modeling "one chip's shard of the routed experts."

  **Known remaining simplification (review's P1-C item, not yet fixed
  here):** simply setting num_experts=64 while leaving top_k=16 means each
  token picks 16 of 64 LOCAL experts, not 16 of 896 GLOBAL experts filtered
  down to whichever land on this shard -- that makes group_sizes far more
  uniform/regular than a real shard would see. At num_tokens=2048: this
  64-expert simplification gives 2048*16/64 = 512 assignments/expert on
  average; a real 16-of-896 shard would see 2048*16/896 ~= 36.6
  assignments/expert -- **~14x higher, not the ~4x an earlier version of
  this docstring claimed** (that was the reviewer catching my own arithmetic
  error, not an approximation -- see KIMI_K3_KERNEL_REVIEW_2026-08-24.md's
  P1-C item). Fixing this properly requires routing over all 896 experts and
  filtering to a local id range before dispatch -- a bigger change
  (dynamic-shape masking) deferred to a follow-up; for now, treat
  run_benchmark/run_fair_baseline's numbers as "does ragged_dot handle
  Kimi's real per-expert matmul shape efficiently," not "what Kimi K3's real
  routing distribution does to ragged_dot."
  """
  return dataclasses.replace(kimi_k3_config(), num_experts=num_experts)


def run_shard_workload_benchmark(
    seed: int = 0, num_experts: int = 64, num_tokens: int = 2048
) -> None:
  """WP-Kimi step 2b, part 1: isolated expert-kernel (three ragged_dot
  calls: gate/up/down + SiTU-GLU, no router/dispatch/combine) benchmark
  using generate_local_shard_workload's realistic-distribution,
  fixed-total-padded input, instead of single_chip_kimi_k3_config's
  dense/uniform 16-of-64 workload. Answers "does the xla-vs-mosaic trend
  change once the workload actually looks like Kimi K3's real
  (~14x-smaller-average, skew-prone) shard distribution," ahead of the
  bigger global-routing-filter rewrite (part 2, not yet implemented -- see
  generate_local_shard_workload's docstring). `group_sizes` here is the
  REAL per-expert counts (genuinely skewed, identical across all three
  implementations below since it's generated once and reused) plus one
  trailing padding-bucket entry -- the expert weight tensors get one matching
  dummy `+1`th row for that bucket, never touched by any real token.
  """
  config = single_chip_kimi_k3_config(num_experts)
  key = jax.random.key(seed)
  key_workload, key_weights = jax.random.split(key)

  sorted_tokens, group_sizes, valid_mask, per_expert_counts = generate_local_shard_workload(
      key_workload,
      num_tokens=num_tokens,
      global_num_experts=kimi_k3_config().num_experts,  # 896, the REAL global count
      top_k=config.top_k,
      local_num_experts=num_experts,
      latent_size=config.latent_size,
      dtype=jnp.bfloat16,
  )
  expected_mean = num_tokens * config.top_k / kimi_k3_config().num_experts
  print(
      f"\n[shard-workload] num_tokens={num_tokens} local_num_experts={num_experts} "
      f"M_padded={sorted_tokens.shape[0]} pad_bucket_size={int(group_sizes[-1])} "
      f"valid_rows={int(jnp.sum(valid_mask))} "
      f"mean_per_expert={float(jnp.mean(per_expert_counts)):.2f} "
      f"(expected~{expected_mean:.2f}) min={int(jnp.min(per_expert_counts))} "
      f"max={int(jnp.max(per_expert_counts))}"
  )

  # +1 dummy expert row for the trailing padding bucket -- its weights are
  # never exercised by real data (that bucket's input rows are all zero).
  keys = jax.random.split(key_weights, 3)
  scale = 0.02
  expert_gate = (
      jax.random.normal(keys[0], (num_experts + 1, config.latent_size, config.intermediate_size))
      * scale
  ).astype(jnp.bfloat16)
  expert_up = (
      jax.random.normal(keys[1], (num_experts + 1, config.latent_size, config.intermediate_size))
      * scale
  ).astype(jnp.bfloat16)
  expert_down = (
      jax.random.normal(keys[2], (num_experts + 1, config.intermediate_size, config.latent_size))
      * scale
  ).astype(jnp.bfloat16)

  def _expert_ffn(x, gate_w, up_w, down_w, implementation):
    gate = tokamax.ragged_dot(x, gate_w, group_sizes, implementation=implementation)
    up = tokamax.ragged_dot(x, up_w, group_sizes, implementation=implementation)
    activated = _situ_and_mul(
        gate, up, config.activation_situ_beta, config.activation_situ_linear_beta
    )
    return tokamax.ragged_dot(activated, down_w, group_sizes, implementation=implementation)

  for impl in ("xla", "mosaic", "mosaic_tpu_v2"):
    try:
      f_impl = jax.jit(
          lambda x, gw, uw, dw, impl=impl: _expert_ffn(x, gw, uw, dw, implementation=impl)
      )
      std_f, args = tokamax.standardize_function(
          f_impl, sorted_tokens, expert_gate, expert_up, expert_down
      )
      bench = tokamax.benchmark(jax.jit(std_f), args, method="hermetic_xprof")
      print(
          f"  {impl}: compile={bench.compile_time_ms:.2f}ms "
          f"median_exec={bench.median_evaluation_time_ms:.4f}ms "
          f"peak_mem={bench.peak_memory_mb:.2f}MB"
      )
    except NotImplementedError as e:
      print(f"  {impl}: SKIPPED: {e}")


def check_shard_workload_correctness(
    seed: int = 0, num_experts: int = 64, num_tokens: int = 2048
) -> bool:
  """Correctness check for generate_local_shard_workload's fixed-total,
  tile-aligned padded workload (real per-expert group_sizes + one trailing
  padding bucket). xla's ragged_dot output is the ground truth here (same
  pattern as check_mosaic_correctness -- there's no separate naive-loop
  reference for this router/dispatch-free, expert-kernel-only path), diffed
  against mosaic/mosaic_tpu_v2's output on the exact SAME padded input,
  **restricted to `valid_mask` rows only**: padded rows hold no real token
  (their output is a deterministic function of zero input, not something
  meaningful to diff), so they're masked out of the max-error computation
  rather than silently included. The valid row count is reported alongside
  the error, not just discarded, per the same "preserve the valid-token
  count" requirement generate_local_shard_workload documents for its
  `valid_mask`/`per_expert_counts` outputs.

  Runs in float32 (not the benchmark's bf16) so max_err reflects ragged_dot
  tiling/reduction-order differences, not bf16 rounding noise -- same
  reasoning as check_correctness's tolerance comment.
  """
  config = single_chip_kimi_k3_config(num_experts)
  key = jax.random.key(seed)
  key_workload, key_weights = jax.random.split(key)

  sorted_tokens, group_sizes, valid_mask, _ = generate_local_shard_workload(
      key_workload,
      num_tokens=num_tokens,
      global_num_experts=kimi_k3_config().num_experts,
      top_k=config.top_k,
      local_num_experts=num_experts,
      latent_size=config.latent_size,
      dtype=jnp.float32,
  )

  # +1 dummy expert row for the trailing padding bucket, matching group_sizes.
  keys = jax.random.split(key_weights, 3)
  scale = 0.02
  expert_gate = jax.random.normal(
      keys[0], (num_experts + 1, config.latent_size, config.intermediate_size)
  ) * scale
  expert_up = jax.random.normal(
      keys[1], (num_experts + 1, config.latent_size, config.intermediate_size)
  ) * scale
  expert_down = jax.random.normal(
      keys[2], (num_experts + 1, config.intermediate_size, config.latent_size)
  ) * scale

  def _expert_ffn(implementation: str) -> jax.Array:
    gate = tokamax.ragged_dot(sorted_tokens, expert_gate, group_sizes, implementation=implementation)
    up = tokamax.ragged_dot(sorted_tokens, expert_up, group_sizes, implementation=implementation)
    activated = _situ_and_mul(
        gate, up, config.activation_situ_beta, config.activation_situ_linear_beta
    )
    return tokamax.ragged_dot(activated, expert_down, group_sizes, implementation=implementation)

  reference_out = _expert_ffn("xla")
  num_valid = int(jnp.sum(valid_mask))

  all_ok = True
  for impl in ("mosaic", "mosaic_tpu_v2"):
    try:
      out = _expert_ffn(impl)
    except NotImplementedError as e:
      print(f"[shard-workload-correctness] implementation={impl!r}: SKIPPED ({e}) -- counted as FAIL")
      all_ok = False
      continue
    diff = jnp.where(valid_mask[:, None], jnp.abs(out - reference_out), 0.0)
    max_err = float(jnp.max(diff))
    ok = max_err < 1e-3
    print(
        f"[shard-workload-correctness] implementation={impl!r} "
        f"max_err(valid-only)={max_err:.2e} {'OK' if ok else 'FAIL'} "
        f"(valid_rows={num_valid}/{valid_mask.shape[0]})"
    )
    all_ok = all_ok and ok
  return all_ok


def run_benchmark(seed: int = 0, num_experts: int = 64, num_tokens: int = 2048) -> None:
  """Kimi K3 real per-expert scale (see single_chip_kimi_k3_config), one
  chip's worth of experts: xla vs mosaic latency + memory.

  Mirrors benchmark_harness.py's run_one/run_fair_baseline pattern (same
  tokamax.standardize_function + tokamax.benchmark(..., method="hermetic_xprof")
  call, and the same positional-args tokamax.autotune(f, lhs, rhs,
  group_sizes, all_implementations=True) call -- see that file's docstring
  for the two tokamax docs bugs those calls work around).
  """
  config = single_chip_kimi_k3_config(num_experts)
  key = jax.random.key(seed)
  key_w, key_x = jax.random.split(key)
  weights = init_weights(config, key_w, dtype=jnp.bfloat16)
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size), dtype=jnp.bfloat16)

  for impl in ("xla", "mosaic"):
    f = jax.jit(
        lambda h, w: latent_moe_forward_ragged_dot(h, w, config, implementation=impl)
    )
    std_f, args = tokamax.standardize_function(f, hidden_states, weights)
    result = tokamax.benchmark(jax.jit(std_f), args, method="hermetic_xprof")
    print(f"[benchmark] implementation={impl!r}: {result}")


def run_fair_baseline(
    seed: int = 0, num_experts: int = 64, num_tokens: int = 2048, skip_autotune: bool = True
) -> None:
  """WP-Kimi step 2 (full): xla vs mosaic-v1 vs mosaic-v2, heuristic (and
  optionally tuned), at single-chip-shard Kimi K3 scale. Mirrors
  benchmark_harness.py's run_fair_baseline (WP3.5.1) -- same
  tokamax.autotune(f, *args, all_implementations=True) call, positional args
  (see that file's docstring for why -- the docs' keyword-arg example is
  wrong).

  **`skip_autotune` defaults to True.** Confirmed on hardware (2026-08-24,
  Kimi's real per-expert matmul shape, before this architecture correction):
  autotuning this shape reported "Total microbenchmarks=2404" -- 32x
  WP3.5.1's small-shape run (74 microbenchmarks, ~6 minutes) -- and was
  killed by hand after ~12 minutes still on op-call 1/6, no ETA in sight.
  latent_moe_forward_ragged_dot now has THREE tokamax.ragged_dot call sites
  per forward pass (gate, up, down), not one, which likely multiplies the
  search space further. The heuristic-only loop below already answers the
  practically important question cheaply: at this scale, mosaic v2's
  heuristic config alone beat xla (6.43ms vs 8.97ms, no tuning needed),
  while mosaic v1's heuristic was ~21x slower than xla (188ms) -- consistent
  with the WP3.5.1 large-shape finding, not a new problem. Pass
  skip_autotune=False only if there's a specific reason to see whether
  tuning can close v1's gap, and be ready for it to run a long time.
  """
  config = single_chip_kimi_k3_config(num_experts)
  key = jax.random.key(seed)
  key_w, key_x = jax.random.split(key)
  weights = init_weights(config, key_w, dtype=jnp.bfloat16)
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size), dtype=jnp.bfloat16)

  print(
      f"\n[fair-baseline] single-chip Kimi K3 (G={num_experts}, num_tokens={num_tokens}) "
      "-- heuristic (untuned):"
  )
  for impl in ("xla", "mosaic", "mosaic_tpu_v2"):
    try:
      f_impl = jax.jit(
          lambda h, w: latent_moe_forward_ragged_dot(h, w, config, implementation=impl)
      )
      std_f, args = tokamax.standardize_function(f_impl, hidden_states, weights)
      bench = tokamax.benchmark(jax.jit(std_f), args, method="hermetic_xprof")
      print(
          f"  {impl}: compile={bench.compile_time_ms:.2f}ms "
          f"median_exec={bench.median_evaluation_time_ms:.4f}ms "
          f"peak_mem={bench.peak_memory_mb:.2f}MB"
      )
    except NotImplementedError as e:
      print(f"  {impl}: SKIPPED: {e}")

  if skip_autotune:
    print("\n[fair-baseline] skip_autotune=True -- not running the exhaustive autotune search.")
    return

  # `implementation=None` doesn't matter much here: `all_implementations=True`
  # below overrides it and tunes every registered implementation regardless.
  def f(h, w):
    return latent_moe_forward_ragged_dot(h, w, config, implementation=None)

  print(
      "\n[fair-baseline] single-chip Kimi K3 -- autotuning ALL implementations "
      "(three ragged_dot call sites per forward pass, confirmed VERY slow at "
      "this shape -- see docstring) ..."
  )
  autotune_result = tokamax.autotune(f, hidden_states, weights, all_implementations=True)

  print("\n[fair-baseline] single-chip Kimi K3 -- tuned results per implementation/call-site:")
  for bound_args, data in autotune_result.data:
    impl_name = type(bound_args.op).__name__
    shape_hint = ""
    try:
      lhs = bound_args.arguments.get("lhs")
      rhs = bound_args.arguments.get("rhs")
      if lhs is not None and rhs is not None:
        shape_hint = f" lhs={tuple(lhs.shape)} rhs={tuple(rhs.shape)}"
    except Exception:  # noqa: BLE001 - shape hint is best-effort only
      pass
    try:
      best_config = data.fastest_config
      best = data[best_config]
      print(
          f"  {impl_name}{shape_hint}: tuned median_exec={best.median_evaluation_time_ms:.4f}ms "
          f"peak_mem={best.peak_memory_mb:.2f}MB config={best_config}"
      )
    except Exception as e:  # noqa: BLE001 - reporting tuning failures, not raising
      print(f"  {impl_name}{shape_hint}: FAILED to autotune ({e})")


_DEFAULT_LATENCY_SWEEP_SHAPES: tuple[tuple[int, int], ...] = (
    # (batch_size, seq_len). hidden_states is (num_tokens, hidden_size) --
    # batch_size and seq_len only ever enter this forward pass through their
    # product, num_tokens = batch_size * seq_len (there is no separate
    # batch/sequence axis anywhere in latent_moe_forward_ragged_dot). Several
    # pairs below deliberately share the same num_tokens (e.g. (1,2048) and
    # (2,1024)) as a sanity check: matching latency at matching num_tokens is
    # expected, not a coincidence, and a divergence there would flag a bug.
    (1, 128),
    (1, 512),
    (1, 2048),
    (2, 1024),
    (1, 4096),
    (4, 1024),
)


def run_latency_sweep(
    seed: int = 0,
    num_experts: int = 64,
    shapes: tuple[tuple[int, int], ...] = _DEFAULT_LATENCY_SWEEP_SHAPES,
) -> None:
  """Latency across multiple batch sizes/sequence lengths, per Zifan's
  explicit standing request (2026-08-28, after reviewing the published
  kernel-lab repo): "include the latency for different batch size/sequence
  length whenever you make an optimization to the kernel for comparison."

  Same single-chip-shard scale and heuristic-only (skip_autotune) approach
  as run_benchmark/run_fair_baseline -- autotuning this shape was confirmed
  impractically slow on real hardware (~2400 microbenchmarks, still not
  done after >12 minutes on op-call 1/6, see run_fair_baseline's docstring).
  This function differs only in sweeping several (batch_size, seq_len)
  pairs in one run and reporting them as a single comparison table, instead
  of one hardcoded num_tokens value.

  A NotImplementedError from a given implementation at a given shape (e.g.
  mosaic (v1) below its 128-row tiling floor at small num_tokens) is
  reported as SKIPPED in the table, not a failure or a missing row.
  """
  config = single_chip_kimi_k3_config(num_experts)
  rows: list[tuple[int, int, int, str, float | None, float | None, str | None]] = []

  for batch_size, seq_len in shapes:
    num_tokens = batch_size * seq_len
    key = jax.random.key(hash((seed, batch_size, seq_len)) % (2**31))
    key_w, key_x = jax.random.split(key)
    weights = init_weights(config, key_w, dtype=jnp.bfloat16)
    hidden_states = jax.random.normal(
        key_x, (num_tokens, config.hidden_size), dtype=jnp.bfloat16
    )

    for impl in ("xla", "mosaic", "mosaic_tpu_v2"):
      try:
        # NOTE: no `impl=impl` default parameter here (unlike a naive
        # closure-in-loop fix) -- tokamax.standardize_function introspects
        # the wrapped function's signature and threads every parameter
        # (defaults included) through as a jax.jit-traced argument. Adding
        # `impl` as a real parameter made it try to abstractify the string
        # 'xla' as a JAX array (confirmed on hardware: "TypeError: Argument
        # 'xla' ... is not a valid JAX type"). Closing over the loop
        # variable via a two-parameter lambda instead (matching
        # run_benchmark/run_fair_baseline's proven pattern) is safe here --
        # the lambda is built AND used within the same loop iteration, so
        # there's no deferred-call late-binding hazard.
        f_impl = jax.jit(
            lambda h, w: latent_moe_forward_ragged_dot(h, w, config, implementation=impl)
        )
        std_f, args = tokamax.standardize_function(f_impl, hidden_states, weights)
        bench = tokamax.benchmark(jax.jit(std_f), args, method="hermetic_xprof")
        rows.append(
            (batch_size, seq_len, num_tokens, impl,
             bench.median_evaluation_time_ms, bench.peak_memory_mb, None)
        )
      except NotImplementedError as e:
        rows.append((batch_size, seq_len, num_tokens, impl, None, None, str(e)))

  print(
      f"\n[latency-sweep] single-chip Kimi K3 (G={num_experts}) -- "
      "heuristic (untuned) latency across batch_size/seq_len:"
  )
  header = f"{'batch':>6} {'seq_len':>8} {'num_tokens':>11} {'impl':>14} {'median_exec_ms':>15} {'peak_mem_mb':>12}"
  print(header)
  for batch_size, seq_len, num_tokens, impl, exec_ms, mem_mb, err in rows:
    if err is not None:
      print(
          f"{batch_size:>6} {seq_len:>8} {num_tokens:>11} {impl:>14} "
          f"{'SKIPPED':>15} {'':>12}  ({err})"
      )
    else:
      print(
          f"{batch_size:>6} {seq_len:>8} {num_tokens:>11} {impl:>14} "
          f"{exec_ms:>15.4f} {mem_mb:>12.2f}"
      )


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--correctness", action="store_true")
  parser.add_argument("--benchmark", action="store_true")
  parser.add_argument("--fair-baseline", action="store_true")
  parser.add_argument(
      "--latency-sweep",
      action="store_true",
      help="latency across multiple batch_size/seq_len pairs in one table -- see "
      "run_latency_sweep's docstring (Zifan's 2026-08-28 standing request)",
  )
  parser.add_argument(
      "--shard-workload",
      action="store_true",
      help="isolated expert-kernel benchmark using a realistic 16-of-896-filtered-to-local-64 "
      "group_sizes distribution, instead of single_chip_kimi_k3_config's dense/uniform one -- "
      "see run_shard_workload_benchmark's docstring (WP-Kimi step 2b, part 1)",
  )
  parser.add_argument(
      "--route-filter-correctness",
      action="store_true",
      help="standalone correctness + padding/overflow test for route_and_filter_to_local_shard "
      "(REAL 896-expert routing filtered to a local shard) -- no tokamax dependency, runs "
      "anywhere. See check_route_and_filter_correctness's docstring (WP-Kimi step 2b, part 2)",
  )
  parser.add_argument(
      "--autotune",
      action="store_true",
      help="also run the exhaustive autotune search in --fair-baseline (confirmed VERY slow "
      "at Kimi's real per-expert shape -- see run_fair_baseline's docstring; off by default)",
  )
  args = parser.parse_args()

  if (
      not args.correctness
      and not args.benchmark
      and not args.fair_baseline
      and not args.shard_workload
      and not args.route_filter_correctness
      and not args.latency_sweep
  ):
    args.correctness = True  # default to the cheap check

  if args.correctness:
    ok_toy = check_correctness()
    ok_mosaic = check_mosaic_correctness()
    assert ok_toy, "ragged_dot forward pass diverges from the naive reference at toy scale -- fix before benchmarking"
    assert ok_mosaic, (
        "ragged_dot forward pass diverges from the naive reference at Mosaic-compatible "
        "scale -- fix before benchmarking"
    )

  if args.benchmark:
    run_benchmark()

  if args.fair_baseline:
    run_fair_baseline(skip_autotune=not args.autotune)

  if args.shard_workload:
    ok_shard = check_shard_workload_correctness()
    if not ok_shard:
      print(
          "[shard-workload] correctness check FAILED -- benchmark numbers below are still "
          "printed but should not be trusted until this is fixed"
      )
    run_shard_workload_benchmark()

  if args.route_filter_correctness:
    ok_route = check_route_and_filter_correctness()
    assert ok_route, "route_and_filter_to_local_shard failed its standalone correctness/overflow check"

  if args.latency_sweep:
    run_latency_sweep()
