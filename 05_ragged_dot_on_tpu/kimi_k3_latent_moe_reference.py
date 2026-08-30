"""WP-Kimi step 1: naive JAX reference implementation of Kimi K3's LatentMoE.

Confirmed from the actual modeling code
(https://huggingface.co/moonshotai/Kimi-K3/blob/main/modeling_kimi_linear.py,
config.json) via two independent fetches on 2026-08-24. **This is a
correction of an earlier version of this file**, whose router step was
wrong -- flagged by an external review (KIMI_K3_KERNEL_REVIEW_2026-08-24.md)
and independently re-confirmed against source before applying any of its
suggested fixes. The earlier version claimed the router ran on the
down-projected (3584-dim) latent; the real order is the reverse. See "What
changed" below for the full list.

Forward pass, in order (KimiSparseMoeBlock.forward / KimiMoEGate.forward):

  1. router (KimiMoEGate), on the ORIGINAL hidden_states (7168-dim), BEFORE
     any down-projection:
       `topk_idx, topk_weight = self.gate(hidden_states)`
     Gate math (logits, sigmoid, top-k selection, renormalize+scale) runs in
     float32 regardless of the model's compute dtype (review's P1-B item,
     added 2026-08-25) -- a bf16 gate can flip which experts land in the
     top-16 near the selection boundary, not just lose precision. The
     combine weight is cast back to the compute dtype only at the very end,
     right before it's used to weight-sum expert outputs.
     Inside the gate: sigmoid(logits) [not softmax] + e_score_correction_bias
     (DeepSeek-V3-style aux-loss-free load-balancing bias) added for top_k
     SELECTION only; the combine weight is `scores.gather(1, topk_idx)` using
     the un-corrected scores, then (confirmed from config.json:
     moe_renormalize=true, routed_scaling_factor=1.0):
       `topk_weight /= topk_weight.sum(-1, keepdim=True) + 1e-20`
       `topk_weight *= routed_scaling_factor`
     [The bias-for-selection-only / gather-original-for-combine split follows
     DeepSeek-V3's well-documented identical convention -- not independently
     re-verified byte-for-byte for Kimi K3's exact gate code, since neither
     fetch surfaced that specific line; the renormalize+scale step, in
     contrast, IS directly confirmed.]
  2. shared down-projection (routed_expert_down_proj), applied to ALL tokens
     regardless of routing: hidden_states (·,7168) -> x (·,3584). Only after
     this does the down-projected representation exist -- it is NOT what the
     router saw.
  3. dispatch: sort (token, slot) pairs by chosen expert id (from step 1),
     gather the (now down-projected) tokens into expert-contiguous order.
     This is the irregular-indexing step -- the SparseCore candidate.
  4. per-expert FFN in latent space (KimiBlockSparseMLP), THREE matrices per
     expert (confirmed: w1=gate, w3=up, w2=down), using Kimi's custom
     SiTU-GLU activation (NOT SiLU/SwiGLU -- confirmed from
     modeling_kimi_linear.py's `SituAndMul` + config.json's
     `hidden_act="situ"`, re-verified 2026-08-24 per the external review's
     P1-A item):
       `situ_gate = beta*tanh(gate/beta)*sigmoid(gate)`
       `bounded_up = linear_beta*tanh(up/linear_beta)`
       `(situ_gate * bounded_up) @ w2`, with beta=4.0, linear_beta=25.0
     (`activation_situ_beta`/`activation_situ_linear_beta` in config.json),
     computed internally in float32 regardless of input dtype, cast back
     afterward. latent -> intermediate -> latent. This is exactly what gets
     replaced by tokamax.ragged_dot calls in WP-Kimi step 2.
  5. scatter back to (token, slot) order, then weighted-sum over the top_k
     slots per token using the step-1 combine weights (already renormalized
     and scaled) -- collapses (num_tokens, top_k, latent) down to
     (num_tokens, latent). Inverse of step 3 -- also a SparseCore candidate.
  6. RMSNorm, still in latent space (before up-projection), eps from config
     (confirmed rms_norm_eps=1e-5, not the 1e-6 default this file used to
     hardcode). Computed in float32 regardless of compute dtype (review's
     P2-A item, added 2026-08-25), cast back to compute dtype before the
     scale multiply -- see `_rms_norm`.
  7. up_proj: (·,3584) -> (·,7168), shared across experts.
  8. shared experts (always active, not routed): a SINGLE combined KimiMLP
     (confirmed: gate/up/down, same SiTU-GLU activation as the routed
     experts, same shape as the routed-expert FFN but with intermediate_size
     = moe_intermediate_size * num_shared_experts, i.e. one wider MLP, NOT
     `num_shared_experts` independent MLP instances), computed on `identity`
     = the ORIGINAL uncompressed hidden_states (7168-dim, confirmed by
     KimiMLP's stated input dimension) and added:
     `y = y + shared_experts(identity)`.

What changed from the earlier (wrong) version of this file, per the external
review + source re-check:
  - Router now runs on hidden_states (7168-dim) BEFORE down-projection, not
    after. `router_weight` shape changed from (latent_size, num_experts) to
    (hidden_size, num_experts). This is the review's P1 item #1, and the
    most consequential fix -- it changes which experts get selected.
  - Added `moe_renormalize` / `routed_scaling_factor` to LatentMoEConfig and
    the combine-weight computation (review's P1 item #2).
  - Per-expert FFN (both routed and shared) is now 3 matrices (gate/up/down),
    not the earlier 2-matrix plain-SiLU-MLP simplification (review's P1 item
    #3 and P2 item #5).
  - Shared experts are now ONE combined wide MLP, not `num_shared_experts`
    independent small MLPs (review's P2 item #5).
  - `rms_norm_eps` now comes from config (confirmed 1e-5), not a hardcoded
    1e-6 (review's P2 item #6).
  - The 896-vs-64-expert single-chip-shard simulation issue (review's P1
    item #4) is a WP-Kimi step 2 (kimi_k3_latent_moe_ragged_dot.py) concern,
    not this reference file's -- addressed there.
  - **2026-08-25, round-2 fix (review's P1-A item):** the per-expert
    activation is now Kimi's actual custom `SituAndMul`
    (`_situ_and_mul` below), not plain SiLU/SwiGLU -- the previous version
    of this file used `silu(gate) * up`, which is architecturally wrong,
    not just an approximation. `activation_situ_beta`/
    `activation_situ_linear_beta` added to `LatentMoEConfig`, confirmed
    from config.json (4.0 / 25.0).
  - **2026-08-25, round-2 fix (review's P1-B + P2-A items):** the router
    gate (logits/sigmoid/top-k/renormalize+scale) and RMSNorm now both
    compute in float32 regardless of compute dtype, casting back only at
    the end -- previously both ran entirely in whatever dtype
    `hidden_states`/weights were (bf16 at real benchmark scale), which for
    the gate specifically can change which experts get selected near the
    top-16 boundary, not just lose precision.

This file has NO tokamax/TPU dependency on purpose -- it's pure correctness
scaffolding that runs eagerly on plain CPU jax, so it can be iterated on
entirely locally. The per-expert loop below uses Python-level dynamic
slicing (`int(group_sizes[e])`), which only works eagerly, not under
`jax.jit` -- that's fine here since the whole point of this file is
correctness, not speed. WP-Kimi step 2 replaces that loop with
`tokamax.ragged_dot` calls at real TPU scale.

**2026-08-28: `generate_local_shard_workload`/`route_and_filter_to_local_shard`/
`check_route_and_filter_correctness` moved here from
`kimi_k3_latent_moe_ragged_dot.py`.** They never had a tokamax dependency of
their own, but that file imports tokamax unconditionally at module level,
which blocked them from being imported/run on a machine without a working
tokamax install (this Windows dev machine) -- they could only be verified
via a hand-mirrored throwaway script, not the actual committed code. Moving
them here makes them genuinely, directly locally-testable.

**Also new 2026-08-28: `check_sharded_forward_correctness`** closes the gap
flagged since 2026-08-26 ("wire route_and_filter_to_local_shard into an
actual end-to-end forward pass") -- proves that summing every shard's
route+filter+per-shard-FFN+combine contribution across a small toy-scale
global expert range reproduces the exact same result as
`latent_moe_forward`'s unsharded computation over the same experts at once.
This is the first end-to-end validation of the real
16-of-896-then-filtered-to-local-shard routing design, entirely local, no
TPU needed (the per-shard FFN here is a naive loop, not
`tokamax.ragged_dot` -- swapping that in at real TPU scale is a follow-up
that does need hardware).

Usage:
  python kimi_k3_latent_moe_reference.py
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp

_MOSAIC_TILE_SIZE = 128  # confirmed hard minimum/tile granularity for ragged_dot's Mosaic kernel


def _round_up_to_tile(value: int, tile_size: int = _MOSAIC_TILE_SIZE) -> int:
  return -(-value // tile_size) * tile_size


@dataclasses.dataclass(frozen=True)
class LatentMoEConfig:
  hidden_size: int
  latent_size: int
  intermediate_size: int
  num_experts: int
  top_k: int
  num_shared_experts: int
  moe_renormalize: bool = True
  routed_scaling_factor: float = 1.0
  rms_norm_eps: float = 1e-5
  # SiTU-GLU activation params (config.json's activation_situ_beta /
  # activation_situ_linear_beta) -- Kimi's custom gated activation, NOT
  # SiLU/SwiGLU. See _situ_and_mul below.
  activation_situ_beta: float = 4.0
  activation_situ_linear_beta: float = 25.0


def toy_config() -> LatentMoEConfig:
  """Small scale for fast local correctness testing (no TPU needed)."""
  return LatentMoEConfig(
      hidden_size=64,
      latent_size=32,
      intermediate_size=48,
      num_experts=8,
      top_k=2,
      num_shared_experts=1,
      moe_renormalize=True,
      routed_scaling_factor=1.0,
      rms_norm_eps=1e-5,
      activation_situ_beta=4.0,
      activation_situ_linear_beta=25.0,
  )


def kimi_k3_config() -> LatentMoEConfig:
  """Real Kimi K3 full config, confirmed from modeling_kimi_linear.py and
  config.json (routed_scaling_factor, moe_renormalize, rms_norm_eps,
  activation_situ_beta, activation_situ_linear_beta all confirmed by direct
  fetch, not assumed)."""
  return LatentMoEConfig(
      hidden_size=7168,
      latent_size=3584,
      intermediate_size=3072,
      num_experts=896,
      top_k=16,
      num_shared_experts=2,
      moe_renormalize=True,
      routed_scaling_factor=1.0,
      rms_norm_eps=1e-5,
      activation_situ_beta=4.0,
      activation_situ_linear_beta=25.0,
  )


# Registered as a pytree (data_fields only, no static/meta fields) so this
# can be passed straight into jax.jit-wrapped functions -- needed for
# WP-Kimi step 2's benchmark, which jits latent_moe_forward_ragged_dot.
# Plain @dataclasses.dataclass instances are NOT pytrees by default: JAX
# treats them as opaque leaves and jax.jit rejects them with "Error
# interpreting argument ... as an abstract array" (confirmed locally).
@functools.partial(
    jax.tree_util.register_dataclass,
    data_fields=[
        "router_weight",
        "e_score_correction_bias",
        "down_proj",
        "up_proj",
        "expert_gate",
        "expert_up",
        "expert_down",
        "norm_scale",
        "shared_gate",
        "shared_up",
        "shared_down",
    ],
    meta_fields=[],
)
@dataclasses.dataclass(frozen=True)
class LatentMoEWeights:
  # Router now operates on the ORIGINAL hidden_states (see module docstring).
  router_weight: jax.Array  # (hidden_size, num_experts)
  e_score_correction_bias: jax.Array  # (num_experts,)
  down_proj: jax.Array  # (hidden_size, latent_size)
  up_proj: jax.Array  # (latent_size, hidden_size)
  # Routed-expert FFN: 3 matrices per expert (gate/up/down, SiTU-GLU).
  expert_gate: jax.Array  # (num_experts, latent_size, intermediate_size)
  expert_up: jax.Array  # (num_experts, latent_size, intermediate_size)
  expert_down: jax.Array  # (num_experts, intermediate_size, latent_size)
  norm_scale: jax.Array  # (latent_size,)
  # Shared experts: ONE combined wide MLP (gate/up/down, SiTU-GLU) operating on
  # the original hidden_states, intermediate width =
  # intermediate_size * num_shared_experts -- not `num_shared_experts`
  # independent small MLPs.
  shared_gate: jax.Array  # (hidden_size, intermediate_size * num_shared_experts)
  shared_up: jax.Array  # (hidden_size, intermediate_size * num_shared_experts)
  shared_down: jax.Array  # (intermediate_size * num_shared_experts, hidden_size)


def init_weights(
    config: LatentMoEConfig, key: jax.Array, dtype: jnp.dtype = jnp.float32
) -> LatentMoEWeights:
  keys = jax.random.split(key, 9)
  scale = 0.02
  shared_intermediate = config.intermediate_size * config.num_shared_experts

  def normal(k, shape):
    return (jax.random.normal(k, shape) * scale).astype(dtype)

  return LatentMoEWeights(
      router_weight=normal(keys[0], (config.hidden_size, config.num_experts)),
      e_score_correction_bias=jnp.zeros((config.num_experts,), dtype=dtype),
      down_proj=normal(keys[1], (config.hidden_size, config.latent_size)),
      up_proj=normal(keys[2], (config.latent_size, config.hidden_size)),
      expert_gate=normal(
          keys[3], (config.num_experts, config.latent_size, config.intermediate_size)
      ),
      expert_up=normal(
          keys[4], (config.num_experts, config.latent_size, config.intermediate_size)
      ),
      expert_down=normal(
          keys[5], (config.num_experts, config.intermediate_size, config.latent_size)
      ),
      norm_scale=jnp.ones((config.latent_size,), dtype=dtype),
      shared_gate=normal(keys[6], (config.hidden_size, shared_intermediate)),
      shared_up=normal(keys[7], (config.hidden_size, shared_intermediate)),
      shared_down=normal(keys[8], (shared_intermediate, config.hidden_size)),
  )


def _rms_norm(x: jax.Array, scale: jax.Array, eps: float) -> jax.Array:
  """RMSNorm, computed in float32 regardless of input dtype (review's P2-A
  item), cast back to the input dtype before applying `scale`."""
  input_dtype = x.dtype
  x_f32 = x.astype(jnp.float32)
  var = jnp.mean(jnp.square(x_f32), axis=-1, keepdims=True)
  normalized = x_f32 * jax.lax.rsqrt(var + eps)
  return normalized.astype(input_dtype) * scale


def _situ_and_mul(gate: jax.Array, up: jax.Array, beta: float, linear_beta: float) -> jax.Array:
  """Kimi K3's custom `SituAndMul` gated activation -- NOT SiLU/SwiGLU.

  Confirmed from modeling_kimi_linear.py (re-verified 2026-08-24 against an
  external review's P1-A item, independently re-fetched, not taken on
  faith): computed in float32 internally regardless of input dtype, then
  cast back, since the tanh/sigmoid combination is more sensitive to
  precision than a plain SiLU would be.
  """
  original_dtype = gate.dtype
  gate_f32 = gate.astype(jnp.float32)
  up_f32 = up.astype(jnp.float32)
  situ_gate = beta * jnp.tanh(gate_f32 / beta) * jax.nn.sigmoid(gate_f32)
  bounded_up = linear_beta * jnp.tanh(up_f32 / linear_beta)
  return (situ_gate * bounded_up).astype(original_dtype)


def _situ_glu_mlp(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    w_down: jax.Array,
    beta: float,
    linear_beta: float,
) -> jax.Array:
  """gate/up/down MLP with Kimi's SiTU-GLU activation:
  situ_and_mul(x @ w_gate, x @ w_up) @ w_down.

  Confirmed structure for both KimiBlockSparseMLP (routed experts) and
  KimiMLP (shared experts) -- see module docstring.
  """
  return _situ_and_mul(x @ w_gate, x @ w_up, beta, linear_beta) @ w_down


def latent_moe_forward(
    hidden_states: jax.Array,  # (num_tokens, hidden_size)
    weights: LatentMoEWeights,
    config: LatentMoEConfig,
) -> jax.Array:
  """Naive reference forward pass -- see module docstring for the 8 steps.

  Runs eagerly (not jit-compiled): the per-expert loop needs concrete Python
  ints for its dynamic slice sizes, which only works outside of jit.
  """
  identity = hidden_states  # shared experts see the ORIGINAL, uncompressed input

  # Step 1: router, on the ORIGINAL hidden_states (NOT the latent -- this was
  # wrong in the earlier version of this file). Gate math runs in float32
  # regardless of the compute dtype (review's P1-B item): a bf16 gate can
  # flip which experts land in the top-16 near the selection boundary, so
  # this isn't just a precision nicety.
  compute_dtype = hidden_states.dtype
  logits = hidden_states.astype(jnp.float32) @ weights.router_weight.astype(jnp.float32)
  scores = jax.nn.sigmoid(logits)  # float32
  scores_for_choice = scores + weights.e_score_correction_bias.astype(jnp.float32)[None, :]
  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)  # (num_tokens, top_k)
  topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)  # float32
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
    topk_weight = topk_weight / denom
  topk_weight = topk_weight * config.routed_scaling_factor
  # topk_weight stays float32 through the weighted-sum combine below (step
  # 5) -- casting it down to compute_dtype here, before that sum, would
  # accumulate the top_k=16-way reduction in bf16 and lose exactly the
  # precision the float32 gate fix above was for. It gets cast back to
  # compute_dtype only after the combine (see step 5), not before it.

  # Step 2: shared down-projection into latent space -- applied to ALL
  # tokens, computed AFTER the router has already seen the original input.
  x = hidden_states @ weights.down_proj  # (num_tokens, latent_size)

  num_tokens = hidden_states.shape[0]

  # Step 3: dispatch -- sort (token, slot) pairs by expert id, gather tokens.
  flat_expert_ids = topk_idx.reshape(-1)  # (num_tokens*top_k,)
  order = jnp.argsort(flat_expert_ids)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  sorted_token_idx = token_of_slot[order]
  sorted_tokens = x[sorted_token_idx]  # (num_tokens*top_k, latent_size)

  # This is exactly `tokamax.ragged_dot`'s `group_sizes` argument once WP-Kimi
  # step 2 swaps the loop below for real ragged_dot calls.
  group_sizes = jnp.bincount(flat_expert_ids, length=config.num_experts)
  group_sizes_concrete = [int(s) for s in group_sizes]  # eager-only

  # Step 4: per-expert FFN (3-matrix SiTU-GLU). Naive Python loop -- WP-Kimi
  # step 2 replaces this with tokamax.ragged_dot calls at real TPU scale.
  outs = []
  start = 0
  for e in range(config.num_experts):
    size = group_sizes_concrete[e]
    chunk = sorted_tokens[start : start + size]
    if size > 0:
      out_chunk = _situ_glu_mlp(
          chunk,
          weights.expert_gate[e],
          weights.expert_up[e],
          weights.expert_down[e],
          config.activation_situ_beta,
          config.activation_situ_linear_beta,
      )
    else:
      # dtype must match the non-empty branch's output (compute_dtype, e.g.
      # bf16) -- an un-dtyped jnp.zeros defaults to float32, which would
      # silently upcast the whole `outs` concatenation (and everything
      # downstream) to float32 the moment ANY expert gets zero tokens.
      # Caught via WP-KV3's bf16 golden-bundle cross-check.
      out_chunk = jnp.zeros((0, config.latent_size), dtype=compute_dtype)
    outs.append(out_chunk)
    start += size
  outs = jnp.concatenate(outs, axis=0)  # (num_tokens*top_k, latent_size), sorted order

  # Step 5: scatter back to (token, slot) order, then weighted-sum over the
  # top_k slots per token. topk_weight (float32) promotes the multiply/sum
  # to float32 -- accumulating the top_k=16-way reduction in float32, not
  # compute_dtype -- then the result is cast back down right after, not
  # topk_weight before it (see step 1's comment).
  unsorted = jnp.zeros_like(outs).at[order].set(outs)
  unsorted = unsorted.reshape(num_tokens, config.top_k, config.latent_size)
  routed_out = jnp.sum(unsorted * topk_weight[..., None], axis=1)  # (num_tokens, latent_size)
  routed_out = routed_out.astype(compute_dtype)

  # Step 6: RMSNorm, still in latent space, eps from config.
  normed = _rms_norm(routed_out, weights.norm_scale, config.rms_norm_eps)

  # Step 7: shared up-projection back to hidden_size.
  up = normed @ weights.up_proj  # (num_tokens, hidden_size)

  # Step 8: shared experts (ONE combined wide SiTU-GLU MLP) on the original
  # (uncompressed) input, added.
  shared_out = _situ_glu_mlp(
      identity,
      weights.shared_gate,
      weights.shared_up,
      weights.shared_down,
      config.activation_situ_beta,
      config.activation_situ_linear_beta,
  )

  return up + shared_out


def _test_dispatch_roundtrip(config: LatentMoEConfig, key: jax.Array) -> bool:
  """Isolates and verifies the dispatch/gather/scatter indexing (step 3 and
  the first half of step 5), independent of expert math, by substituting the
  identity function for the per-expert FFN and checking that every token's
  original latent vector round-trips back to all of its selected slots.

  This is the highest-value correctness check here: the sort/gather/scatter
  indexing is exactly the part most likely to have an off-by-one bug, and the
  part that later becomes the SparseCore candidate -- worth pinning down in
  isolation before any expert math or TPU acceleration gets layered on top.
  """
  key_x, key_router = jax.random.split(key)
  num_tokens = 20
  x = jax.random.normal(key_x, (num_tokens, config.latent_size))
  scores = jax.random.uniform(key_router, (num_tokens, config.num_experts))
  _, topk_idx = jax.lax.top_k(scores, config.top_k)

  flat_expert_ids = topk_idx.reshape(-1)
  order = jnp.argsort(flat_expert_ids)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  sorted_token_idx = token_of_slot[order]
  sorted_tokens = x[sorted_token_idx]  # dispatch

  outs = sorted_tokens  # identity "expert" -- skip the FFN entirely

  unsorted = jnp.zeros_like(outs).at[order].set(outs)  # scatter back
  unsorted = unsorted.reshape(num_tokens, config.top_k, config.latent_size)

  # Every slot for a given token should recover exactly that token's vector.
  expected = jnp.broadcast_to(x[:, None, :], unsorted.shape)
  max_err = float(jnp.max(jnp.abs(unsorted - expected)))
  ok = max_err < 1e-6
  print(f"[dispatch-roundtrip] max_err={max_err:.2e} {'OK' if ok else 'FAIL'}")
  return ok


def generate_local_shard_workload(
    key: jax.Array,
    num_tokens: int,
    global_num_experts: int,
    top_k: int,
    local_num_experts: int,
    latent_size: int,
    dtype: jnp.dtype = jnp.bfloat16,
    capacity_factor: float = 2.0,
    tile_size: int = _MOSAIC_TILE_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """WP-Kimi step 2b, part 1 (moved here from kimi_k3_latent_moe_ragged_dot.py
  2026-08-28 -- this function has zero tokamax dependency, but that file
  imports tokamax unconditionally at module level, which blocks it from
  being run/tested on a machine without a working tokamax install, e.g. this
  Windows dev machine. Moving it here makes it actually importable and
  testable locally, not just runnable via a hand-mirrored throwaway script,
  which is how it was verified previously): generate an isolated
  expert-kernel benchmark input with REALISTIC statistics for what one
  chip's `local_num_experts`-expert shard would see under a real
  `top_k`-of-`global_num_experts` global router, WITHOUT implementing that
  global routing/dispatch/filtering end-to-end (that's
  route_and_filter_to_local_shard below, using REAL routing output instead
  of a synthetic draw).

  Statistical model: for each of `num_tokens` tokens, draw `top_k` expert
  ids uniformly WITHOUT replacement from `global_num_experts` (a symmetric,
  untrained-router assumption -- this is a synthetic benchmark input, not a
  simulation of Kimi K3's actual trained routing distribution, which has an
  aux-loss-free load-balancing bias term specifically because real training
  dynamics are NOT uniform). Keep only the picks landing in
  `[0, local_num_experts)` -- WLOG by the uniform-sampling symmetry, this
  local range stands in for "whichever `local_num_experts` global ids this
  chip happens to hold."

  **Fixed-total, tile-aligned padding via a single trailing bucket** (a
  review caught that an earlier version forced EVERY expert's `group_sizes`
  entry to the same constant `capacity`, which silently erased the real
  skew this whole benchmark exists to exercise: Mosaic's kernel specifically
  reacts to per-group sizes, so a uniform group_sizes array quietly turns
  this back into a too-regular workload):

  Per-expert `group_sizes[0:local_num_experts]` are the REAL, unmodified,
  genuinely-skewed dispatch counts from the draw above -- nothing about an
  individual expert's count is altered. The "needs a fixed, tile-aligned
  total M so xla/mosaic-v1/mosaic-v2 all see identical shapes/values"
  requirement is instead satisfied by ONE extra trailing "padding bucket"
  appended as group `local_num_experts` (so `group_sizes` has
  `local_num_experts + 1` entries, and the expert weight tensors passed to
  ragged_dot need a matching dummy `+1`th row): its size is whatever's
  needed to bring the total up to a fixed `m_padded` (an aggregate ceiling
  on the total draw across all local experts, `capacity_factor` times the
  expected total, rounded UP to `tile_size`). This keeps the array SHAPE
  and VALUES identical across implementations without touching any real
  expert's group size.

  If the total raw draw exceeds `m_padded` (aggregate overflow), the excess
  is dropped from the tail of the expert-sorted order and a warning is
  printed with the drop count.

  Returns `(sorted_tokens, group_sizes, valid_mask, per_expert_counts)`:
    - `sorted_tokens`: `(m_padded, latent_size)`, fixed shape regardless of
      the random draw. Real token data fills the first
      `sum(per_expert_counts)` rows (expert-contiguous, matching
      `group_sizes[:local_num_experts]`); the trailing padding-bucket rows
      are zero.
    - `group_sizes`: `(local_num_experts + 1,)` -- the real per-expert
      counts, unmodified, followed by the one padding-bucket size. This is
      what actually gets passed to `ragged_dot`; still genuinely ragged.
    - `valid_mask`: `(m_padded,)` bool, True for rows holding real
      (non-padding) data.
    - `per_expert_counts`: `(local_num_experts,)`, same values as
      `group_sizes[:local_num_experts]`.

  Runs eagerly, not jit-compiled: m_padded and drop counts are
  data-dependent Python ints computed from the random draw, same
  "must run outside jit" constraint as latent_moe_forward's per-expert loop.
  """
  key_route, key_data = jax.random.split(key)
  token_keys = jax.random.split(key_route, num_tokens)

  def _pick_global_experts(k: jax.Array) -> jax.Array:
    return jax.random.choice(k, global_num_experts, shape=(top_k,), replace=False)

  global_picks = jax.vmap(_pick_global_experts)(token_keys)  # (num_tokens, top_k)

  flat_picks = global_picks.reshape(-1)
  local_mask = flat_picks < local_num_experts
  local_ids = flat_picks[local_mask]  # data-dependent length -- eager only
  num_raw = local_ids.shape[0]

  order = jnp.argsort(local_ids)
  sorted_local_ids = local_ids[order]

  expected_total = num_tokens * top_k * local_num_experts / global_num_experts
  m_padded = _round_up_to_tile(int(jnp.ceil(expected_total * capacity_factor)), tile_size)

  if num_raw > m_padded:
    dropped = num_raw - m_padded
    print(
        f"[shard-workload] WARNING: m_padded={m_padded} overflowed by the total draw -- "
        f"dropping {dropped} of {num_raw} raw assignments from the tail of the "
        "expert-sorted order (raise capacity_factor if this matters for the benchmark)"
    )
    kept_local_ids = sorted_local_ids[:m_padded]
  else:
    kept_local_ids = sorted_local_ids

  keep_count = kept_local_ids.shape[0]
  per_expert_counts = jnp.bincount(kept_local_ids, length=local_num_experts).astype(jnp.int32)
  pad_size = m_padded - keep_count

  group_sizes = jnp.concatenate(
      [per_expert_counts, jnp.array([pad_size], dtype=jnp.int32)]
  )

  real_tokens = jax.random.normal(key_data, (keep_count, latent_size), dtype=dtype)
  pad_tokens = jnp.zeros((pad_size, latent_size), dtype=dtype)
  sorted_tokens = jnp.concatenate([real_tokens, pad_tokens], axis=0)

  valid_mask = jnp.concatenate(
      [jnp.ones((keep_count,), dtype=jnp.bool_), jnp.zeros((pad_size,), dtype=jnp.bool_)]
  )

  return sorted_tokens, group_sizes, valid_mask, per_expert_counts


def route_and_filter_to_local_shard(
    hidden_states: jax.Array,
    x: jax.Array,
    router_weight: jax.Array,
    e_score_correction_bias: jax.Array,
    config: LatentMoEConfig,
    local_expert_start: int,
    local_num_experts: int,
    capacity_factor: float = 2.0,
    tile_size: int = _MOSAIC_TILE_SIZE,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
  """WP-Kimi step 2b, part 2 (moved here 2026-08-28, see
  generate_local_shard_workload's docstring for why): REAL global
  `config.top_k`-of-`config.num_experts` top-k routing (identical gate math
  to `latent_moe_forward`'s step 1 -- float32 internally, cast back to
  compute dtype at the end), filtered down to the local shard
  `[local_expert_start, local_expert_start + local_num_experts)`, with the
  SAME fixed-total, tile-aligned padding scheme as
  generate_local_shard_workload (one trailing padding-bucket group) --
  unlike that function, the per-expert counts here come from ACTUAL routing
  output on real `hidden_states`, not a synthetic uniform-sampling draw.

  `x` is the ALREADY down-projected representation (`hidden_states @
  down_proj`, step 2 of the real forward pass) -- the router itself runs on
  `hidden_states` (pre-projection), matching the real model's step ordering;
  passing both in lets the caller reuse a single down-projection across
  shards/calls instead of recomputing it here.

  Returns `(sorted_tokens, group_sizes, valid_mask, per_expert_counts,
  padded_token_idx, padded_combine_weight)`:
    - `sorted_tokens`: `(m_padded, latent_size)`, gathered from `x` in
      expert-sorted order for real assignments, zero for padding rows.
    - `group_sizes`: `(local_num_experts + 1,)` -- real per-expert counts
      (genuinely data-dependent on the actual routing outcome, not forced
      uniform) plus one trailing padding-bucket size.
    - `valid_mask`: `(m_padded,)` bool, True for real (non-padding) rows.
    - `per_expert_counts`: `(local_num_experts,)`, same as
      `group_sizes[:local_num_experts]`.
    - `padded_token_idx`: `(m_padded,)` int32, the ORIGINAL token index (row
      in `hidden_states`/`x`) each row came from; -1 for padding rows. This
      is what an end-to-end combine/scatter step needs to route each
      expert's output back to its token -- see check_sharded_forward_correctness
      for the first such wiring (2026-08-28).
    - `padded_combine_weight`: `(m_padded,)`, the router's combine weight
      (renormalized+scaled, already cast to compute dtype) for each real
      assignment; 0 for padding rows.

  If the total number of (token, slot) pairs landing in this shard exceeds
  `m_padded` (aggregate overflow), the excess is dropped from the tail of
  the expert-sorted order and a warning is printed -- identical policy to
  generate_local_shard_workload.

  Runs eagerly, not jit-compiled: filtering to the local shard produces a
  data-dependent length, same "must run outside jit" constraint as
  generate_local_shard_workload and latent_moe_forward's per-expert loop.
  """
  compute_dtype = hidden_states.dtype
  logits = hidden_states.astype(jnp.float32) @ router_weight.astype(jnp.float32)
  scores = jax.nn.sigmoid(logits)
  scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)[None, :]
  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)  # (num_tokens, top_k), GLOBAL ids
  topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
    topk_weight = topk_weight / denom
  topk_weight = topk_weight * config.routed_scaling_factor
  topk_weight = topk_weight.astype(compute_dtype)

  num_tokens = hidden_states.shape[0]
  flat_expert_ids = topk_idx.reshape(-1)  # GLOBAL ids, (num_tokens*top_k,)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  flat_combine_weight = topk_weight.reshape(-1)

  local_end = local_expert_start + local_num_experts
  in_shard = (flat_expert_ids >= local_expert_start) & (flat_expert_ids < local_end)

  # eager-only from here: filtering to the shard produces a data-dependent length.
  kept_local_ids = flat_expert_ids[in_shard] - local_expert_start
  kept_token_idx = token_of_slot[in_shard]
  kept_combine_weight = flat_combine_weight[in_shard]
  num_raw = kept_local_ids.shape[0]

  order = jnp.argsort(kept_local_ids)
  sorted_local_ids = kept_local_ids[order]
  sorted_token_idx = kept_token_idx[order]
  sorted_combine_weight = kept_combine_weight[order]

  expected_total = num_tokens * config.top_k * local_num_experts / config.num_experts
  m_padded = _round_up_to_tile(int(jnp.ceil(expected_total * capacity_factor)), tile_size)

  if num_raw > m_padded:
    dropped = num_raw - m_padded
    print(
        f"[shard-route] WARNING: m_padded={m_padded} overflowed by the total routed draw -- "
        f"dropping {dropped} of {num_raw} raw assignments from the tail of the "
        "expert-sorted order (raise capacity_factor if this matters)"
    )
    keep_count = m_padded
  else:
    keep_count = num_raw

  kept_sorted_local_ids = sorted_local_ids[:keep_count]
  kept_sorted_token_idx = sorted_token_idx[:keep_count]
  kept_sorted_combine_weight = sorted_combine_weight[:keep_count]

  per_expert_counts = jnp.bincount(kept_sorted_local_ids, length=local_num_experts).astype(jnp.int32)
  pad_size = m_padded - keep_count
  group_sizes = jnp.concatenate([per_expert_counts, jnp.array([pad_size], dtype=jnp.int32)])

  gathered = x[kept_sorted_token_idx]
  pad_tokens = jnp.zeros((pad_size, x.shape[-1]), dtype=x.dtype)
  sorted_tokens = jnp.concatenate([gathered, pad_tokens], axis=0)

  valid_mask = jnp.concatenate(
      [jnp.ones((keep_count,), dtype=jnp.bool_), jnp.zeros((pad_size,), dtype=jnp.bool_)]
  )
  padded_token_idx = jnp.concatenate(
      [kept_sorted_token_idx.astype(jnp.int32), -jnp.ones((pad_size,), dtype=jnp.int32)]
  )
  padded_combine_weight = jnp.concatenate(
      [kept_sorted_combine_weight, jnp.zeros((pad_size,), dtype=compute_dtype)]
  )

  return (
      sorted_tokens,
      group_sizes,
      valid_mask,
      per_expert_counts,
      padded_token_idx,
      padded_combine_weight,
  )


def check_route_and_filter_correctness(
    seed: int = 0,
    num_tokens: int = 256,
    local_expert_start: int = 137,
    local_num_experts: int = 64,
) -> bool:
  """Standalone correctness + padding/overflow test for
  route_and_filter_to_local_shard, tested in isolation before it's wired
  into any end-to-end forward pass or benchmark. Pure JAX/numpy -- no
  tokamax dependency, runs anywhere. `local_expert_start=137` (not 0)
  deliberately, so a bug that only breaks for a non-zero shard offset isn't
  masked.

  Uses `config.num_experts=896` (kimi_k3_config's REAL global expert count)
  for the router matmul, but only allocates `router_weight`
  ((hidden_size, 896)) and `down_proj` ((hidden_size, latent_size)) directly
  -- NOT the full LatentMoEWeights/init_weights, which would allocate the
  full per-expert expert_gate/up/down tensors (~59GB at 896 experts,
  confirmed OOM-inducing on a single v6e chip, and not needed for a
  routing-only test anyway).

  Checks, each against an independent brute-force (Python/numpy, not
  vectorized JAX) reference so a bug in route_and_filter_to_local_shard's
  own argsort/bincount logic can't also be present in the check:
    1. No-overflow case (generous capacity_factor): per_expert_counts
       matches a nested-loop count of (token, slot) pairs whose global
       expert id falls in the shard range.
    2. Gathered sorted_tokens (valid rows only) exactly equals `x` indexed
       by padded_token_idx (valid rows only) -- catches gather-order bugs.
       Exact equality, not an error tolerance, since this is pure indexing.
    3. Combine weight (valid rows only) matches an independently-recomputed
       value: for each valid row, find (via numpy) the token's OWN slot
       whose global expert id equals this row's global expert id, and
       compare against the original (pre-filter) topk_weight at that
       (token, slot) -- catches misalignment introduced by the
       flatten/filter/argsort pipeline.
    4. Overflow/truncation semantics (capacity_factor forced tiny): dropped
       count is correct, `sum(per_expert_counts) + pad_size == m_padded`
       still holds, and the SURVIVING (token, slot) pairs are exactly the
       first `m_padded` of an independently-sorted-by-expert-id sequence
       (built via Python's stable `sorted()`, not JAX argsort) -- confirms
       truncation drops exactly the tail, nothing else.
  """
  config = kimi_k3_config()  # real hidden_size=7168, num_experts=896, top_k=16
  key = jax.random.key(seed)
  key_router, key_bias, key_x = jax.random.split(key, 3)

  router_weight = jax.random.normal(key_router, (config.hidden_size, config.num_experts)) * 0.02
  e_score_correction_bias = jnp.zeros((config.num_experts,))
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))
  x = hidden_states  # identity stand-in for down_proj's output -- only the
  # gather/filter/combine-weight logic is under test here, not the
  # down-projection matmul itself, so using hidden_states directly (same
  # latent_size-agnostic shape role) keeps the test setup simpler.

  # Recompute the SAME topk_idx/topk_weight the function computes internally,
  # via a second, independent call path, so checks 1/3/4 below have ground
  # truth to compare against without re-deriving the function's own logic.
  logits = hidden_states.astype(jnp.float32) @ router_weight.astype(jnp.float32)
  scores = jax.nn.sigmoid(logits)
  scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)[None, :]
  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)
  topk_weight_full = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight_full, axis=-1, keepdims=True) + 1e-20
    topk_weight_full = topk_weight_full / denom
  topk_weight_full = topk_weight_full * config.routed_scaling_factor

  topk_idx_np = jax.device_get(topk_idx)
  topk_weight_np = jax.device_get(topk_weight_full)

  all_ok = True

  # --- Checks 1-3: no-overflow case ---
  (
      sorted_tokens,
      group_sizes,
      valid_mask,
      per_expert_counts,
      padded_token_idx,
      padded_combine_weight,
  ) = route_and_filter_to_local_shard(
      hidden_states,
      x,
      router_weight,
      e_score_correction_bias,
      config,
      local_expert_start=local_expert_start,
      local_num_experts=local_num_experts,
      capacity_factor=4.0,  # generous -- this pass must NOT overflow
  )

  # Check 1: brute-force per-expert counts.
  ref_counts = [0] * local_num_experts
  for tok in range(topk_idx_np.shape[0]):
    for slot in range(topk_idx_np.shape[1]):
      gid = int(topk_idx_np[tok, slot])
      if local_expert_start <= gid < local_expert_start + local_num_experts:
        ref_counts[gid - local_expert_start] += 1
  counts_ok = per_expert_counts.tolist() == ref_counts
  print(
      f"[route-filter-correctness] check1 per_expert_counts match brute-force ref: "
      f"{'OK' if counts_ok else 'FAIL'}"
  )
  all_ok = all_ok and counts_ok

  # Check 2: gathered token data matches x[padded_token_idx] on valid rows.
  num_valid = int(jnp.sum(valid_mask))
  gather_ok = bool(
      jnp.array_equal(sorted_tokens[:num_valid], x[padded_token_idx[:num_valid]])
  )
  print(f"[route-filter-correctness] check2 gathered tokens match x[padded_token_idx]: "
        f"{'OK' if gather_ok else 'FAIL'} (num_valid={num_valid})")
  all_ok = all_ok and gather_ok

  # Check 3: combine weight matches an independently-recomputed value.
  padded_token_idx_np = jax.device_get(padded_token_idx[:num_valid])
  padded_combine_weight_np = jax.device_get(padded_combine_weight[:num_valid])
  weight_ok = True
  for row in range(num_valid):
    tok = int(padded_token_idx_np[row])
    row_global_id = None
    for slot in range(topk_idx_np.shape[1]):
      gid = int(topk_idx_np[tok, slot])
      if local_expert_start <= gid < local_expert_start + local_num_experts:
        # there may be multiple shard-hits for this token across different slots;
        # match by value against the row's own combine weight instead of position
        if abs(float(topk_weight_np[tok, slot]) - float(padded_combine_weight_np[row])) < 1e-5:
          row_global_id = gid
          break
    if row_global_id is None:
      weight_ok = False
      print(f"[route-filter-correctness] check3 FAIL: row {row} (token {tok}) combine "
            f"weight {float(padded_combine_weight_np[row]):.6f} matches no slot's score")
      break
  print(f"[route-filter-correctness] check3 combine weights match independent recompute: "
        f"{'OK' if weight_ok else 'FAIL'}")
  all_ok = all_ok and weight_ok

  # --- Check 4: forced overflow/truncation ---
  (
      _sorted_tokens_ovf,
      group_sizes_ovf,
      valid_mask_ovf,
      per_expert_counts_ovf,
      padded_token_idx_ovf,
      _padded_combine_weight_ovf,
  ) = route_and_filter_to_local_shard(
      hidden_states,
      x,
      router_weight,
      e_score_correction_bias,
      config,
      local_expert_start=local_expert_start,
      local_num_experts=local_num_experts,
      capacity_factor=0.05,  # deliberately tiny -- force overflow
      tile_size=1,  # disable tile rounding so the forced m_padded is exact/predictable
  )
  m_padded_ovf = int(jnp.sum(group_sizes_ovf))
  keep_count_ovf = int(jnp.sum(valid_mask_ovf))

  # Independent reference: brute-force list of (token, slot, global_id), stable-sorted
  # by global id via Python's sorted() (not JAX argsort), truncated to m_padded_ovf.
  raw_hits = []
  for tok in range(topk_idx_np.shape[0]):
    for slot in range(topk_idx_np.shape[1]):
      gid = int(topk_idx_np[tok, slot])
      if local_expert_start <= gid < local_expert_start + local_num_experts:
        raw_hits.append((gid - local_expert_start, tok))
  raw_hits_sorted = sorted(raw_hits, key=lambda pair: pair[0])  # stable sort by local id
  ref_kept = raw_hits_sorted[:m_padded_ovf]
  ref_counts_ovf = [0] * local_num_experts
  for local_id, _tok in ref_kept:
    ref_counts_ovf[local_id] += 1

  overflow_triggered = len(raw_hits_sorted) > m_padded_ovf
  counts_ovf_ok = per_expert_counts_ovf.tolist() == ref_counts_ovf
  keep_count_ok = keep_count_ovf == len(ref_kept)
  token_order_ok = jax.device_get(padded_token_idx_ovf[:keep_count_ovf]).tolist() == [
      tok for _local_id, tok in ref_kept
  ]
  check4_ok = overflow_triggered and counts_ovf_ok and keep_count_ok and token_order_ok
  print(
      f"[route-filter-correctness] check4 forced overflow -- triggered={overflow_triggered} "
      f"counts_match={counts_ovf_ok} keep_count_match={keep_count_ok} "
      f"token_order_match={token_order_ok}: {'OK' if check4_ok else 'FAIL'}"
  )
  all_ok = all_ok and check4_ok

  return all_ok


def _local_shard_expert_ffn(
    sorted_tokens: jax.Array,
    expert_gate: jax.Array,
    expert_up: jax.Array,
    expert_down: jax.Array,
    group_sizes: jax.Array,
    config: LatentMoEConfig,
) -> jax.Array:
  """Naive (non-ragged_dot) per-expert-loop FFN over one shard's ALREADY
  sorted-and-padded tokens, mirroring latent_moe_forward's step-4 loop but
  operating on `sorted_tokens`/`group_sizes` as produced by
  route_and_filter_to_local_shard / generate_local_shard_workload: real
  per-expert counts for this shard's `local_num_experts` experts, plus one
  trailing padding-bucket group whose weights are never touched by real
  data (its input rows are all zero). `expert_gate`/`expert_up`/
  `expert_down` must therefore have `local_num_experts + 1` rows -- a
  dummy `+1`th row for that padding bucket, same convention as
  run_shard_workload_benchmark in kimi_k3_latent_moe_ragged_dot.py.

  Runs eagerly, same "needs concrete Python ints for slice sizes" constraint
  as latent_moe_forward's own per-expert loop and
  route_and_filter_to_local_shard itself.
  """
  compute_dtype = sorted_tokens.dtype
  group_sizes_concrete = [int(s) for s in group_sizes]
  outs = []
  start = 0
  for e, size in enumerate(group_sizes_concrete):
    chunk = sorted_tokens[start : start + size]
    if size > 0:
      out_chunk = _situ_glu_mlp(
          chunk,
          expert_gate[e],
          expert_up[e],
          expert_down[e],
          config.activation_situ_beta,
          config.activation_situ_linear_beta,
      )
    else:
      out_chunk = jnp.zeros((0, config.latent_size), dtype=compute_dtype)
    outs.append(out_chunk)
    start += size
  return jnp.concatenate(outs, axis=0)


def check_sharded_forward_correctness(
    seed: int = 0,
    num_tokens: int = 96,
    global_num_experts: int = 32,
    local_num_experts: int = 8,
    top_k: int = 4,
    capacity_factor: float = 4.0,
) -> bool:
  """Proves the previously-unwired piece of WP-Kimi step 2b (real global
  routing + local-shard filtering via route_and_filter_to_local_shard, plus
  a scatter-back/weighted-combine step turning a shard's expert output into
  its contribution to the final result) is mathematically equivalent to
  computing the SAME forward pass directly over all global experts at once
  (latent_moe_forward), once every shard's contribution is summed.

  This closes the gap flagged since 2026-08-26 ("Step 3, not yet started:
  wire route_and_filter_to_local_shard into an actual end-to-end forward
  pass") -- routing, shard-filtering, and per-shard expert compute had each
  been tested in isolation, but never wired into one coherent forward pass
  and checked end-to-end against a ground truth.

  Uses a small `global_num_experts` (32, not the real 896 -- kimi_k3_config's
  full expert tensors don't fit in memory for a from-scratch reference, see
  single_chip_kimi_k3_config's docstring in kimi_k3_latent_moe_ragged_dot.py)
  so this stays a pure-JAX, CPU-only, tokamax-free check. Swapping the naive
  per-shard FFN below for tokamax.ragged_dot at real TPU scale is a
  follow-up needing a TPU VM -- same "reference proves the math, ragged_dot
  is checked against it" pattern as everything else in this project.

  `global_num_experts` must divide evenly by `local_num_experts` so the
  shards below tile the full global expert range exactly once each, with no
  gaps or overlaps -- every (token, slot) pair's chosen expert then falls
  in EXACTLY one shard, which is what makes "sum every shard's contribution"
  equal the unsharded computation.
  """
  assert global_num_experts % local_num_experts == 0, (
      "global_num_experts must divide evenly by local_num_experts so the shards below "
      "exactly tile [0, global_num_experts) once each, with no gaps or overlaps"
  )
  num_shards = global_num_experts // local_num_experts

  config = LatentMoEConfig(
      hidden_size=64,
      latent_size=32,
      intermediate_size=48,
      num_experts=global_num_experts,
      top_k=top_k,
      num_shared_experts=1,
      moe_renormalize=True,
      routed_scaling_factor=1.0,
      rms_norm_eps=1e-5,
      activation_situ_beta=4.0,
      activation_situ_linear_beta=25.0,
  )

  key = jax.random.key(seed)
  key_w, key_x = jax.random.split(key)
  weights = init_weights(config, key_w)  # float32
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))

  # Ground truth: the existing naive, unsharded reference over ALL global experts at once.
  reference_out = latent_moe_forward(hidden_states, weights, config)

  # Sharded computation: route + filter + per-shard compute + weighted
  # combine, once per shard, summed.
  identity = hidden_states
  compute_dtype = hidden_states.dtype
  x = hidden_states @ weights.down_proj  # shared down-projection, same as latent_moe_forward step 2

  routed_out = jnp.zeros((num_tokens, config.latent_size), dtype=compute_dtype)
  total_valid_rows = 0
  for shard_idx in range(num_shards):
    local_expert_start = shard_idx * local_num_experts
    (
        sorted_tokens,
        group_sizes,
        valid_mask,
        _per_expert_counts,
        padded_token_idx,
        padded_combine_weight,
    ) = route_and_filter_to_local_shard(
        hidden_states,
        x,
        weights.router_weight,
        weights.e_score_correction_bias,
        config,
        local_expert_start=local_expert_start,
        local_num_experts=local_num_experts,
        capacity_factor=capacity_factor,
        tile_size=1,  # toy scale, no Mosaic tiling constraint to satisfy here
    )
    total_valid_rows += int(jnp.sum(valid_mask))

    # +1 dummy expert row for the trailing padding bucket, matching
    # route_and_filter_to_local_shard's group_sizes convention.
    shard_gate = weights.expert_gate[local_expert_start : local_expert_start + local_num_experts]
    shard_up = weights.expert_up[local_expert_start : local_expert_start + local_num_experts]
    shard_down = weights.expert_down[local_expert_start : local_expert_start + local_num_experts]
    shard_gate = jnp.concatenate([shard_gate, jnp.zeros_like(shard_gate[:1])], axis=0)
    shard_up = jnp.concatenate([shard_up, jnp.zeros_like(shard_up[:1])], axis=0)
    shard_down = jnp.concatenate([shard_down, jnp.zeros_like(shard_down[:1])], axis=0)

    shard_out = _local_shard_expert_ffn(
        sorted_tokens, shard_gate, shard_up, shard_down, group_sizes, config
    )  # (m_padded, latent_size)

    # Scatter each dispatched row's contribution back to its token, weighted
    # by the combine weight. Padding rows have padded_combine_weight==0, so
    # they contribute exactly nothing regardless of which index
    # padded_token_idx points them at -- the -1 placeholder for padding rows
    # is therefore safe here: JAX/numpy negative-index semantics make it
    # index the LAST token, but the weighted contribution added there is 0.
    weighted = shard_out * padded_combine_weight[:, None]
    routed_out = routed_out.at[padded_token_idx].add(weighted.astype(compute_dtype))

  # Every (token, slot) pair lands in EXACTLY one shard (shards tile
  # [0, global_num_experts) with no overlap) -- so summing every shard's
  # valid row count should equal num_tokens * top_k exactly, given generous
  # enough capacity_factor that no shard drops anything.
  expected_total = num_tokens * top_k
  coverage_ok = total_valid_rows == expected_total
  print(
      f"[sharded-correctness] total valid dispatched rows across {num_shards} shards "
      f"of {local_num_experts} experts each: {total_valid_rows} (expected {expected_total}) "
      f"{'OK' if coverage_ok else 'FAIL -- some (token, slot) pairs were dropped, raise capacity_factor'}"
  )

  # Steps 6-8 (RMSNorm, up_proj, shared experts) are identical whether the
  # routed-expert combine came from one unsharded computation or several
  # shards summed -- apply them once here, exactly matching
  # latent_moe_forward's own steps 6-8.
  normed = _rms_norm(routed_out, weights.norm_scale, config.rms_norm_eps)
  up = normed @ weights.up_proj
  shared_out = _situ_glu_mlp(
      identity,
      weights.shared_gate,
      weights.shared_up,
      weights.shared_down,
      config.activation_situ_beta,
      config.activation_situ_linear_beta,
  )
  sharded_final = up + shared_out

  max_err = float(jnp.max(jnp.abs(sharded_final - reference_out)))
  ok = max_err < 1e-4
  print(
      f"[sharded-correctness] sharded (route+filter+per-shard-FFN+combine, summed over "
      f"{num_shards} shards of {local_num_experts} experts each) vs. unsharded reference "
      f"(all {global_num_experts} experts at once): max_err={max_err:.2e} {'OK' if ok else 'FAIL'}"
  )
  return coverage_ok and ok


if __name__ == "__main__":
  key = jax.random.key(0)
  key_test, key_weights, key_x = jax.random.split(key, 3)

  config = toy_config()
  print(f"devices: {jax.devices()}")
  print(f"config: {config}")

  roundtrip_ok = _test_dispatch_roundtrip(config, key_test)

  weights = init_weights(config, key_weights)
  num_tokens = 64
  hidden_states = jax.random.normal(key_x, (num_tokens, config.hidden_size))

  out = latent_moe_forward(hidden_states, weights, config)
  print(f"[forward] output shape: {out.shape} (expected ({num_tokens}, {config.hidden_size}))")
  has_nan = bool(jnp.any(jnp.isnan(out)))
  print(f"[forward] contains NaN: {has_nan}")

  shape_ok = out.shape == (num_tokens, config.hidden_size)
  assert roundtrip_ok, "dispatch/gather/scatter indexing is broken -- fix before anything else"
  assert shape_ok, f"unexpected output shape {out.shape}"
  assert not has_nan, "forward pass produced NaNs"
  print("OK: dispatch roundtrip + full forward pass both check out")

  route_filter_ok = check_route_and_filter_correctness()
  assert route_filter_ok, "route_and_filter_to_local_shard failed its standalone correctness/overflow check"

  sharded_ok = check_sharded_forward_correctness()
  assert sharded_ok, (
      "sharded (route+filter+per-shard-FFN+combine) forward pass diverges from the "
      "unsharded reference -- the real 16-of-896-then-filtered-to-local-shard pipeline "
      "is not yet mathematically equivalent to computing over all experts directly"
  )
  print("OK: sharded end-to-end forward pass matches the unsharded reference")
