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

Usage:
  python kimi_k3_latent_moe_reference.py
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp


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
