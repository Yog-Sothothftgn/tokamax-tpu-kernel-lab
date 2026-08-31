"""WP4 (SparseCore feasibility): profiling harness isolating the cost of
Kimi K3 LatentMoE's irregular indexing steps (dispatch: router + argsort +
gather; combine: scatter + weighted-sum) from the regular dense per-expert
matmul cost, within one forward pass.

This feeds WP4's decision criteria directly (see project research plan):
only build a SparseCore prototype if profiling actually shows the irregular
part is a real bottleneck (time or memory-traffic sense) -- a hypothesis
("SparseCore for irregular indexing, MXU for dense matmul") is not itself
evidence. WP-Pre already confirmed SparseCore's pallas API exists; this
file is the first attempt at the actual measurement WP4 needs before
deciding whether to build anything.

**Two things this file does NOT measure, on purpose:**
1. The real `tokamax.ragged_dot` matmul cost -- this machine has no working
   tokamax install (Windows long-path issue, see project memory), so the
   "regular compute" side below uses a single dense matmul over all
   dispatched rows with ONE shared weight matrix as a rough magnitude
   stand-in (ignores group_sizes/per-expert boundaries entirely -- NOT
   numerically meaningful, only useful for a "how expensive is a matmul
   this size" timing comparison). The real comparison against
   `tokamax.ragged_dot`'s actual cost needs a TPU VM -- see
   `kimi_k3_latent_moe_ragged_dot.py` for a tokamax-dependent follow-up if
   this rough version's ratio looks close enough to be worth pinning down
   precisely.
2. HBM bytes moved / VMEM traffic -- CPU timing has no MXU/VMEM to reflect,
   so these numbers say nothing about TPU memory-traffic patterns (the
   actual quantity WP4's research question cares about, per the "memory
   performance first" framing in the research plan). This is a
   dispatch-vs-compute TIME ratio sanity check on CPU, not a TPU memory
   profiling result -- do not cite it as a memory-traffic finding.

No tokamax dependency -- runs anywhere `kimi_k3_latent_moe_reference.py`
does. Follows the same timing discipline as
`01_pallas_basics/03_matmul_k_tiled.py`'s `check()`: one untimed warmup call
(excludes compile time), then `num_repeats` timed calls, `block_until_ready()`
on each.

Usage:
  python profile_dispatch_vs_compute.py
"""

import functools
import time

import jax
import jax.numpy as jnp

from kimi_k3_latent_moe_reference import (
    LatentMoEConfig,
    _situ_and_mul,
    kimi_k3_config,
)


def _dispatch_only(
    hidden_states: jax.Array,
    router_weight: jax.Array,
    e_score_correction_bias: jax.Array,
    down_proj: jax.Array,
    config: LatentMoEConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Steps 1-3 of `latent_moe_forward`: router + shared down-projection +
  dispatch (argsort + gather). Unlike `latent_moe_forward`'s per-expert
  loop, this has no dynamic Python-int slicing, so it IS jit-compatible --
  needed to get a real device-timed measurement rather than eager
  Python-dispatch overhead.
  """
  logits = hidden_states.astype(jnp.float32) @ router_weight.astype(jnp.float32)
  scores = jax.nn.sigmoid(logits)
  scores_for_choice = scores + e_score_correction_bias.astype(jnp.float32)[None, :]
  _, topk_idx = jax.lax.top_k(scores_for_choice, config.top_k)
  topk_weight = jnp.take_along_axis(scores, topk_idx, axis=-1)
  if config.top_k > 1 and config.moe_renormalize:
    denom = jnp.sum(topk_weight, axis=-1, keepdims=True) + 1e-20
    topk_weight = topk_weight / denom
  topk_weight = topk_weight * config.routed_scaling_factor

  x = hidden_states @ down_proj
  num_tokens = hidden_states.shape[0]
  flat_expert_ids = topk_idx.reshape(-1)
  order = jnp.argsort(flat_expert_ids)
  token_of_slot = jnp.arange(num_tokens * config.top_k) // config.top_k
  sorted_token_idx = token_of_slot[order]
  sorted_tokens = x[sorted_token_idx]
  group_sizes = jnp.bincount(flat_expert_ids, length=config.num_experts)
  return sorted_tokens, group_sizes, order, topk_weight


def _combine_only(
    outs: jax.Array,
    order: jax.Array,
    topk_weight: jax.Array,
    num_tokens: int,
    config: LatentMoEConfig,
) -> jax.Array:
  """Step 5 of `latent_moe_forward`: scatter back to (token, slot) order,
  then weighted-sum over the top_k slots per token. Jit-compatible (fixed
  output shape, no dynamic slicing)."""
  unsorted = jnp.zeros_like(outs).at[order].set(outs)
  unsorted = unsorted.reshape(num_tokens, config.top_k, config.latent_size)
  routed_out = jnp.sum(unsorted * topk_weight[..., None], axis=1)
  return routed_out.astype(outs.dtype)


def _dense_matmul_stand_in(
    sorted_tokens: jax.Array,
    dense_gate: jax.Array,
    dense_up: jax.Array,
    dense_down: jax.Array,
    config: LatentMoEConfig,
) -> jax.Array:
  """A single, unsharded dense matmul over ALL dispatched rows at once,
  using ONE shared weight matrix -- ignores group_sizes/per-expert
  boundaries entirely. NOT numerically meaningful as a replacement for the
  real per-expert FFN; purely a rough magnitude stand-in for "how expensive
  is a matmul this size," since the real per-expert loop can't be jitted
  (needs concrete Python ints for its dynamic slice sizes) and
  `tokamax.ragged_dot` isn't available on this machine. See this module's
  docstring for what a real `tokamax.ragged_dot`-based comparison would
  need instead.
  """
  gate = sorted_tokens @ dense_gate
  up = sorted_tokens @ dense_up
  activated = _situ_and_mul(gate, up, config.activation_situ_beta, config.activation_situ_linear_beta)
  return activated @ dense_down


def _time_jit_fn(f, *args, num_repeats: int = 20) -> float:
  """`f` must already have any non-array arguments (e.g. `config`, a plain
  dataclass -- not a registered pytree, so jax.jit can't trace it directly)
  bound via `functools.partial` before being passed in here -- jax.jit only
  ever traces what's actually passed at call time, so a pre-bound kwarg on a
  partial is invisible to it (confirmed the hard way: passing `config`
  straight through as a positional arg raised "Error interpreting argument
  ... as an abstract array" for the `LatentMoEConfig` value)."""
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)  # untimed warmup -- excludes compile time from the measurement

  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
  jax.block_until_ready(out)
  elapsed_ms = (time.perf_counter() - t0) * 1000 / num_repeats
  return elapsed_ms


def profile_dispatch_combine_vs_matmul(
    config: LatentMoEConfig | None = None,
    num_tokens: int = 512,
    seed: int = 0,
    num_repeats: int = 20,
) -> dict:
  """Times dispatch, combine, and a rough matmul stand-in separately, at
  real Kimi K3 per-expert dims (default `kimi_k3_config()`) unless a
  smaller config is passed in for faster local iteration. Returns a dict of
  millisecond timings plus the dispatch+combine share of the total -- WP4's
  actual decision input, not a hypothesis.

  **CPU numbers here are a methodology check, not a TPU finding** -- CPU has
  no MXU/VMEM, so the dispatch-vs-matmul RATIO on CPU may not resemble the
  ratio on real TPU hardware at all (e.g. TPU's MXU is dramatically faster
  at dense matmul than CPU is, which would make dispatch/combine a LARGER
  relative share of total time on TPU than what's measured here -- or
  smaller, if TPU's gather/scatter path is also much faster; there's no way
  to know without actually measuring on the target hardware). Re-run the
  same methodology on a TPU VM before treating any ratio here as evidence
  for or against a SparseCore prototype.
  """
  config = config or kimi_k3_config()
  key = jax.random.key(seed)
  keys = jax.random.split(key, 6)
  scale = 0.02

  def normal(k, shape):
    return (jax.random.normal(k, shape) * scale).astype(jnp.float32)

  router_weight = normal(keys[0], (config.hidden_size, config.num_experts))
  e_score_correction_bias = jnp.zeros((config.num_experts,), dtype=jnp.float32)
  down_proj = normal(keys[1], (config.hidden_size, config.latent_size))
  dense_gate = normal(keys[2], (config.latent_size, config.intermediate_size))
  dense_up = normal(keys[3], (config.latent_size, config.intermediate_size))
  dense_down = normal(keys[4], (config.intermediate_size, config.latent_size))
  hidden_states = jax.random.normal(keys[5], (num_tokens, config.hidden_size), dtype=jnp.float32)

  # config (a plain dataclass, not a registered pytree) and num_tokens (used
  # as a static .reshape() target shape, not an array value) both get bound
  # via functools.partial rather than passed as jit-traced call arguments --
  # see _time_jit_fn's docstring for why passing them directly crashes.
  dispatch_fn = functools.partial(_dispatch_only, config=config)
  dispatch_ms = _time_jit_fn(
      dispatch_fn, hidden_states, router_weight, e_score_correction_bias, down_proj,
      num_repeats=num_repeats,
  )

  sorted_tokens, _group_sizes, order, topk_weight = jax.jit(dispatch_fn)(
      hidden_states, router_weight, e_score_correction_bias, down_proj
  )

  matmul_fn = functools.partial(_dense_matmul_stand_in, config=config)
  matmul_ms = _time_jit_fn(
      matmul_fn, sorted_tokens, dense_gate, dense_up, dense_down,
      num_repeats=num_repeats,
  )

  outs = jax.jit(matmul_fn)(sorted_tokens, dense_gate, dense_up, dense_down)

  combine_fn = functools.partial(_combine_only, num_tokens=num_tokens, config=config)
  combine_ms = _time_jit_fn(
      combine_fn, outs, order, topk_weight,
      num_repeats=num_repeats,
  )

  total_ms = dispatch_ms + matmul_ms + combine_ms
  irregular_ms = dispatch_ms + combine_ms
  irregular_share = irregular_ms / total_ms

  result = {
      "num_tokens": num_tokens,
      "dispatch_ms": dispatch_ms,
      "combine_ms": combine_ms,
      "matmul_stand_in_ms": matmul_ms,
      "irregular_share_of_total": irregular_share,
  }
  print(
      f"[wp4-profile] num_tokens={num_tokens} dispatch={dispatch_ms:.3f}ms "
      f"combine={combine_ms:.3f}ms matmul_stand_in={matmul_ms:.3f}ms "
      f"irregular_share={irregular_share:.1%}"
  )
  return result


if __name__ == "__main__":
  print(f"devices: {jax.devices()}")
  print(
      "NOTE: this run is on whatever device is available locally (CPU on this "
      "machine) -- see profile_dispatch_combine_vs_matmul's docstring for why "
      "the resulting ratio is a methodology check, not a TPU finding.\n"
  )
  for num_tokens in (128, 512, 2048):
    profile_dispatch_combine_vs_matmul(num_tokens=num_tokens)
