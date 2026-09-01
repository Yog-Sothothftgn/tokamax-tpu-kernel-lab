"""WP4 (SparseCore feasibility): profiling harness isolating the cost of
Kimi K3 LatentMoE's irregular indexing steps from the regular compute steps
around them, split into the four stages actually relevant to the SparseCore
decision:

  A. Router + projection (regular, matmul-heavy): router matmul, sigmoid,
     top-k selection, latent down-projection.
  B. Dispatch indexing (irregular -- the SparseCore candidate): reshape
     expert ids, filter to a local shard, argsort, generate token index,
     gather, bincount into group_sizes, capacity padding.
  C. Expert compute (regular, matmul-heavy): the per-expert FFN. On this
     machine, a dense-matmul stand-in (see below); on a TPU VM, the REAL
     `tokamax.ragged_dot` -- see `kimi_k3_latent_moe_ragged_dot.py`'s
     `profile_four_stages_wp4`.
  D. Combine (irregular -- also a SparseCore candidate): scatter each
     shard's contribution back to its token, weighted by the router's
     combine weight.

This is a refinement of an earlier, cruder 2-stage version of this file
(dispatch=A+B lumped together, combine=D) -- lumping A into "dispatch"
diluted the irregular-share metric WP4 actually needs with regular compute
that has nothing to do with the SparseCore question. A/B/D below reuse the
SAME functions `kimi_k3_latent_moe_ragged_dot.py`'s `profile_four_stages_wp4`
uses on real hardware (`router_and_projection`, `filter_and_pad_to_shard`,
`_combine_shard_contribution`, all in `kimi_k3_latent_moe_reference.py`) --
only Stage C differs between the two files (dense stand-in here vs. real
ragged_dot there), so A/B/D numbers from this CPU run and a future TPU run
are directly comparable stage-by-stage.

**What this file does NOT measure, on purpose:**
1. The real `tokamax.ragged_dot` matmul cost (Stage C) -- this machine has
   no working tokamax install (Windows long-path issue, see project
   memory), so Stage C here is a single dense matmul over the shard's
   dispatched rows with ONE shared weight matrix (ignores group_sizes/
   per-expert boundaries entirely -- NOT numerically meaningful, only a
   rough magnitude stand-in). The real Stage C cost needs a TPU VM.
2. HBM bytes moved / VMEM traffic -- CPU timing has no MXU/VMEM to reflect,
   so none of this says anything about TPU memory-traffic patterns (the
   actual quantity WP4's research question cares about). This is a
   dispatch-vs-compute TIME ratio methodology check on CPU, not a TPU
   memory-profiling result.

No tokamax dependency -- runs anywhere `kimi_k3_latent_moe_reference.py`
does. Follows the same timing discipline as
`01_pallas_basics/03_matmul_k_tiled.py`'s `check()`: one untimed warmup call
(excludes compile time), then `num_repeats` timed calls, `block_until_ready()`
on each. Stage B is timed eagerly (not jitted) since it's genuinely not
jit-compatible (filtering to a shard produces a data-dependent length, same
constraint as `filter_and_pad_to_shard`/`generate_local_shard_workload`
elsewhere in this project) -- this is the real usage pattern, not a
measurement artifact.

Usage:
  python profile_dispatch_vs_compute.py
"""

import csv as _csv
import functools
import pathlib
import time

import jax
import jax.numpy as jnp

from kimi_k3_latent_moe_reference import (
    LatentMoEConfig,
    _combine_shard_contribution,
    _situ_and_mul,
    filter_and_pad_to_shard,
    kimi_k3_config,
    router_and_projection,
)


def _dense_matmul_stand_in(
    sorted_tokens: jax.Array,
    dense_gate: jax.Array,
    dense_up: jax.Array,
    dense_down: jax.Array,
    config: LatentMoEConfig,
) -> jax.Array:
  """Stage C stand-in: a single, unsharded dense matmul over all of a
  shard's dispatched rows at once, using ONE shared weight matrix --
  ignores group_sizes/per-expert boundaries entirely. NOT numerically
  meaningful as a replacement for the real per-expert FFN; purely a rough
  magnitude stand-in for "how expensive is a matmul this size," since the
  real per-expert loop can't be jitted and `tokamax.ragged_dot` isn't
  available on this machine. See this module's docstring for what the real
  comparison needs instead.
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


def _time_eager_fn(f, *args, num_repeats: int = 20) -> float:
  """Times `f` called eagerly (not jitted) -- for Stage B, which is
  genuinely not jit-compatible (data-dependent output length from filtering
  to a shard), so eager timing IS the real usage pattern here, not a
  simplification. One untimed warmup call first (JAX still compiles each
  internal op eagerly the first time it sees a given shape, so this still
  excludes most of that one-time cost)."""
  out = f(*args)
  jax.block_until_ready(out)

  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f(*args)
  jax.block_until_ready(out)
  elapsed_ms = (time.perf_counter() - t0) * 1000 / num_repeats
  return elapsed_ms


def profile_four_stages_cpu(
    config: LatentMoEConfig | None = None,
    num_tokens: int = 512,
    local_num_experts: int = 64,
    capacity_factor: float = 2.0,
    seed: int = 0,
    num_repeats: int = 20,
    output_dir: pathlib.Path | None = None,
) -> dict:
  """Times Stage A (router+projection), Stage B (dispatch indexing to one
  local shard), and Stage D (combine) at real Kimi K3 per-expert dims
  (default `kimi_k3_config()`), plus a Stage C dense-matmul stand-in (see
  module docstring for why it's not the real `tokamax.ragged_dot` cost).

  **CPU numbers here are a methodology check, not a TPU finding** -- CPU has
  no MXU/VMEM, so the irregular-vs-regular RATIO on CPU may not resemble the
  ratio on real TPU hardware at all. Re-run
  `kimi_k3_latent_moe_ragged_dot.py`'s `profile_four_stages_wp4` (same
  Stage A/B/D functions, real Stage C) on a TPU VM before treating any
  ratio here as evidence for or against a SparseCore prototype.

  **Second caveat, flagged by a reviewer (2026-09-02), applies here too**:
  Stage B is timed via `_time_eager_fn` (genuinely not jit-compatible, see
  that helper's docstring), while A/C/D use `_time_jit_fn` -- so Stage B's
  number includes Python dispatch/host-device sync overhead that A/C/D's
  compiled-loop timing does not. `irregular_share` below is therefore an
  "eager pipeline latency share," not a clean device-only comparison, on
  CPU same as on TPU -- see `profile_four_stages_wp4`'s matching docstring
  in `kimi_k3_latent_moe_ragged_dot.py` for the two ways to fix this
  properly (neither attempted yet).
  """
  config = config or kimi_k3_config()
  key = jax.random.key(seed)
  keys = jax.random.split(key, 7)
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

  # config (a plain dataclass, not a registered pytree) is bound via
  # functools.partial rather than passed as a jit-traced call argument --
  # see _time_jit_fn's docstring for why passing it directly crashes.
  stage_a_fn = functools.partial(router_and_projection, config=config)
  stage_a_ms = _time_jit_fn(
      stage_a_fn, hidden_states, router_weight, e_score_correction_bias, down_proj,
      num_repeats=num_repeats,
  )
  topk_idx, topk_weight, x = jax.jit(stage_a_fn)(
      hidden_states, router_weight, e_score_correction_bias, down_proj
  )

  stage_b_fn = functools.partial(
      filter_and_pad_to_shard, config=config, local_expert_start=0,
      local_num_experts=local_num_experts, capacity_factor=capacity_factor,
  )
  stage_b_ms = _time_eager_fn(stage_b_fn, topk_idx, topk_weight, x, num_repeats=num_repeats)
  (
      sorted_tokens, group_sizes, valid_mask, per_expert_counts,
      padded_token_idx, padded_combine_weight,
  ) = stage_b_fn(topk_idx, topk_weight, x)
  print(
      f"[wp4-profile] Stage B output: M_padded={sorted_tokens.shape[0]} "
      f"valid_rows={int(jnp.sum(valid_mask))} "
      f"mean_per_expert={float(jnp.mean(per_expert_counts)):.2f}"
  )

  stage_c_fn = functools.partial(_dense_matmul_stand_in, config=config)
  stage_c_ms = _time_jit_fn(
      stage_c_fn, sorted_tokens, dense_gate, dense_up, dense_down,
      num_repeats=num_repeats,
  )
  shard_out = jax.jit(stage_c_fn)(sorted_tokens, dense_gate, dense_up, dense_down)

  routed_out_init = jnp.zeros((num_tokens, config.latent_size), dtype=jnp.float32)
  stage_d_ms = _time_jit_fn(
      _combine_shard_contribution, routed_out_init, shard_out, padded_token_idx, padded_combine_weight,
      num_repeats=num_repeats,
  )

  total_ms = stage_a_ms + stage_b_ms + stage_c_ms + stage_d_ms
  irregular_ms = stage_b_ms + stage_d_ms  # B and D are the SparseCore-relevant steps; A and C are regular compute
  irregular_share = irregular_ms / total_ms

  result = {
      "num_tokens": num_tokens,
      "stage_a_router_projection_ms": stage_a_ms,
      "stage_b_dispatch_indexing_ms": stage_b_ms,
      "stage_c_matmul_stand_in_ms": stage_c_ms,
      "stage_d_combine_ms": stage_d_ms,
      "irregular_share_of_total": irregular_share,
  }
  print(
      f"[wp4-profile] num_tokens={num_tokens} "
      f"A(router+proj)={stage_a_ms:.3f}ms B(dispatch-idx, EAGER-timed)={stage_b_ms:.3f}ms "
      f"C(matmul-stand-in)={stage_c_ms:.3f}ms D(combine)={stage_d_ms:.3f}ms "
      f"irregular_share(B+D)={irregular_share:.1%}"
  )

  if output_dir is not None:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "wp4_profiling_cpu.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="", encoding="utf-8") as f:
      writer = _csv.DictWriter(f, fieldnames=list(result.keys()))
      if write_header:
        writer.writeheader()
      writer.writerow(result)
    print(f"  (structured data appended to {csv_path})")
  return result


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-dir", type=pathlib.Path, default=None,
      help="if given, also append structured results to wp4_profiling_cpu.csv there",
  )
  args = parser.parse_args()

  print(f"devices: {jax.devices()}")
  print(
      "NOTE: this run is on whatever device is available locally (CPU on this "
      "machine) -- see profile_four_stages_cpu's docstring for why the "
      "resulting ratio is a methodology check, not a TPU finding. Stage C is a "
      "dense-matmul stand-in, not the real tokamax.ragged_dot cost.\n"
  )
  for num_tokens in (128, 512, 2048):
    profile_four_stages_cpu(num_tokens=num_tokens, output_dir=args.output_dir)
