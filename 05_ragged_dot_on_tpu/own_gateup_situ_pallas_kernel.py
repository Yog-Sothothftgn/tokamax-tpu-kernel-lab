"""A from-scratch, hand-written Pallas TPU kernel for Kimi K3's gate/up
projection + SiTU-GLU activation -- written to actually satisfy Zifan's
"push to use raw Pallas and do kernel fusion" request, after the previous
attempt (`_local_shard_expert_ffn_ragged_dot_fused_gateup` in
`kimi_k3_latent_moe_ragged_dot.py`) turned out to just be a correct CALL
into tokamax's own already-built fusion feature -- zero Pallas code
written by this project, same category of shortcut Zifan originally
flagged ("you're just importing ops from tokamax").

**Deliberately scoped down from the real MoE case, per explicit user
direction (2026-09-20)**: ONE expert (dense, no ragged/grouped dispatch),
real Kimi K3 per-expert matrix dimensions (`latent_size=3584`,
`intermediate_size=3072`, confirmed from `kimi_k3_config()` in
`kimi_k3_latent_moe_reference.py`):

  X:              (M, 3584)
  W_gate, W_up:   (3584, 3072)
  Y (output):     (M, 3072)

This is the K-tiled matmul template this project already built and
verified in `01_pallas_basics/03_matmul_k_tiled.py` (Task A) -- extended
from ONE rhs matrix to TWO (gate and up, sharing the same lhs and the same
K-tiling schedule), with a fused activation applied to the two
accumulators on the LAST K step, instead of writing each matmul's result
back to HBM separately.

Kernel structure (grid = (m_tiles, n_tiles, k_tiles), K innermost/
"arbitrary" since the same (i,j) output block accumulates across K):
  1. Output tiling: each grid step (i, j, k) owns output block (i, j);
     `out_specs`' index_map ignores `k` (same convention as the K-tiled
     matmul template) -- the block is only actually WRITTEN on the last k.
  2. `in_specs` walk `x` along (i, k) and `w_gate`/`w_up` along (k, j) --
     the K-dimension reduction is loaded incrementally, tile by tile, not
     materialized as one whole (M, 3584) / (3584, 3072) block in VMEM.
  3. TWO VMEM scratch accumulators (`gate_acc_ref`, `up_acc_ref`), zeroed
     at `k_id == 0`, each accumulating its own partial matmul independently
     every K step.
  4. On the last K step: round each accumulator to `round_dtype` (bf16)
     first, THEN upcast to float32 to compute SiTU-GLU -- reproducing this
     project's established `_situ_and_mul` convention (see that function's
     docstring in `kimi_k3_latent_moe_reference.py`), not the raw fp32
     accumulator directly, so this kernel's numerics are directly
     comparable to the rest of this project's Kimi K3 pipeline.
  5. Only the ACTIVATED result is ever written to `o_ref` -- `gate_acc_ref`/
     `up_acc_ref` never touch HBM at all, at any K step.

Verified via `pl.pallas_call(..., interpret=True)` -- genuine Pallas
kernel semantics (grid iteration, BlockSpec indexing, scratch persistence
across the K loop, `pl.when` boundary conditions), not just a plain-JAX
formula check, runnable on this CPU-only Windows machine with NO real TPU
-- same technique `03_matmul_k_tiled.py`'s own `check()` already
established for this project. Real Mosaic-compiled TPU behavior (VMEM
budget at these real, much larger-than-`03_matmul_k_tiled.py`'s-toy-sizes
dimensions, actual tile-size tuning) still needs the real v6e TPU VM --
not yet run there as of this file's creation.

To run (local, interpret mode, no TPU needed):
  python own_gateup_situ_pallas_kernel.py
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Real Kimi K3 per-expert dimensions (kimi_k3_config()) and SiTU-GLU
# constants (config.json's activation_situ_beta/activation_situ_linear_beta).
LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def fused_gateup_situ_kernel(
    x_ref,
    wgate_ref,
    wup_ref,
    o_ref,
    gate_acc_ref,
    up_acc_ref,
    *,
    num_k_tiles: int,
    beta: float,
    linear_beta: float,
    round_dtype: jnp.dtype,
):
  k_id = pl.program_id(2)

  @pl.when(k_id == 0)
  def _():
    gate_acc_ref[...] = jnp.zeros_like(gate_acc_ref)
    up_acc_ref[...] = jnp.zeros_like(up_acc_ref)

  gate_acc_ref[...] += jnp.dot(
      x_ref[...], wgate_ref[...], preferred_element_type=jnp.float32
  )
  up_acc_ref[...] += jnp.dot(
      x_ref[...], wup_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(k_id == num_k_tiles - 1)
  def _():
    # Deliberately round to bf16 THEN upcast, matching this project's
    # established _situ_and_mul convention (see kimi_k3_latent_moe_reference.py)
    # -- not the raw fp32 accumulator directly.
    gate = gate_acc_ref[...].astype(round_dtype).astype(jnp.float32)
    up = up_acc_ref[...].astype(round_dtype).astype(jnp.float32)
    situ_gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    bounded_up = linear_beta * jnp.tanh(up / linear_beta)
    activated = situ_gate * bounded_up
    o_ref[...] = activated.astype(o_ref.dtype)


def fused_gateup_situ(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    *,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    round_dtype: jnp.dtype = jnp.bfloat16,
    bm: int = 128,
    bk: int = 512,
    bn: int = 512,
) -> jax.Array:
  """One expert's gate/up projection + SiTU-GLU, fused into a single
  hand-written Pallas kernel. `x`: (M, K). `w_gate`/`w_up`: (K, N). Returns
  (M, N) -- the ACTIVATED result only; gate/up never leave VMEM.
  """
  m, k = x.shape
  k2, n = w_gate.shape
  assert k == k2, f"x/w_gate inner dims don't match: {x.shape} @ {w_gate.shape}"
  assert w_up.shape == w_gate.shape, (
      f"w_gate/w_up shape mismatch: {w_gate.shape} vs {w_up.shape}"
  )
  assert m % bm == 0 and n % bn == 0 and k % bk == 0, (
      "keep the exercise simple: only support shape/tile combinations that "
      "divide evenly for now (matches 03_matmul_k_tiled.py's own convention)"
  )

  num_k_tiles = k // bk
  has_tpu = any(d.platform == "tpu" for d in jax.devices())

  return pl.pallas_call(
      functools.partial(
          fused_gateup_situ_kernel,
          num_k_tiles=num_k_tiles,
          beta=beta,
          linear_beta=linear_beta,
          round_dtype=round_dtype,
      ),
      grid=(m // bm, n // bn, num_k_tiles),
      in_specs=[
          # x: row-block i fixed, walks the k-th K-tile.
          pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
          # w_gate: walks the k-th K-tile, column-block j fixed.
          pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
          # w_up: same walk as w_gate, independent weight array.
          pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
      ],
      # Same (i, j) output block "visited" every K step, only WRITTEN on
      # the last one (see pl.when in the kernel) -- identical convention to
      # 03_matmul_k_tiled.py.
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      scratch_shapes=[
          pltpu.VMEM((bm, bn), jnp.float32),  # gate_acc
          pltpu.VMEM((bm, bn), jnp.float32),  # up_acc
      ],
      compiler_params=(
          pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary"),
          )
          if has_tpu
          else None
      ),
      interpret=not has_tpu,
  )(x, w_gate, w_up)


def _reference_gateup_situ(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    beta: float,
    linear_beta: float,
    round_dtype: jnp.dtype,
) -> jax.Array:
  """Plain-JAX reference: same bf16-round-then-upcast convention as the
  kernel, but via ordinary (non-Pallas) matmuls -- what
  `_situ_and_mul(x @ w_gate, x @ w_up, ...)` computes in
  `kimi_k3_latent_moe_reference.py`, reproduced standalone here so this
  file has zero dependency on that module (or tokamax) for its own
  correctness check.
  """
  gate = (x @ w_gate).astype(round_dtype).astype(jnp.float32)
  up = (x @ w_up).astype(round_dtype).astype(jnp.float32)
  situ_gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
  bounded_up = linear_beta * jnp.tanh(up / linear_beta)
  return (situ_gate * bounded_up).astype(x.dtype)


def check(m: int, bm: int, bk: int, bn: int, dtype=jnp.bfloat16) -> bool:
  """Correctness check, runnable in interpret mode with no TPU -- mirrors
  03_matmul_k_tiled.py's own check() structure. Uses real Kimi K3 per-expert
  dimensions (LATENT_SIZE=3584, INTERMEDIATE_SIZE=3072), only `m` (token
  count) and tile sizes vary across configs.
  """
  key = jax.random.key(hash((m, bm, bk, bn)) % (2**31))
  kx, kg, ku = jax.random.split(key, 3)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w_gate = (jax.random.normal(kg, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)
  w_up = (jax.random.normal(ku, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  fused_jit = jax.jit(functools.partial(fused_gateup_situ, bm=bm, bk=bk, bn=bn))
  out = fused_jit(x, w_gate, w_up)
  jax.block_until_ready(out)  # run once first, exclude compile overhead from timing

  t0 = time.perf_counter()
  out = fused_jit(x, w_gate, w_up)
  jax.block_until_ready(out)
  elapsed_ms = (time.perf_counter() - t0) * 1000

  expected = _reference_gateup_situ(
      x, w_gate, w_up, SITU_BETA, SITU_LINEAR_BETA, jnp.bfloat16
  )
  diff = jnp.abs(out.astype(jnp.float32) - expected.astype(jnp.float32))
  max_abs_diff = float(jnp.max(diff))
  out_scale = float(jnp.std(expected.astype(jnp.float32))) + 1e-8
  relative_max_diff = max_abs_diff / out_scale
  has_nan = bool(jnp.any(jnp.isnan(out)))
  has_inf = bool(jnp.any(jnp.isinf(out)))
  tolerance = 0.05
  ok = relative_max_diff < tolerance and not has_nan and not has_inf
  status = "OK" if ok else "FAIL"
  print(
      f"[{status}] m={m} k={LATENT_SIZE} n={INTERMEDIATE_SIZE}"
      f" tile=(bm={bm},bk={bk},bn={bn})"
      f" max_abs_diff={max_abs_diff:.4e} relative_max_diff={relative_max_diff:.4f}"
      f" has_nan={has_nan} has_inf={has_inf} time={elapsed_ms:.3f}ms"
  )
  return ok


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  print(
      "Note: interpret-mode timing (no real TPU here) only confirms the "
      "kernel runs correctly -- it says nothing about real TPU performance; "
      "real timing needs the v6e TPU VM."
  )
  configs = [
      # (m, bm, bk, bn) -- 3584 = 512*7 = 256*14 = 128*28; 3072 = 512*6 = 256*12 = 128*24
      (128, 128, 512, 512),
      (2048, 256, 512, 512),
      (2048, 128, 256, 256),
  ]
  results = [check(*c) for c in configs]
  assert all(results), (
      "some config's result is wrong -- go back and check the two-accumulator "
      "K-tiling logic / pl.when boundary conditions / bf16-round-then-upcast order"
  )
  print(f"all {len(results)} shape/tile configs passed")
