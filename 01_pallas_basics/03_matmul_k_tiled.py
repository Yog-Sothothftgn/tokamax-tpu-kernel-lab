"""Task A (the next priority task set by the 2026-08-12 handoff document):
K/reduction-dimension tiled matmul.

In `02_matmul_tiled.py`, the K dimension is moved into VMEM as one whole
chunk -- it doesn't actually solve the "K is large, doesn't fit in VMEM"
problem. This exercise expands the grid from `(m_tiles, n_tiles)` to
`(m_tiles, n_tiles, k_tiles)`, introducing two new concepts:

  - `scratch_shapes` + `pltpu.VMEM((bm, bn), dtype)`: requests a resident VMEM
    temporary buffer (the accumulator) that doesn't correspond to any
    input/output array -- the kernel function receives one extra
    corresponding Ref parameter (here, `acc_ref`), which persists across grid
    iterations rather than being re-fetched every time the way `in_specs`/
    `out_specs` buffers are.
  - `compiler_params=pltpu.CompilerParams(dimension_semantics=...)`: tells the
    compiler the "semantics" of each grid dimension. `"parallel"` dimensions
    have no dependencies between them and can be reordered/parallelized; the
    K dimension must be marked `"arbitrary"`, because the same (i, j) output
    block gets accumulated in order across multiple K-dimension iterations --
    out-of-order or parallel execution would produce wrong results.

Kernel logic (the same `pl.when`-accumulator template seen in
03_case_study_cross_entropy):
  - When `k_id == 0`, zero out the accumulator;
  - Every K tile does one partial matmul, accumulated into `acc_ref`;
  - When `k_id == num_k_tiles - 1` (the last K tile), write the accumulator
    back to the real output `o_ref` (outside this step, `o_ref` is never
    touched -- the intermediate K steps only read/write scratch, producing no
    partial writes to the output).

To run:
  python 03_matmul_k_tiled.py
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def matmul_kernel(x_ref, y_ref, o_ref, acc_ref, *, num_k_tiles: int):
  k_id = pl.program_id(2)

  @pl.when(k_id == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )

  @pl.when(k_id == num_k_tiles - 1)
  def _():
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def matmul(
    x: jax.Array, y: jax.Array, *, bm: int = 128, bk: int = 128, bn: int = 128
) -> jax.Array:
  m, k = x.shape
  k2, n = y.shape
  assert k == k2, f"inner dims don't match: {x.shape} @ {y.shape}"
  assert m % bm == 0 and n % bn == 0 and k % bk == 0, (
      "keep the exercise simple: only support shape/tile combinations that divide evenly for now"
  )

  num_k_tiles = k // bk
  has_tpu = any(d.platform == "tpu" for d in jax.devices())

  return pl.pallas_call(
      functools.partial(matmul_kernel, num_k_tiles=num_k_tiles),
      grid=(m // bm, n // bn, num_k_tiles),
      in_specs=[
          # Block (i, j, k) of x: row-block i is fixed, walks to the k-th K-tile along the K dimension.
          pl.BlockSpec((bm, bk), lambda i, j, k: (i, k)),
          # Block (i, j, k) of y: walks to the k-th K-tile along the K dimension, column-block j is fixed.
          pl.BlockSpec((bk, bn), lambda i, j, k: (k, j)),
      ],
      # The output block's index_map doesn't depend on k: the same (i, j)
      # output block gets "visited" repeatedly across multiple grid
      # iterations along K (but is only actually written on the last one, see
      # pl.when in the kernel).
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, k: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
      compiler_params=(
          pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary"),
          )
          if has_tpu
          else None
      ),
      interpret=not has_tpu,
  )(x, y)


def check(m: int, k: int, n: int, bm: int, bk: int, bn: int) -> bool:
  key = jax.random.key(hash((m, k, n, bm, bk, bn)) % (2**31))
  kx, ky = jax.random.split(key)
  x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
  y = jax.random.normal(ky, (k, n), dtype=jnp.float32)

  matmul_jit = jax.jit(functools.partial(matmul, bm=bm, bk=bk, bn=bn))
  out = matmul_jit(x, y).block_until_ready()  # run once first, to exclude compile overhead from timing

  t0 = time.perf_counter()
  out = matmul_jit(x, y).block_until_ready()
  elapsed_ms = (time.perf_counter() - t0) * 1000

  expected = x @ y
  max_err = float(jnp.max(jnp.abs(out - expected)))
  ok = max_err < 1e-2
  status = "OK" if ok else "FAIL"
  print(
      f"[{status}] shape=(m={m},k={k},n={n})"
      f" tile=(bm={bm},bk={bk},bn={bn})"
      f" max_err={max_err:.3e} time={elapsed_ms:.2f}ms"
  )
  return ok


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  # Note: timing under CPU/interpret mode is only to confirm the script runs,
  # it says nothing about real TPU performance; the timing that actually
  # matters only counts when this script runs on a TPU VM.
  configs = [
      # (m, k, n, bm, bk, bn) -- three different shape / tile combinations
      (512, 512, 512, 128, 128, 128),
      (512, 1024, 512, 256, 128, 256),
      (1024, 1024, 1024, 128, 256, 128),
  ]
  results = [check(*c) for c in configs]
  assert all(results), "some config's result is wrong, go back and check the accumulator / pl.when boundary conditions"
  print(f"all {len(results)} shape/tile configs passed, K-dimension tiling + VMEM accumulator logic is correct")
