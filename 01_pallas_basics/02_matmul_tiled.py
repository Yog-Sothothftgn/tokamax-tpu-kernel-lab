"""Exercise 2: tiled matmul -- introducing grid / BlockSpec / index_map.

Concepts to learn:
  - Real TPU inputs are usually far larger than VMEM (typically a few MB to
    ~10+ MB), so they can't be moved in as one whole block like exercise 1 --
    the computation needs to be chunked into tiles, moved in and computed one
    tile at a time. This is exactly the problem `grid` + `BlockSpec` solve.
  - `grid=(m_tiles, n_tiles)`: the kernel function gets "logically" invoked
    m_tiles * n_tiles times, and each invocation knows its own position in the
    grid via `pl.program_id(axis)`.
  - `BlockSpec(block_shape, index_map)`: tells pallas which "block" of the
    input/output to move into VMEM on each invocation. `index_map(i, j) ->
    (block_row, block_col)` returns *block coordinates* (which block), not
    element coordinates -- pallas automatically multiplies by block_shape.
  - To keep grid/BlockSpec clear for now, this exercise doesn't chunk the K
    dimension (the one matmul reduces/accumulates over) -- the full K length
    is moved in every time, for both the row and column. Once you're
    comfortable with this step, try: also chunk K, making the grid
    (m_tiles, n_tiles, k_tiles) -- now multiple invocations write to the same
    output block, requiring a `pltpu.VMEM` scratch accumulator +
    `pl.when(k_id == 0)` for initialization and
    `pl.when(k_id == nsteps - 1)` to write back -- almost every production
    kernel in tokamax (see 03_case_study_cross_entropy) uses this "3D grid +
    accumulator" pattern.

To run:
  python 02_matmul_tiled.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def matmul_kernel(x_ref, y_ref, o_ref):
  # Here x_ref is a (bm, K) block, y_ref is a (K, bn) block (K isn't chunked),
  # o_ref is the (bm, bn) output block. Each grid position is invoked exactly
  # once, so we can assign directly -- no accumulation needed.
  o_ref[...] = jnp.dot(
      x_ref[...], y_ref[...], preferred_element_type=jnp.float32
  )


def matmul(x: jax.Array, y: jax.Array, *, bm: int = 128, bn: int = 128) -> jax.Array:
  m, k = x.shape
  k2, n = y.shape
  assert k == k2, f"inner dims don't match: {x.shape} @ {y.shape}"
  assert m % bm == 0 and n % bn == 0, "keep the exercise simple: only support shapes divisible by the tile size for now"

  has_tpu = any(d.platform == "tpu" for d in jax.devices())

  return pl.pallas_call(
      matmul_kernel,
      grid=(m // bm, n // bn),
      in_specs=[
          # Block (i, j) of x: take row-block i, need the full K dimension.
          pl.BlockSpec((bm, k), lambda i, j: (i, 0)),
          # Block (i, j) of y: need the full K dimension, take column-block j.
          pl.BlockSpec((k, bn), lambda i, j: (0, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), jnp.float32),
      interpret=not has_tpu,
  )(x, y)


if __name__ == "__main__":
  key = jax.random.key(0)
  kx, ky = jax.random.split(key)
  m, k, n = 512, 256, 384
  x = jax.random.normal(kx, (m, k), dtype=jnp.float32)
  y = jax.random.normal(ky, (k, n), dtype=jnp.float32)

  out = matmul(x, y, bm=128, bn=128)
  expected = x @ y

  max_err = jnp.max(jnp.abs(out - expected))
  print("devices:", jax.devices())
  print("max abs error vs. x @ y:", max_err)
  # Tiled accumulation order isn't exactly the same as XLA's matmul; a little float32 error is expected.
  assert max_err < 1e-2, "tiled matmul result deviates too much, check BlockSpec/index_map"
  print("OK: pallas matmul matches x @ y (within floating-point tolerance)")
