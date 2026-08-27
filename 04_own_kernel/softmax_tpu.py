"""Task B: V-dimension tiling + online softmax, actually fixing the VMEM
problem for large vocabularies.

The previous version (now replaced) only chunked the batch (B) dimension,
moving the whole vocab (V) row into VMEM -- with `V=32000`, no matter how
small the batch block was shrunk, it still hit `RESOURCE_EXHAUSTED` (confirmed
by re-testing on real v6e hardware, see the correction in
`01_pallas_basics/notes.md`). This version also chunks V, completely
eliminating the need for a full row to stay resident in VMEM.

## Why "chunk and accumulate" doesn't just work naively

The softmax formula: `softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))`.
Computing `max(x)` and `sum(...)` in theory requires seeing the whole row
first. But if V is split into several chunks, there's no way to get the whole
row's max/sum in one pass -- this is *not* the same pattern as "K-dimension
chunked accumulation" in matmul (`03_matmul_k_tiled.py`): matmul's
accumulation is associative (`sum_k(x@y)` -- compute any chunk first, sum them
up at the end, still correct), but softmax's denominator `sum(exp(...))`
depends on a global max that isn't known until the whole row has been scanned
-- it can't be chunked and accumulated directly.

## Online softmax: correcting as you scan

The solution is to maintain a running max / running sum "online," updating
with this formula as each new V chunk arrives (this is exactly the same
technique used by Flash Attention, and by the real TPU kernel in
`03_case_study_cross_entropy`):

```
new_max = max(running_max, current chunk's max)
running_sum = running_sum * exp(running_max - new_max) + current chunk's sum(exp(x - new_max))
running_max = new_max
```

After scanning all V chunks, `running_max`/`running_sum` are the row's true
max/sum, and throughout the whole process only one small `(b_block, v_block)`
chunk needs to be resident in VMEM at a time.

## Two kernel passes (unlike matmul's single-pass accumulation)

Once the global max/sum are known, each element's softmax value
`exp(x_i - global_max) / global_sum` depends only on the block it's in -- it
doesn't require "going back and modifying" output blocks already computed
(this differs from Flash Attention, where "the output has to be re-weighted
as max gets updated," because here there's no further weighted sum over V,
just elementwise normalization). So the implementation is split into two
passes, two `pl.pallas_call`s:

  1. `_stats_kernel`: grid=(b_tiles, v_tiles), V is the inner ("arbitrary")
     dimension, updates running max/sum online, writes out two small arrays
     `row_max`, `row_sum` (shape `(B, 1)`).
  2. `_normalize_kernel`: grid=(b_tiles, v_tiles), reads back `row_max`/
     `row_sum` (for the same b block, all v blocks share the same value,
     `index_map` doesn't depend on v), computes
     `exp(x - row_max) / row_sum` per block, writes the output directly, no
     accumulator needed, both dimensions can be marked `"parallel"`.

To run:
  python softmax_tpu.py
  python softmax_tpu_test.py
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu


def _stats_kernel(x_ref, max_ref, sum_ref, running_max_ref, running_sum_ref, *, num_v_tiles: int):
  v_id = pl.program_id(1)

  @pl.when(v_id == 0)
  def _():
    running_max_ref[...] = jnp.zeros_like(running_max_ref) - jnp.inf
    running_sum_ref[...] = jnp.zeros_like(running_sum_ref)

  x = x_ref[...]
  old_max = running_max_ref[...]
  old_sum = running_sum_ref[...]

  block_max = jnp.max(x, axis=-1, keepdims=True)
  new_max = jnp.maximum(old_max, block_max)
  # When old_max == -inf, exp(old_max - new_max) == exp(-inf) == 0, and old_sum
  # is also 0, so 0 * 0 == 0 -- no NaN results.
  block_sum = jnp.sum(jnp.exp(x - new_max), axis=-1, keepdims=True)
  new_sum = old_sum * jnp.exp(old_max - new_max) + block_sum

  running_max_ref[...] = new_max
  running_sum_ref[...] = new_sum

  @pl.when(v_id == num_v_tiles - 1)
  def _():
    max_ref[...] = running_max_ref[...]
    sum_ref[...] = running_sum_ref[...]


def _normalize_kernel(x_ref, max_ref, sum_ref, o_ref):
  x = x_ref[...]
  o_ref[...] = (jnp.exp(x - max_ref[...]) / sum_ref[...]).astype(o_ref.dtype)


def softmax(
    x: jax.Array, *, b_block_size: int = 128, v_block_size: int = 1024
) -> jax.Array:
  """Row-wise softmax, x: (B, V), normalized along the last dimension. Both B and V are chunked."""
  b, v = x.shape
  assert b % b_block_size == 0, "keep the exercise simple: only support B divisible by the block size for now"
  assert v % v_block_size == 0, "keep the exercise simple: only support V divisible by the block size for now"

  num_v_tiles = v // v_block_size
  has_tpu = any(d.platform == "tpu" for d in jax.devices())
  interpret = not has_tpu

  # First pass: scan through all V chunks online, computing each row's true max, sum.
  row_max, row_sum = pl.pallas_call(
      functools.partial(_stats_kernel, num_v_tiles=num_v_tiles),
      grid=(b // b_block_size, num_v_tiles),
      in_specs=[pl.BlockSpec((b_block_size, v_block_size), lambda i, j: (i, j))],
      out_specs=[
          pl.BlockSpec((b_block_size, 1), lambda i, j: (i, 0)),
          pl.BlockSpec((b_block_size, 1), lambda i, j: (i, 0)),
      ],
      out_shape=[
          jax.ShapeDtypeStruct((b, 1), jnp.float32),
          jax.ShapeDtypeStruct((b, 1), jnp.float32),
      ],
      scratch_shapes=[
          pltpu.VMEM((b_block_size, 1), jnp.float32),
          pltpu.VMEM((b_block_size, 1), jnp.float32),
      ],
      compiler_params=(
          pltpu.CompilerParams(dimension_semantics=("parallel", "arbitrary"))
          if has_tpu
          else None
      ),
      interpret=interpret,
  )(x)

  # Second pass: given the (B, 1) max/sum, normalize x block-by-block into the real softmax output.
  return pl.pallas_call(
      _normalize_kernel,
      grid=(b // b_block_size, num_v_tiles),
      in_specs=[
          pl.BlockSpec((b_block_size, v_block_size), lambda i, j: (i, j)),
          pl.BlockSpec((b_block_size, 1), lambda i, j: (i, 0)),
          pl.BlockSpec((b_block_size, 1), lambda i, j: (i, 0)),
      ],
      out_specs=pl.BlockSpec((b_block_size, v_block_size), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((b, v), x.dtype),
      compiler_params=(
          pltpu.CompilerParams(dimension_semantics=("parallel", "parallel"))
          if has_tpu
          else None
      ),
      interpret=interpret,
  )(x, row_max, row_sum)


if __name__ == "__main__":
  key = jax.random.key(0)
  x = jax.random.normal(key, (512, 1024), dtype=jnp.float32) * 10.0  # scale up values to test stability

  out = softmax(x, b_block_size=128, v_block_size=256)
  expected = jax.nn.softmax(x, axis=-1)

  max_err = jnp.max(jnp.abs(out - expected))
  print("devices:", jax.devices())
  print("max abs error vs. jax.nn.softmax:", max_err)
  assert max_err < 1e-5, "softmax kernel result deviates too much"
  print("OK: pallas softmax (V-dimension tiling + online softmax) matches jax.nn.softmax")
