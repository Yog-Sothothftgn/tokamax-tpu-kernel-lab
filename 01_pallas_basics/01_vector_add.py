"""Exercise 1: the simplest Pallas TPU kernel -- vector addition.

Concepts to learn:
  - `pl.pallas_call`: wraps an elementwise/blockwise computation written as a
    "kernel function" into an ordinary function traceable by jax.jit.
  - The kernel function's parameters are *Refs* (references), not plain
    arrays: `x_ref[...]` reads out the data, and assigning to `o_ref[...]`
    writes the output. This is because a TPU kernel operates directly on
    buffers in on-chip memory (VMEM), not by passing values the way an
    ordinary JAX function does.
  - No `grid` / `in_specs` / `out_specs` are specified here: by default the
    whole array is treated as a single block and moved into VMEM all at once.
    This is simplest when the arrays are small (much smaller than ~16MB VMEM).
  - `interpret=True`: has pallas_call simulate the kernel's semantics on
    ordinary XLA/CPU, no real TPU needed. Good for verifying logic correctness
    first on a machine without a TPU.

To run:
  python 01_vector_add.py
"""

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl


def add_kernel(x_ref, y_ref, o_ref):
  # x_ref, y_ref, o_ref are all Refs to the whole array (since no BlockSpec/grid was specified).
  o_ref[...] = x_ref[...] + y_ref[...]


def add_vectors(x: jax.Array, y: jax.Array) -> jax.Array:
  has_tpu = any(d.platform == "tpu" for d in jax.devices())
  return pl.pallas_call(
      add_kernel,
      out_shape=jax.ShapeDtypeStruct(x.shape, x.dtype),
      interpret=not has_tpu,
  )(x, y)


if __name__ == "__main__":
  key = jax.random.key(0)
  kx, ky = jax.random.split(key)
  x = jax.random.normal(kx, (8, 128), dtype=jnp.float32)
  y = jax.random.normal(ky, (8, 128), dtype=jnp.float32)

  out = add_vectors(x, y)
  expected = x + y

  max_err = jnp.max(jnp.abs(out - expected))
  print("devices:", jax.devices())
  print("max abs error vs. plain jnp:", max_err)
  assert max_err == 0.0, "vector add result is wrong, fix before continuing"
  print("OK: pallas add_vectors matches jnp")
