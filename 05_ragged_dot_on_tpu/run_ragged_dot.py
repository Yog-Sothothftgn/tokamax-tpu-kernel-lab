"""Run tokamax.ragged_dot on a real TPU VM, comparing the XLA reference
implementation against the real Pallas:Mosaic TPU kernel numerically.

Background: ragged_dot is the core op in MoE (Mixture-of-Experts) -- `lhs` is
(total token count M, hidden dim K), `rhs` is (num experts G, K, output dim
N), `group_sizes` is a length-G array specifying that the first
group_sizes[0] tokens belong to expert 0, the next group_sizes[1] tokens
belong to expert 1, and so on -- each token only does a matmul against the
weights of the expert it's assigned to.

Must be run on a TPU VM, with ~/tokamax/.venv activated:
  cd ~/tokamax && source .venv/bin/activate
  cd ~/kernel-lab/05_ragged_dot_on_tpu
  python3 run_ragged_dot.py

(The "mosaic" implementation's `supported_on` in pallas_mosaic_tpu.py:508
requires TPU generation >= 5 -- this branch can't run in a local CPU
environment.)
"""

import jax
import jax.numpy as jnp
import tokamax


def main():
  print("devices:", jax.devices())

  key = jax.random.key(0)
  k_lhs, k_rhs = jax.random.split(key)

  m, k, n, g = 512, 256, 384, 4
  # Token counts assigned to the 4 experts, deliberately uneven, summing to m.
  group_sizes = jnp.array([150, 100, 130, 132], dtype=jnp.int32)
  assert int(jnp.sum(group_sizes)) == m, "group_sizes must sum to M"

  lhs = jax.random.normal(k_lhs, (m, k), dtype=jnp.float32)
  rhs = jax.random.normal(k_rhs, (g, k, n), dtype=jnp.float32)

  print("\n--- implementation='xla' (reference implementation) ---")
  out_xla = tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="xla")
  print("output shape:", out_xla.shape, "dtype:", out_xla.dtype)

  print("\n--- implementation='mosaic' (automatically selects mosaic_tpu on TPU v5e) ---")
  out_tpu = tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="mosaic")
  print("output shape:", out_tpu.shape, "dtype:", out_tpu.dtype)

  max_err = jnp.max(jnp.abs(out_xla - out_tpu))
  rel_err = max_err / (jnp.max(jnp.abs(out_xla)) + 1e-8)
  print(f"\nmax abs diff: {max_err}")
  print(f"max rel diff: {rel_err}")

  # The MXU commonly uses lower precision internally for multiply-accumulate;
  # some error is expected under float32 inputs, but it shouldn't be extreme.
  assert rel_err < 1e-2, "xla and mosaic_tpu results differ too much, worth digging into"
  print("\nOK: xla and mosaic_tpu(v5e) ragged_dot results match (within tolerance)")


if __name__ == "__main__":
  main()
