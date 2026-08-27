"""More systematic numerical testing of softmax_tpu.softmax.

To run:
  python softmax_tpu_test.py
"""

import jax
import jax.numpy as jnp
from softmax_tpu import softmax


def check(b: int, v: int, b_block_size: int, v_block_size: int, dtype=jnp.float32):
  key = jax.random.key(hash((b, v, b_block_size, v_block_size)) % (2**31))
  x = jax.random.normal(key, (b, v), dtype=dtype) * 10.0

  out = softmax(x, b_block_size=b_block_size, v_block_size=v_block_size)
  expected = jax.nn.softmax(x, axis=-1)

  max_err = jnp.max(jnp.abs(out - expected))
  ok = max_err < 1e-5
  status = "OK" if ok else "FAIL"
  print(
      f"[{status}] shape=({b},{v}) b_block={b_block_size} v_block={v_block_size}"
      f" max_err={max_err}"
  )
  return ok


if __name__ == "__main__":
  print("devices:", jax.devices())
  cases = [
      # (b, v, b_block_size, v_block_size)
      (128, 256, 128, 256),   # V is only 1 block, degenerates to the same as the old version
      (512, 1024, 128, 512),  # V split into 2 blocks
      (512, 1024, 256, 256),  # V split into 4 blocks
      # This case previously (old version only chunked the batch dimension)
      # hit RESOURCE_EXHAUSTED on real v6e hardware no matter how small
      # b_block was shrunk (see the correction in 01_pallas_basics/notes.md).
      # Now that V is also chunked, only one (128, 3200) block needs to be
      # resident in VMEM at a time (about 1.6MB under float32, far below
      # v6e's VMEM budget), so in theory it shouldn't OOM anymore -- remember
      # to verify this first next time this runs on real hardware.
      (1024, 32000, 128, 3200),
  ]
  results = [check(*c) for c in cases]
  assert all(results), "some test case failed, go back and check the kernel logic in softmax_tpu.py"
  print(f"all {len(results)} test cases passed")
