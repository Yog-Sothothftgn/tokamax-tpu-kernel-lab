"""Control experiment: where should the bf16 unpack run -- inside the
SparseCore kernel, or outside it on the TensorCore (2026-10-02), per
explicit user-approved plan. A single variable changes; everything else is
held fixed (column-half packing built live on every call, W=8, cross-core
split with the same padding, identical real production indices, explicit
jit arguments, same correctness/timing methodology):

  version              inside the SparseCore kernel    outside the kernel
  sc_unpack_inside     gather + unpack, two bf16       concatenate, mask
                       output streams
  sc_unpack_outside    gather only, one packed int32   unpack, concatenate, mask
                       output stream

(plus plain XLA bf16 as the same-round reference).

It answers two questions directly:
  1. How much shorter is the SparseCore interval once the in-kernel unpack
     is gone? (device trace: kernel wall, per-step `ep_run_kernel`.)
  2. Is the COMPLETE call faster when plain XLA does the unpack?
     (same-round timing, pipelined and per-call, all costs included: live
     packing, SparseCore work, unpack, concatenate, mask.)

Caveat stated up front: the kernel's output layout and write-back also
change (one int32 stream instead of two bf16 streams carrying the same
total bytes), so the difference is NOT pure unpack-instruction time -- it
is a placement comparison, which is what decides whether the work belongs
inside or outside the SparseCore. Nothing here is a prediction; the
numbers come from the run.

Final output is checked value-for-value AND bit-for-bit (raw uint16 view,
which would catch -0.0 vs 0.0) against XLA bf16.

To run (real v6e VM only):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_gather_unpack_location.py
"""

import functools
import pathlib
import sys
import traceback

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sparsecore_gather_prototype import LATENT_SIZE, real_dispatch_indices  # noqa: E402
from sparsecore_gather_group_b_comparison import (  # noqa: E402
    NUM_TOKENS,
    LOCAL_NUM_EXPERTS,
    _run_comparison,
    xla_gather,
)
from sparsecore_gather_bf16_colhalf import (  # noqa: E402
    _named,
    sparsecore_gather_colhalf_bf16,
    sparsecore_gather_colhalf_bf16_unpack_outside,
)
import sparsecore_gather_device_trace as dt  # noqa: E402

TRACE_DIR_UNPACK = "/tmp/sparsecore_unpack_location_trace"


def run_unpack_location_experiment(seed: int = 0) -> None:
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")
  x_bf16, padded_token_idx, valid_mask, _ = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} num_valid={int(jnp.sum(valid_mask))}"
  )
  args = (x_bf16, padded_token_idx, valid_mask)

  candidates = {
      "xla_bf16": xla_gather,
      "sc_unpack_inside": functools.partial(
          sparsecore_gather_colhalf_bf16, window_size=8, core_split=True),
      "sc_unpack_outside": functools.partial(
          sparsecore_gather_colhalf_bf16_unpack_outside, window_size=8, core_split=True),
  }

  expected = jax.jit(xla_gather)(*args)
  jax.block_until_ready(expected)
  working = {}
  for name, fn in candidates.items():
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
    except Exception as e:  # noqa: BLE001 -- surfacing the real error is the point
      print(f"\n[{name}] FAILED to compile/run -- full error below:\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
      continue
    ok_val = bool(jnp.array_equal(out, expected))
    ok_bits = bool(jnp.array_equal(out.view(jnp.uint16), expected.view(jnp.uint16)))
    print(f"[{name}] COMPILED; value match={ok_val}, raw-bit match={ok_bits}")
    if ok_val and ok_bits:
      working[name] = fn
  if len(working) < 2:
    print("\nFewer than 2 working variants -- stopping before timing/trace.")
    return

  print("\n" + "=" * 78 + "\nTiming (bf16, real scale): unpack inside vs outside the SparseCore + xla\n" + "=" * 78)
  _run_comparison(working, args, label="unpack-location", bitwise_check=True)

  runs = {n: (jax.jit(_named(n, fn)), x_bf16) for n, fn in working.items()}
  print("\n" + "=" * 78 + "\nDevice trace attribution\n" + "=" * 78)
  dt.capture_trace_runs(runs, padded_token_idx, valid_mask, trace_dir=TRACE_DIR_UNPACK)
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir=TRACE_DIR_UNPACK)


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  if pltpu.get_tpu_info().sparse_core is None:
    print("No SparseCore on this TPU -- cannot run this experiment here.")
    raise SystemExit(1)
  run_unpack_location_experiment()
