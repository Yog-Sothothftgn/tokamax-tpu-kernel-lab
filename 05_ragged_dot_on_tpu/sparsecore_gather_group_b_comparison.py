"""Group B, scaled down per explicit user request (2026-10-01): compares
exactly THREE gather implementations at real production scale, answering
one question each:

  implementation                                    question
  -------------------------------------------------  ----------------------------------
  XLA gather                                         how fast is the current baseline?
  SparseCore single-kernel, 128-column chunking      how fast was the earlier chunked plan?
  SparseCore single-kernel, W=8, whole-row (no chunk) how much does NOT chunking help?

`W=16` deliberately excluded -- only `W=8` has been confirmed correct at
the real `LATENT_SIZE=3584` (via `sparsecore_gather_window_size_diagnosis.
py`'s `run_a3_real_width`); `W=16` has not been validated at this width.

Scope: INT32 ONLY, per the user's own established "int32 first, bf16
last, don't mix variables" methodology -- `x` is a real-shaped but
synthetic int32 stand-in (same non-degeneracy fix as
`sparsecore_gather_prototype.py`'s `check_chunked`: `jnp.arange`-based
distinct values, NOT `x.astype(int32)` on small-scale bf16 values, which
truncates almost everything to 0). Real bf16 packing is a SEPARATE, later
step, only once one of these three implementations is confirmed worth it.

Methodology (matching this project's established discipline):
  - Correctness checked FIRST, exact match required (pure data movement,
    no rounding-noise tolerance) -- timing is not measured for an
    implementation that fails correctness.
  - All three implementations use the SAME real production
    `padded_token_idx`/`valid_mask` (from
    `route_and_filter_to_local_shard_jittable`, real `num_tokens=2048`,
    real `local_num_experts=64`, real `m_padded`) and the SAME masking
    logic, included inside the timed region for all three.
  - 10 rounds x 20 calls per implementation per round; the THREE
    implementations' order ROTATES every round (not fixed A-then-B-then-C)
    to cancel systematic drift (thermal, scheduling) that a fixed order
    could alias into a fake "X is always faster" pattern.
  - BOTH timing conventions kept separate and reported separately
    (pipelined/throughput-style vs. per-call blocking) -- never called
    "pure device time".
  - Reports MEDIAN and min/max spread across the 10 rounds, not just the
    single fastest run.

To run (real v6e VM only -- SparseCore ops have no CPU interpret-mode
stub):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_gather_group_b_comparison.py \\
    2>&1 | tee sparsecore_group_b_comparison.log
"""

import functools
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from sparsecore_gather_prototype import (  # noqa: E402
    LATENT_SIZE,
    real_dispatch_indices,
    sparsecore_gather_chunked_single_kernel,
)

NUM_TOKENS = 2048
LOCAL_NUM_EXPERTS = 64


def xla_gather(x: jax.Array, padded_token_idx: jax.Array, valid_mask: jax.Array) -> jax.Array:
  """The current production baseline -- plain XLA gather, masked the same
  way `filter_and_pad_to_shard_jittable` computes its real `sorted_tokens`
  (see `sparsecore_gather_prototype.py`'s `current_jit_gather` for the
  full equivalence argument).
  """
  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx)
  gathered = x[safe_idx]
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def sparsecore_gather_whole_row_w8(
    x_int32: jax.Array, padded_token_idx: jax.Array, valid_mask: jax.Array, window_size: int = 8
) -> jax.Array:
  """SparseCore, ONE kernel launch, `window_size=8`, WHOLE ROW (no column
  chunking at all -- `chunk_width=value_dim=LATENT_SIZE=3584` in one
  `sync_copy`, not 28 separate ones). Structurally the exact 1D, REF-based
  index convention confirmed correct at this real width by
  `sparsecore_gather_window_size_diagnosis.py`'s `run_a3_real_width`/
  `gather_once(indices_2d=False, materialize_index=False)` -- adapted here
  to the production `padded_token_idx` (-1 sentinel) / `valid_mask`
  convention instead of that diagnostic's plain test indices.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  num_tokens, value_dim = x_int32.shape
  assert value_dim == LATENT_SIZE, f"expected value_dim={LATENT_SIZE}, got {value_dim}"
  assert x_int32.dtype == jnp.int32, f"expected int32, got {x_int32.dtype}"
  num_indices = padded_token_idx.shape[0]
  assert num_indices % window_size == 0

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this comparison"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)

  @pl.kernel(out_type=jax.ShapeDtypeStruct((num_indices, value_dim), jnp.int32), mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      pltpu.sync_copy(x_hbm.at[i_vmem], o_vmem)  # 1D ref -- the confirmed-working convention

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // window_size,),
        in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
        out_specs=[pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  gathered = kernel(x_int32, safe_idx)
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def _time_pipelined(f, *args, num_repeats: int = 20) -> float:
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def _time_blocking(f, *args, num_repeats: int = 20) -> float:
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
    jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def _median_spread(vals: list[float]) -> dict:
  s = sorted(vals)
  n = len(s)
  median = s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2
  return {"median": median, "min": s[0], "max": s[-1]}


def run_group_b(num_rounds: int = 10, num_repeats: int = 20, seed: int = 0) -> None:
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")
  print(f"seed: {seed}")

  x_bf16, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} "
      f"num_valid={int(jnp.sum(valid_mask))}"
  )

  # Non-degenerate int32 stand-in for the real bf16 activation -- real
  # VALUES don't matter here (int32 is a shape/mechanism proxy, bf16
  # packing is a later step), but they must be genuinely distinct: casting
  # the real (normal()*0.02-scale) bf16 straight to int32 truncates nearly
  # everything to 0 (the exact bug sparsecore_gather_prototype.py's
  # check_chunked hit and fixed).
  x_int32 = jnp.arange(NUM_TOKENS * LATENT_SIZE, dtype=jnp.int32).reshape(NUM_TOKENS, LATENT_SIZE)

  impls = {
      "xla": xla_gather,
      "sparsecore_128chunk_1launch": functools.partial(
          sparsecore_gather_chunked_single_kernel, chunk_width=128, gather_window_size=128
      ),
      "sparsecore_wholerow_w8": functools.partial(sparsecore_gather_whole_row_w8, window_size=8),
  }
  names = list(impls.keys())

  # Correctness FIRST, for all three, before any timing.
  expected = xla_gather(x_int32, padded_token_idx, valid_mask)
  jitted = {}
  all_ok = True
  for name, fn in impls.items():
    f = jax.jit(fn)
    out = f(x_int32, padded_token_idx, valid_mask)
    jax.block_until_ready(out)
    ok = bool(jnp.array_equal(out, expected))
    print(f"[correctness] {name}: {'OK' if ok else 'FAIL'}")
    all_ok = all_ok and ok
    jitted[name] = f
  if not all_ok:
    print("\nCorrectness FAILED for at least one implementation -- stopping before any timing.")
    return

  pipe_times = {n: [] for n in names}
  block_times = {n: [] for n in names}

  for round_idx in range(num_rounds):
    rot = round_idx % len(names)
    order = names[rot:] + names[:rot]  # rotate the comparison order every round
    for name in order:
      f = jitted[name]
      pipe_times[name].append(
          _time_pipelined(f, x_int32, padded_token_idx, valid_mask, num_repeats=num_repeats)
      )
      block_times[name].append(
          _time_blocking(f, x_int32, padded_token_idx, valid_mask, num_repeats=num_repeats)
      )
    print(f"[round {round_idx}] order={order}")

  print(f"\n[Group B results -- int32, real scale, {num_rounds} rounds x {num_repeats} calls each]")
  for name in names:
    p = _median_spread(pipe_times[name])
    b = _median_spread(block_times[name])
    print(
        f"  {name:28s} pipelined: median={p['median']:.4f}ms (min={p['min']:.4f} max={p['max']:.4f})  "
        f"per-call: median={b['median']:.4f}ms (min={b['min']:.4f} max={b['max']:.4f})"
    )

  xla_pipe_median = _median_spread(pipe_times["xla"])["median"]
  xla_block_median = _median_spread(block_times["xla"])["median"]
  print("\n[speedup vs xla (median), pipelined / per-call]")
  for name in names:
    if name == "xla":
      continue
    p_med = _median_spread(pipe_times[name])["median"]
    b_med = _median_spread(block_times[name])["median"]
    print(
        f"  {name:28s} pipelined_speedup={xla_pipe_median / p_med:.3f}x  "
        f"per_call_speedup={xla_block_median / b_med:.3f}x"
    )


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    print("No SparseCore on this TPU -- cannot run this comparison here.")
    raise SystemExit(1)
  print(f"sparse_core info: {sc_info}")
  run_group_b()
