"""Group B, scaled down per explicit user request (2026-10-01), TWO rounds
in this file:

**Round 1, `run_group_b` (int32)**: compares exactly THREE gather
implementations at real production scale, answering one question each:

  implementation                                    question
  -------------------------------------------------  ----------------------------------
  XLA gather                                         how fast is the current baseline?
  SparseCore single-kernel, 128-column chunking      how fast was the earlier chunked plan?
  SparseCore single-kernel, W=8, whole-row (no chunk) how much does NOT chunking help?

`W=16` deliberately excluded -- only `W=8` has been confirmed correct at
the real `LATENT_SIZE=3584` (via `sparsecore_gather_window_size_diagnosis.
py`'s `run_a3_real_width`); `W=16` has not been validated at this width.
Scope: INT32 ONLY, per the user's own established "int32 first, bf16 last,
don't mix variables" methodology -- `x` is a real-shaped but synthetic
int32 stand-in (same non-degeneracy fix as
`sparsecore_gather_prototype.py`'s `check_chunked`: `jnp.arange`-based
distinct values, NOT `x.astype(int32)` on small-scale bf16 values, which
truncates almost everything to 0).

**Round 2, `run_bf16_comparison` (REAL bf16)**, added once round 1 showed
whole-row-W8 was the best int32 candidate (still ~11x/~5x slower than
XLA, but ~2.3x faster than the 128-chunk version) -- per explicit user
request, keeps the SAME indices/W=8/whole-row shape, now at real bf16:

  1. XLA bf16 gather (real production dtype baseline)
  2. SparseCore, x ALREADY packed into int32 pairs-of-rows (packing cost
     excluded from the timed call -- best case)
  3. SparseCore, plain bf16 x, packed ON THE FLY inside the timed call
     (the realistic case for a live forward pass)

This exact combination (bf16 packing x whole-row x W=8) had not been
tried before -- bf16 adds an unpack step and a temporary `gather_vmem`
scratch buffer the int32 round never needed, so whether it even compiles
is checked here (full error captured if not), not assumed.

Methodology (matching this project's established discipline, both rounds):
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


def sparsecore_gather_whole_row_w8_bf16(
    x: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    window_size: int = 8,
    repack_every_call: bool = True,
    core_split: bool = False,
) -> jax.Array:
  """Whole-row (no column chunking), `W=8`, SparseCore gather at REAL
  bf16 -- packs PAIRS OF ADJACENT ROWS into int32 (SparseCore DMA only
  natively moves 32-bit words), applied to the confirmed-working whole-
  row/`W=8`/1D-ref index convention. The official guide's own
  `gather_bf16_packed` example unpacks via a direct `int32 -> bfloat16`
  `.view()` inside the kernel -- that is CONFIRMED NOT IMPLEMENTED on
  this SC backend ("Changing bitwidths not supported"), found and worked
  around via the isolated `sparsecore_bf16_bitwise_unpack_probe.py`
  (2026-10-02): unpack via bitwise `&`/`>>`/narrowing-`astype` to get two
  `uint16` halves, then a SAME-WIDTH `uint16 -> bfloat16` `.view()`
  (confirmed working), with the interleave-back-to-original-order and
  even/odd-row selection moved OUTSIDE the kernel (stack/reshape are
  restricted inside the SC kernel body too, per that same probe).

  `repack_every_call`: `True` (realistic -- `x` arrives as plain bf16
  every forward pass, the reshape/view packing cost is INSIDE this jitted
  function) vs `False` (best case -- caller already has `x` in packed
  int32 layout, shape `(num_tokens // 2, LATENT_SIZE)`, packing cost
  excluded from this call).
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  packing = 2  # 32 // 16, bf16 -> int32
  num_indices = padded_token_idx.shape[0]
  assert num_indices % window_size == 0

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this comparison"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)

  # Optional cross-SparseCore split (see sparsecore_gather_core_split.py):
  # core_axis_name=('core','subcore') partitions ONE flat grid over both
  # SparseCores x 16 subcores instead of each SparseCore running the whole
  # grid. The grid must be divisible by the 32 workers, so the index array
  # is padded with zeros (valid in-bounds gathers) to a multiple of
  # window*cores*subcores and the kernel outputs are sliced back below.
  if core_split:
    quantum = window_size * sc_info.num_cores * sc_info.num_subcores
    pad = (-num_indices) % quantum
  else:
    pad = 0
  kernel_n = num_indices + pad
  safe_idx_k = jnp.pad(safe_idx, (0, pad)) if pad else safe_idx
  core_axes = ('core', 'subcore') if core_split else 'subcore'

  # value_dim is DERIVED from x's own shape, not hardcoded to the real
  # LATENT_SIZE=3584 -- lets this same function run both the real-width
  # case and a reduced-width VMEM-capacity control (per explicit user
  # request, 2026-10-02: keep W=8 and every other variable fixed, halve
  # ONLY value_dim, to check whether the whole bf16 pipeline is correct
  # when it comfortably fits VMEM, before deciding how to cover the real
  # 3584 columns).
  if repack_every_call:
    num_tokens, value_dim = x.shape
    assert x.dtype == jnp.bfloat16, f"repack_every_call=True expects bfloat16, got {x.dtype}"
    assert num_tokens % packing == 0, "row-pairing needs an even token count"
    x_packed = x.reshape(num_tokens // packing, packing * value_dim).view(jnp.int32)
  else:
    _num_packed_rows, value_dim = x.shape
    assert x.dtype == jnp.int32, f"repack_every_call=False expects already-packed int32, got {x.dtype}"
    x_packed = x

  # Unpack strategy CONFIRMED on real hardware via the isolated
  # sparsecore_bf16_bitwise_unpack_probe.py (2026-10-02): a WIDTH-CHANGING
  # int32->bfloat16 `.view()` inside the kernel is not implemented on SC
  # ("Changing bitwidths not supported"), but a SAME-WIDTH uint16->
  # bfloat16 `.view()`, reached via ordinary bitwise extraction (`&`,
  # `>>`, narrowing `.astype`), IS implemented. That probe also found
  # `jnp.stack`/reshape INSIDE the kernel hits a separate SC limitation
  # ("Reshape is not a no-op") -- so this kernel does ONLY shape-
  # preserving, dtype-changing ops (astype/view) inside, and produces TWO
  # separate same-shape outputs (low 16 bits, high 16 bits of each packed
  # int32 word, each reinterpreted as bf16) instead of one combined
  # array. The interleave-back-into-original-column-order (low/high ->
  # original bf16 layout) and the even/odd-original-row selection both
  # happen OUTSIDE the kernel in plain JAX, where stack/reshape have no
  # SC restriction.
  @pl.kernel(
      out_type=(
          jax.ShapeDtypeStruct((kernel_n, value_dim), jnp.bfloat16),
          jax.ShapeDtypeStruct((kernel_n, value_dim), jnp.bfloat16),
      ),
      mesh=vector_mesh,
      scratch_types=dict(gather_vmem=pltpu.VMEM((window_size, value_dim), jnp.int32)),
  )
  def kernel(x_packed_hbm, i_hbm, o_low_hbm, o_high_hbm, *, gather_vmem):
    def body(idx_vmem, o_low_vmem, o_high_vmem):
      # jax.lax.div/mod directly on a ref raised "Triggering
      # __jax_array__() ... no longer supported" -- materialize once,
      # use consistently (window_size=8 == this hardware's num_lanes,
      # which matters for other materialized-index paths, not this div).
      idx_val = idx_vmem[...]
      pltpu.sync_copy(x_packed_hbm.at[jax.lax.div(idx_val, packing)], gather_vmem)
      raw = gather_vmem[...].astype(jnp.uint32)  # materialize + same-width (32->32) reinterpret
      low16 = (raw & 0xFFFF).astype(jnp.uint16)  # standard integer narrow, not a bitcast
      high16 = (raw >> 16).astype(jnp.uint16)    # uint32 >> is a logical (zero-fill) shift
      o_low_vmem[...] = low16.view(jnp.bfloat16)   # SAME-WIDTH (16->16) bitcast -- confirmed working
      o_high_vmem[...] = high16.view(jnp.bfloat16)

    pltpu.emit_pipeline(
        body,
        grid=(kernel_n // window_size,),
        in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
        out_specs=[
            pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0)),
            pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0)),
        ],
        core_axis_name=core_axes,
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_low_hbm, o_high_hbm)

  low_out, high_out = kernel(x_packed, safe_idx_k)
  if pad:
    low_out, high_out = low_out[:num_indices], high_out[:num_indices]
  # Interleave low/high back into the SAME column order `.view(bf16)`
  # would have produced (position 2j=low[j], 2j+1=high[j]), then select
  # the even- or odd-original-row half based on the real gathered index
  # -- all plain JAX, outside the kernel, no SC restriction here.
  unpacked_concat = jnp.stack([low_out, high_out], axis=-1).reshape(num_indices, 2 * value_dim)
  pairs = unpacked_concat.reshape(num_indices, packing, value_dim)
  is_odd = (safe_idx % packing)[:, None]
  gathered = jnp.where(is_odd == 1, pairs[:, 1], pairs[:, 0])
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def sparsecore_gather_two_chunk_w8_bf16(
    x: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    window_size: int = 8,
    repack_every_call: bool = True,
    core_split: bool = False,
) -> jax.Array:
  """Covers the REAL `LATENT_SIZE=3584` by calling the CONFIRMED-CORRECT
  half-width (1792) `sparsecore_gather_whole_row_w8_bf16` kernel TWICE,
  once per 1792-wide column chunk, then concatenating -- per explicit
  user decision (2026-10-02), the lower-risk of two options after the
  real-width whole-shot attempt hit a genuine VMEM budget overflow
  (`CompileTimeSparseCoreAllocationFailure`, ~86143 words needed vs a
  65536-word budget) while the half-width capacity control
  (`run_bf16_half_width_capacity_check`) confirmed the underlying
  pipeline logic itself is correct. Reuses that exact, already-validated
  kernel verbatim (its `value_dim` is derived from the input's own shape,
  so no new kernel code is needed) -- only 2 chunks here, vs the earlier
  int32 investigation's 28 chunks of 128 columns each.

  Each chunk call applies the SAME real `valid_mask` independently (masks
  to zero at invalid rows in both halves identically), so concatenating
  the two already-masked halves gives the correct final result without
  masking twice.
  """
  num_tokens, value_dim = x.shape
  assert value_dim % 2 == 0, f"value_dim={value_dim} must be evenly splittable into 2 chunks"
  half = value_dim // 2
  chunk0 = sparsecore_gather_whole_row_w8_bf16(
      x[:, :half], padded_token_idx, valid_mask, window_size=window_size, repack_every_call=repack_every_call,
      core_split=core_split,
  )
  chunk1 = sparsecore_gather_whole_row_w8_bf16(
      x[:, half:], padded_token_idx, valid_mask, window_size=window_size, repack_every_call=repack_every_call,
      core_split=core_split,
  )
  return jnp.concatenate([chunk0, chunk1], axis=-1)


def run_two_chunk_bf16_correctness_check(seed: int = 0) -> bool:
  """Correctness + compile-ability ONLY (no timing yet, per established
  discipline -- confirm it works at the REAL width before measuring
  anything) for `sparsecore_gather_two_chunk_w8_bf16`, at real production
  scale: real `num_tokens=2048`, real `padded_token_idx`/`valid_mask`,
  real `LATENT_SIZE=3584` bf16 data. Checked against the plain XLA bf16
  gather, exact match required (pure data movement).
  """
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")

  x_bf16, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} (2 chunks of {LATENT_SIZE // 2}) "
      f"num_valid={int(jnp.sum(valid_mask))}"
  )

  def xla_fn():
    return xla_gather(x_bf16, padded_token_idx, valid_mask)

  def sc_fn():
    return sparsecore_gather_two_chunk_w8_bf16(
        x_bf16, padded_token_idx, valid_mask, window_size=8, repack_every_call=True
    )

  try:
    expected = jax.jit(xla_fn)()
    jax.block_until_ready(expected)
    out = jax.jit(sc_fn)()
    jax.block_until_ready(out)
  except Exception as e:  # noqa: BLE001 -- surfacing the real error is the point
    import traceback
    print(
        "\n[two-chunk bf16] FAILED to compile/run -- full error below:\n"
        + "".join(traceback.format_exception(type(e), e, e.__traceback__))
    )
    return False

  ok = bool(jnp.array_equal(out, expected))
  print(
      f"[two-chunk bf16] value_dim={LATENT_SIZE} via 2x{LATENT_SIZE // 2}, W=8: "
      f"COMPILED, correct={ok} (expect exact match -- pure data movement)"
  )
  return ok


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


def _run_comparison(
    impls: dict,
    args: tuple,
    label: str,
    num_rounds: int = 10,
    num_repeats: int = 20,
    bitwise_check: bool = False,
) -> bool:
  """Shared methodology for both the int32 Group B round and the bf16
  round below: correctness FIRST (exact match against the first
  implementation's own output, which callers should order so it's the
  trusted baseline), then `num_rounds` rounds x `num_repeats` calls each,
  rotating the comparison order every round, both timing conventions kept
  separate, reporting median and min/max spread. Returns whether all
  implementations passed correctness (timing is skipped entirely if not).

  `bitwise_check=True` (per explicit user request for bf16 comparisons)
  ADDITIONALLY requires the raw bit patterns to match
  (`.view(jnp.uint16)`), not just value equality via `array_equal` --
  `array_equal` alone would treat `-0.0 == 0.0` as equal (numerically
  correct, but would hide a real bit-level unpack bug for that specific
  value). Meaningful for float dtypes; harmless but unused for int32
  callers (who pass the default `False`).
  """
  names = list(impls.keys())
  reference_name = names[0]

  jitted = {name: jax.jit(fn) for name, fn in impls.items()}
  outputs = {}
  for name, f in jitted.items():
    out = f(*args)
    jax.block_until_ready(out)
    outputs[name] = out

  expected = outputs[reference_name]
  all_ok = True
  for name in names:
    ok = bool(jnp.array_equal(outputs[name], expected))
    if bitwise_check:
      ok = ok and bool(
          jnp.array_equal(outputs[name].view(jnp.uint16), expected.view(jnp.uint16))
      )
    print(f"[correctness, {label}] {name}: {'OK' if ok else 'FAIL'} (vs {reference_name})")
    all_ok = all_ok and ok
  if not all_ok:
    print(f"\n[{label}] correctness FAILED for at least one implementation -- stopping before timing.")
    return False

  pipe_times = {n: [] for n in names}
  block_times = {n: [] for n in names}
  for round_idx in range(num_rounds):
    rot = round_idx % len(names)
    order = names[rot:] + names[:rot]
    for name in order:
      f = jitted[name]
      pipe_times[name].append(_time_pipelined(f, *args, num_repeats=num_repeats))
      block_times[name].append(_time_blocking(f, *args, num_repeats=num_repeats))
    print(f"[round {round_idx}, {label}] order={order}")

  print(f"\n[{label} results -- real scale, {num_rounds} rounds x {num_repeats} calls each]")
  for name in names:
    p = _median_spread(pipe_times[name])
    b = _median_spread(block_times[name])
    print(
        f"  {name:32s} pipelined: median={p['median']:.4f}ms (min={p['min']:.4f} max={p['max']:.4f})  "
        f"per-call: median={b['median']:.4f}ms (min={b['min']:.4f} max={b['max']:.4f})"
    )

  ref_pipe_median = _median_spread(pipe_times[reference_name])["median"]
  ref_block_median = _median_spread(block_times[reference_name])["median"]
  print(f"\n[{label}: speedup vs {reference_name} (median), pipelined / per-call]")
  for name in names:
    if name == reference_name:
      continue
    p_med = _median_spread(pipe_times[name])["median"]
    b_med = _median_spread(block_times[name])["median"]
    print(
        f"  {name:32s} pipelined_speedup={ref_pipe_median / p_med:.3f}x  "
        f"per_call_speedup={ref_block_median / b_med:.3f}x"
    )
  return True


def run_group_b(num_rounds: int = 10, num_repeats: int = 20, seed: int = 0) -> None:
  """INT32-only round (unchanged from the original Group B run)."""
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")
  print(f"seed: {seed}")

  _x_bf16, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
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
  _run_comparison(impls, (x_int32, padded_token_idx, valid_mask), label="int32", num_rounds=num_rounds, num_repeats=num_repeats)


def run_bf16_comparison(num_rounds: int = 10, num_repeats: int = 20, seed: int = 0) -> None:
  """REAL bf16 round, per explicit user request: same real indices, same
  W=8/whole-row candidate, now three implementations: (1) plain XLA bf16
  gather, (2) SparseCore with x ALREADY packed into int32 (packing cost
  excluded), (3) SparseCore with plain bf16 x, packed on the fly inside
  the timed call (the realistic case). Correctness must be exact (pure
  data movement -- no rounding-noise tolerance), checked before any
  timing, same as every other check in this file.
  """
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

  x_packed = x_bf16.reshape(NUM_TOKENS // 2, 2 * LATENT_SIZE).view(jnp.int32)

  # Each implementation needs DIFFERENT x -- xla_bf16/repack_live take
  # plain x_bf16, prepacked takes x_packed. _run_comparison's `args` are
  # shared across implementations, so here each callable closes over the
  # right data and takes none, rather than threading a placeholder arg
  # through the shared helper.
  wrapped = {
      "xla_bf16": lambda: xla_gather(x_bf16, padded_token_idx, valid_mask),
      "sparsecore_prepacked": lambda: sparsecore_gather_whole_row_w8_bf16(
          x_packed, padded_token_idx, valid_mask, window_size=8, repack_every_call=False
      ),
      "sparsecore_repack_live": lambda: sparsecore_gather_whole_row_w8_bf16(
          x_bf16, padded_token_idx, valid_mask, window_size=8, repack_every_call=True
      ),
  }
  try:
    _run_comparison(
        wrapped, (), label="bf16",
        num_rounds=num_rounds, num_repeats=num_repeats,
    )
  except Exception as e:  # noqa: BLE001 -- surfacing the real compile error is the point
    import traceback
    print(
        "\n[bf16] at least one implementation FAILED to compile/run -- "
        "full error below (bf16 packing adds an unpack step and a "
        "temporary scratch buffer this exact combination hadn't been "
        "tried with before):\n"
        + "".join(traceback.format_exception(type(e), e, e.__traceback__))
    )


def run_bf16_half_width_capacity_check(seed: int = 0) -> bool:
  """VMEM-capacity control (2026-10-02), per explicit user request, NOT a
  production fix: the real-width (`value_dim=LATENT_SIZE=3584`) bf16
  whole-row `W=8` gather failed to compile with
  `CompileTimeSparseCoreAllocationFailure` (needs ~86143 words vs a
  65536-word budget -- the extra `gather_vmem` scratch plus TWO
  double-buffered bf16 outputs roughly triple the per-step VMEM
  footprint compared to the int32 version, which only just fit at this
  window size). This control keeps EVERY other variable fixed --
  `window_size=8`, the real production `padded_token_idx`/`valid_mask`,
  the exact same bitwise-unpack/two-output/external-reassembly kernel --
  and changes ONLY `value_dim`: 3584 -> 1792 (half), where the estimated
  footprint (~43000 words) comfortably fits. If this compiles and is
  bit-exact correct, it confirms the bf16 PIPELINE LOGIC itself is
  right and the real blocker is purely this implementation's VMEM
  budget at the real width -- the next decision (chunk the real 3584
  columns, or shrink buffers some other way) can then be made knowing
  that, rather than conflated with "does the bf16 approach even work".
  """
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")

  _x_bf16_real, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  half_value_dim = LATENT_SIZE // 2  # 1792
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={half_value_dim} "
      f"(HALF of real {LATENT_SIZE}, capacity control only) num_valid={int(jnp.sum(valid_mask))}"
  )

  # Synthetic bf16 table at HALF width -- indices/mask are the real,
  # unchanged production ones; only the data table's column count differs.
  key = jax.random.key(seed)
  x_half = (jax.random.normal(key, (NUM_TOKENS, half_value_dim)) * 0.02).astype(jnp.bfloat16)

  def xla_fn():
    return xla_gather(x_half, padded_token_idx, valid_mask)

  def sc_fn():
    return sparsecore_gather_whole_row_w8_bf16(
        x_half, padded_token_idx, valid_mask, window_size=8, repack_every_call=True
    )

  try:
    expected = jax.jit(xla_fn)()
    jax.block_until_ready(expected)
    out = jax.jit(sc_fn)()
    jax.block_until_ready(out)
  except Exception as e:  # noqa: BLE001 -- surfacing the real error is the point
    import traceback
    print(
        "\n[half-width capacity check] FAILED to compile/run -- full error below:\n"
        + "".join(traceback.format_exception(type(e), e, e.__traceback__))
    )
    return False

  ok = bool(jnp.array_equal(out, expected))
  print(
      f"[half-width capacity check] value_dim={half_value_dim}, W=8: "
      f"COMPILED, correct={ok} (expect exact match -- pure data movement)"
  )
  return ok


def run_two_chunk_bf16_timing(num_rounds: int = 10, num_repeats: int = 20, seed: int = 0) -> None:
  """Times the CONFIRMED-CORRECT two-chunk bf16 whole-row-`W=8` gather
  (`sparsecore_gather_two_chunk_w8_bf16`) against plain XLA bf16, at real
  production scale, using the SAME rigorous methodology as every other
  timed comparison in this file (correctness-first, `num_rounds` rounds x
  `num_repeats` calls, rotating order, dual timing convention, median +
  min/max spread). The int32 whole-row-`W=8` numbers (~11x/~5x slower
  than XLA) are the closest prior reference point, but this adds real
  bf16 packing/unpacking cost AND doubles the kernel launches (one per
  1792-wide chunk) on top of that -- NOT assumed to carry over, measured
  fresh here.
  """
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")
  print(f"seed: {seed}")

  x_bf16, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} (2 chunks of {LATENT_SIZE // 2}) "
      f"num_valid={int(jnp.sum(valid_mask))}"
  )

  # Pass x_bf16/padded_token_idx/valid_mask as EXPLICIT jit arguments, not
  # captured via a zero-arg closure -- per explicit review: a zero-arg
  # `lambda: f(x_bf16, ...)` closes over x_bf16 as a Python-level
  # constant, and jax.jit can then treat it as a COMPILE-TIME constant
  # (potentially constant-folding the reshape/view "repack" step away
  # entirely), which would silently defeat the whole point of
  # `repack_every_call=True` measuring LIVE per-call packing cost. With
  # explicit arguments, x_bf16 is a genuine traced input the compiled
  # function must re-process on every call.
  impls = {
      "xla_bf16": xla_gather,
      "sparsecore_two_chunk_w8": lambda x, idx, mask: sparsecore_gather_two_chunk_w8_bf16(
          x, idx, mask, window_size=8, repack_every_call=True
      ),
  }
  args = (x_bf16, padded_token_idx, valid_mask)
  _run_comparison(
      impls, args, label="two-chunk-bf16", num_rounds=num_rounds, num_repeats=num_repeats,
      bitwise_check=True,
  )


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    print("No SparseCore on this TPU -- cannot run this comparison here.")
    raise SystemExit(1)
  print(f"sparse_core info: {sc_info}")
  run_group_b()
  print("\n" + "=" * 78 + "\nMoving to the real bf16 round\n" + "=" * 78)
  run_bf16_comparison()
  print("\n" + "=" * 78 + "\nVMEM-capacity control: same pipeline, half value_dim\n" + "=" * 78)
  run_bf16_half_width_capacity_check()
  print("\n" + "=" * 78 + "\nTwo-chunk (1792+1792) coverage of the real LATENT_SIZE=3584\n" + "=" * 78)
  if run_two_chunk_bf16_correctness_check():
    print("\n" + "=" * 78 + "\nTiming the two-chunk bf16 gather vs plain XLA bf16\n" + "=" * 78)
    run_two_chunk_bf16_timing()
