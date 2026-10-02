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
          jax.ShapeDtypeStruct((num_indices, value_dim), jnp.bfloat16),
          jax.ShapeDtypeStruct((num_indices, value_dim), jnp.bfloat16),
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
        grid=(num_indices // window_size,),
        in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
        out_specs=[
            pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0)),
            pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0)),
        ],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_low_hbm, o_high_hbm)

  low_out, high_out = kernel(x_packed, safe_idx)
  # Interleave low/high back into the SAME column order `.view(bf16)`
  # would have produced (position 2j=low[j], 2j+1=high[j]), then select
  # the even- or odd-original-row half based on the real gathered index
  # -- all plain JAX, outside the kernel, no SC restriction here.
  unpacked_concat = jnp.stack([low_out, high_out], axis=-1).reshape(num_indices, 2 * value_dim)
  pairs = unpacked_concat.reshape(num_indices, packing, value_dim)
  is_odd = (safe_idx % packing)[:, None]
  gathered = jnp.where(is_odd == 1, pairs[:, 1], pairs[:, 0])
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


def _run_comparison(
    impls: dict,
    args: tuple,
    label: str,
    num_rounds: int = 10,
    num_repeats: int = 20,
) -> bool:
  """Shared methodology for both the int32 Group B round and the bf16
  round below: correctness FIRST (exact match against the first
  implementation's own output, which callers should order so it's the
  trusted baseline), then `num_rounds` rounds x `num_repeats` calls each,
  rotating the comparison order every round, both timing conventions kept
  separate, reporting median and min/max spread. Returns whether all
  implementations passed correctness (timing is skipped entirely if not).
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
