"""Data-format experiment for the bf16 SparseCore gather (2026-10-02), per
explicit user request, run as a 2x2 so the effect of the data FORMAT and
the effect of the cross-core WORK SPLIT can be read separately:

                          both cores run full grid | cross-core split
  old format (adjacent-
  column pairing, stack/
  reshape reassembly)     sc_old_nosplit (baseline) | sc_old_split
  new format (column-half
  packing, below)         sc_colhalf_nosplit        | sc_colhalf_split

plus plain XLA bf16 as the reference. (The cross-core split alone was
measured separately on int32 in `sparsecore_gather_core_split.py`:
1.70x faster pipelined than the old kernel, exact, non-overlapping.)

WHY THE OLD FORMAT IS EXPENSIVE (device trace, 2026-10-02): of the old
bf16 two-chunk path's ~4.6ms/call, ~3.66ms (~79%) was TensorCore-side
layout work -- `reshape.13/.14` ~2.17ms (the `jnp.stack([low,high],-1)
.reshape(...)` reassembly, minor dimension 2) and `shift-left_reduce_
fusion` x2 ~1.26ms (XLA's lowering of the live `.view(int32)` packing of
ADJACENT COLUMN PAIRS) -- not SparseCore time.

NEW FORMAT, "column-half packing": pack column j and column j + H (H =
value_dim/2 = 1792) of the SAME row into one int32 word (low 16 bits =
left-half bf16, high 16 bits = right-half bf16). Outside the kernel this
is purely element-wise (`bf16 -> uint16` same-width bitcast, widen, shift,
or) on two contiguous column halves -- no adjacent-column pairing, so no
`stack(-1)`/minor-dim-2 relayout. Rows are gathered by TOKEN index
directly:
  - no even/odd-row selection (the old/row-pair format gathers a packed
    row holding two tokens and throws half of it away);
  - every gathered word is useful, so the gathered/written bytes are about
    half of the old format's (34MB read, 34MB written vs 68MB/68MB);
  - the kernel's two outputs (low half-words, high half-words, same-width
    `uint16 -> bfloat16` view -- the confirmed-working in-kernel unpack)
    ARE the left and right halves of the row, so reassembly outside is a
    plain `concatenate` along columns;
  - the whole row fits ONE kernel launch: scratch (8,1792) int32 + two
    double-buffered (8,1792) bf16 outputs is the same footprint that
    already compiled in the half-width capacity control, so the old
    two-column-chunk workaround is not needed.
This is a different concrete form of "element-wise packing" than the
even-row/odd-row variant first sketched; it dominates it on bytes and on
needing no parity select, and is flagged here because it is a design
choice, not something the user specified.

All variants are timed in ONE round with the same methodology as the rest
of this investigation (correctness first incl. a raw-bit comparison that
would catch -0.0 vs 0.0; 10 rounds x 20 calls; rotating order; pipelined
and per-call; EXPLICIT jit arguments so the live repack is genuinely
re-executed each call). A device trace of every variant is then analysed
by device-side module span (TensorCore op breakdown, SparseCore kernel
time, `ep_run_kernel` coverage) to attribute where each variant's time
goes.

To run (real v6e VM only):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_gather_bf16_colhalf.py
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
    sparsecore_gather_two_chunk_w8_bf16,
    xla_gather,
)
import sparsecore_gather_device_trace as dt  # noqa: E402

TRACE_DIR_COLHALF = "/tmp/sparsecore_colhalf_trace"


def pack_colhalf(x: jax.Array) -> jax.Array:
  """(N, D) bf16 -> (N, D/2) int32; word j = bits(x[:, j]) | bits(x[:, j + D/2]) << 16.
  Pure element-wise on two contiguous column halves (no adjacent-column
  pairing, no stack/reshape)."""
  assert x.dtype == jnp.bfloat16, f"expected bfloat16, got {x.dtype}"
  d = x.shape[1]
  assert d % 2 == 0
  half = d // 2
  bits = x.view(jnp.uint16)  # same-width bitcast
  lo = bits[:, :half].astype(jnp.uint32)
  hi = bits[:, half:].astype(jnp.uint32)
  return (lo | (hi << 16)).view(jnp.int32)  # same-width (32->32) reinterpret


def sparsecore_gather_colhalf_bf16(
    x: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    window_size: int = 8,
    core_split: bool = True,
) -> jax.Array:
  """bf16 whole-row SparseCore gather using column-half packing (see module
  docstring). `core_split=True` partitions the (padded) grid over both
  SparseCores x 16 subcores (`core_axis_name=('core','subcore')`); False
  reproduces the old behaviour where each SparseCore runs the whole grid.
  Packing is done on every call (live), inside this function.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  _num_tokens, value_dim = x.shape
  half = value_dim // 2
  num_indices = padded_token_idx.shape[0]
  assert num_indices % window_size == 0

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)
  if core_split:
    quantum = window_size * sc_info.num_cores * sc_info.num_subcores
    pad = (-num_indices) % quantum
  else:
    pad = 0
  kernel_n = num_indices + pad
  safe_idx_k = jnp.pad(safe_idx, (0, pad)) if pad else safe_idx
  core_axes = ('core', 'subcore') if core_split else 'subcore'

  packed = pack_colhalf(x)  # (num_tokens, half) int32, element-wise, on the TensorCore

  @pl.kernel(
      out_type=(
          jax.ShapeDtypeStruct((kernel_n, half), jnp.bfloat16),
          jax.ShapeDtypeStruct((kernel_n, half), jnp.bfloat16),
      ),
      mesh=vector_mesh,
      scratch_types=dict(gather_vmem=pltpu.VMEM((window_size, half), jnp.int32)),
  )
  def kernel(packed_hbm, i_hbm, o_lo_hbm, o_hi_hbm, *, gather_vmem):
    def body(idx_vmem, o_lo_vmem, o_hi_vmem):
      # Gather by TOKEN index directly (1D ref -- the confirmed-working
      # convention): each gathered word holds this row's left-half and
      # right-half bf16 at the same column offset.
      pltpu.sync_copy(packed_hbm.at[idx_vmem], gather_vmem)
      raw = gather_vmem[...].astype(jnp.uint32)
      lo16 = (raw & 0xFFFF).astype(jnp.uint16)
      hi16 = (raw >> 16).astype(jnp.uint16)
      o_lo_vmem[...] = lo16.view(jnp.bfloat16)   # same-width (16->16) bitcast
      o_hi_vmem[...] = hi16.view(jnp.bfloat16)

    pltpu.emit_pipeline(
        body,
        grid=(kernel_n // window_size,),
        in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
        out_specs=[
            pl.BlockSpec((window_size, half), index_map=lambda i: (i, 0)),
            pl.BlockSpec((window_size, half), index_map=lambda i: (i, 0)),
        ],
        core_axis_name=core_axes,
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_lo_hbm, o_hi_hbm)

  lo_out, hi_out = kernel(packed, safe_idx_k)
  if pad:
    lo_out, hi_out = lo_out[:num_indices], hi_out[:num_indices]
  gathered = jnp.concatenate([lo_out, hi_out], axis=-1)  # left half | right half = the full row
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def sparsecore_gather_colhalf_bf16_unpack_outside(
    x: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    window_size: int = 8,
    core_split: bool = True,
) -> jax.Array:
  """CONTROL for "where should the unpack happen": identical to
  `sparsecore_gather_colhalf_bf16` (same column-half packing, W=8, same
  cross-core split + padding, same indices, live packing on every call)
  EXCEPT the SparseCore kernel does ONLY the gather -- it copies packed
  int32 rows straight into its (single) int32 output block, no scratch, no
  unpack -- and the unpack (`& 0xFFFF` / `>> 16`, narrow to uint16,
  same-width view as bf16), the column concatenate and the mask all run
  OUTSIDE the kernel as ordinary element-wise XLA ops on the TensorCore.

  Note the kernel's output layout and write-back also change (one int32
  stream of (kernel_n, D/2) words instead of two bf16 streams of the same
  total bytes), so the time difference vs the unpack-inside variant is not
  purely unpack-instruction time -- it answers a placement question
  (inside vs outside the SparseCore), not an instruction-cost one.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  _num_tokens, value_dim = x.shape
  half = value_dim // 2
  num_indices = padded_token_idx.shape[0]
  assert num_indices % window_size == 0

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)
  if core_split:
    quantum = window_size * sc_info.num_cores * sc_info.num_subcores
    pad = (-num_indices) % quantum
  else:
    pad = 0
  kernel_n = num_indices + pad
  safe_idx_k = jnp.pad(safe_idx, (0, pad)) if pad else safe_idx
  core_axes = ('core', 'subcore') if core_split else 'subcore'

  packed = pack_colhalf(x)

  @pl.kernel(out_type=jax.ShapeDtypeStruct((kernel_n, half), jnp.int32), mesh=vector_mesh)
  def kernel(packed_hbm, i_hbm, o_hbm):
    def body(idx_vmem, o_vmem):
      pltpu.sync_copy(packed_hbm.at[idx_vmem], o_vmem)  # pure gather, no unpack

    pltpu.emit_pipeline(
        body,
        grid=(kernel_n // window_size,),
        in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
        out_specs=[pl.BlockSpec((window_size, half), index_map=lambda i: (i, 0))],
        core_axis_name=core_axes,
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  gathered_packed = kernel(packed, safe_idx_k)
  if pad:
    gathered_packed = gathered_packed[:num_indices]
  bits = gathered_packed.view(jnp.uint32)  # same-width (32->32) reinterpret
  lo = (bits & 0xFFFF).astype(jnp.uint16).view(jnp.bfloat16)
  hi = (bits >> 16).astype(jnp.uint16).view(jnp.bfloat16)
  gathered = jnp.concatenate([lo, hi], axis=-1)
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def _named(name, fn):
  def f(x, idx, mask):
    return fn(x, idx, mask)
  f.__name__ = name
  f.__qualname__ = name
  return f


def run_bf16_pack_split_experiment(seed: int = 0) -> None:
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
      "sc_old_nosplit": functools.partial(
          sparsecore_gather_two_chunk_w8_bf16, window_size=8, repack_every_call=True, core_split=False),
      "sc_old_split": functools.partial(
          sparsecore_gather_two_chunk_w8_bf16, window_size=8, repack_every_call=True, core_split=True),
      "sc_colhalf_nosplit": functools.partial(
          sparsecore_gather_colhalf_bf16, window_size=8, core_split=False),
      "sc_colhalf_split": functools.partial(
          sparsecore_gather_colhalf_bf16, window_size=8, core_split=True),
  }

  # Step 1: which variants compile, and are they exactly (bit-for-bit) correct?
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

  # Step 2: one timing round, all working variants (reference = xla_bf16).
  print("\n" + "=" * 78 + "\nTiming (bf16, real scale): 2x2 format x split + xla\n" + "=" * 78)
  _run_comparison(working, args, label="bf16-format-x-split", bitwise_check=True)

  # Step 3: device trace + attribution for every working variant.
  runs = {n: (jax.jit(_named(n, fn)), x_bf16) for n, fn in working.items()}
  print("\n" + "=" * 78 + "\nDevice trace attribution\n" + "=" * 78)
  dt.capture_trace_runs(runs, padded_token_idx, valid_mask, trace_dir=TRACE_DIR_COLHALF)
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir=TRACE_DIR_COLHALF)


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  if pltpu.get_tpu_info().sparse_core is None:
    print("No SparseCore on this TPU -- cannot run this experiment here.")
    raise SystemExit(1)
  run_bf16_pack_split_experiment()
