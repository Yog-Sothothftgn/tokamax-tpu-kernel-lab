"""Cross-SparseCore work split for the int32 whole-row W=8 gather
(2026-10-02), a single-variable experiment per explicit user request.

Motivation (device trace, `sparsecore_gather_device_trace.py`): the old
`sparsecore_gather_whole_row_w8` partitions its 592-step grid only over the
`subcore` axis (`emit_pipeline(core_axis_name='subcore')`), so EACH of the
two SparseCores executes the ENTIRE grid -- the trace showed
`ep_run_kernel` summed over one SparseCore's 16 TECs = 592.0 (the whole
grid), i.e. 1184 steps per call across both cores for 592 steps of useful
work. This file changes ONLY that: the two SparseCores together execute
the grid once, each owning a different, disjoint range of output rows.
Everything else (int32, W=8, whole row, 1D ref indices, sync_copy gather,
mask outside the kernel) is identical to the old kernel.

Two split implementations are provided, so the API behaviour itself is
tested rather than assumed:
  - `explicit_core_halves`: core c owns output rows
    [c*rows, (c+1)*rows) (a disjoint HBM-ref slice of the index input and
    of the output, selected with `jax.lax.axis_index('core')`), and the
    16 subcores of that core split ITS half via the usual
    `core_axis_name='subcore'`. Disjointness is structural.
  - `tuple_axes`: one flat grid with `core_axis_name=('core','subcore')`,
    leaving the partitioning to `emit_pipeline`.

**A side effect that is reported, not hidden**: the old grid (4736 rows /
W=8 = 592 steps) is not divisible by 2 cores x 16 subcores = 32 workers
(592/32 = 18.5), and a 2-D "split per core, then per subcore" needs
rows-per-core divisible by 16 x W. The index array is therefore padded
with zeros (valid, in-bounds gathers) up to a multiple of W x 32 = 256
(4736 -> 4864, +2.7% rows) and the output is sliced back to the real
length. The new kernel does 2.7% extra work relative to the old one;
that slightly UNDERSTATES the split's benefit.

Three things are verified (per the request):
  1. Output is exactly correct (vs plain XLA gather, and vs the old kernel).
  2. The two cores' work does not overlap: from a device trace, the
     `ep_run_kernel` count summed over BOTH SparseCores per call must be
     ~(padded steps) = 608, NOT the old 1184 (and per-SparseCore ~304,
     not 592).
  3. In one measurement round, speed vs the old kernel and vs XLA
     (same methodology as the rest of this investigation: correctness
     first, 10 rounds x 20 calls, rotating order, pipelined + per-call,
     explicit jit arguments).

To run (real v6e VM only):
  python sparsecore_gather_core_split.py
"""

import functools
import pathlib
import sys

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sparsecore_gather_prototype import LATENT_SIZE, real_dispatch_indices  # noqa: E402
from sparsecore_gather_group_b_comparison import (  # noqa: E402
    NUM_TOKENS,
    LOCAL_NUM_EXPERTS,
    _run_comparison,
    sparsecore_gather_whole_row_w8,
    xla_gather,
)
import sparsecore_gather_device_trace as dt  # noqa: E402

SPLIT_MODES = ("explicit_core_halves", "tuple_axes")
TRACE_DIR_SPLIT = "/tmp/sparsecore_core_split_trace"


def sparsecore_gather_whole_row_w8_core_split(
    x_int32: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    window_size: int = 8,
    split_mode: str = "explicit_core_halves",
) -> jax.Array:
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  assert split_mode in SPLIT_MODES, f"unknown split_mode {split_mode!r}"
  _num_tokens, value_dim = x_int32.shape
  assert x_int32.dtype == jnp.int32, f"expected int32, got {x_int32.dtype}"
  num_indices = padded_token_idx.shape[0]

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU"
  num_cores, num_subcores = sc_info.num_cores, sc_info.num_subcores
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  # Pad the index array so (rows / window) is divisible by num_cores *
  # num_subcores (see module docstring -- a reported +2.7% extra work).
  quantum = window_size * num_cores * num_subcores
  pad = (-num_indices) % quantum
  padded_n = num_indices + pad
  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)
  safe_idx_padded = jnp.pad(safe_idx, (0, pad))  # zeros: valid, in-bounds gather indices

  @pl.kernel(out_type=jax.ShapeDtypeStruct((padded_n, value_dim), jnp.int32), mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      pltpu.sync_copy(x_hbm.at[i_vmem], o_vmem)  # 1D ref -- the confirmed-working convention

    if split_mode == "explicit_core_halves":
      rows_per_core = padded_n // num_cores
      core = jax.lax.axis_index("core")
      i_part = i_hbm.at[pl.ds(core * rows_per_core, rows_per_core)]
      o_part = o_hbm.at[pl.ds(core * rows_per_core, rows_per_core)]
      pltpu.emit_pipeline(
          body,
          grid=(rows_per_core // window_size,),
          in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
          out_specs=[pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0))],
          core_axis_name='subcore',
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_part, o_part)
    else:  # "tuple_axes"
      pltpu.emit_pipeline(
          body,
          grid=(padded_n // window_size,),
          in_specs=[pl.BlockSpec((window_size,), index_map=lambda i: (i,))],
          out_specs=[pl.BlockSpec((window_size, value_dim), index_map=lambda i: (i, 0))],
          core_axis_name=('core', 'subcore'),
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

  gathered = kernel(x_int32, safe_idx_padded)[:num_indices]
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def _named(name, fn):
  """Wrap `fn` as a function whose __name__ is `name`, so its XLA module is
  `jit_<name>(...)` and distinct in the trace."""
  def f(x, idx, mask):
    return fn(x, idx, mask)
  f.__name__ = name
  f.__qualname__ = name
  return f


def run_core_split_experiment(seed: int = 0) -> None:
  import traceback

  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")
  _x_bf16, padded_token_idx, valid_mask, _ = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} num_valid={int(jnp.sum(valid_mask))}"
  )
  # Non-degenerate int32 data (jnp.arange, distinct per element).
  x_int32 = jnp.arange(NUM_TOKENS * LATENT_SIZE, dtype=jnp.int32).reshape(NUM_TOKENS, LATENT_SIZE)
  args = (x_int32, padded_token_idx, valid_mask)

  expected = jax.jit(xla_gather)(*args)
  old_out = jax.jit(functools.partial(sparsecore_gather_whole_row_w8, window_size=8))(*args)
  jax.block_until_ready((expected, old_out))
  print(f"[sanity] old whole-row kernel == xla: {bool(jnp.array_equal(old_out, expected))}")

  # Step 1: does each split mode compile, and is it exactly correct?
  working = {}
  for mode in SPLIT_MODES:
    fn = functools.partial(sparsecore_gather_whole_row_w8_core_split, window_size=8, split_mode=mode)
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
    except Exception as e:  # noqa: BLE001 -- surfacing the real error is the point
      print(f"\n[split mode {mode}] FAILED to compile/run -- full error below:\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
      continue
    ok_xla = bool(jnp.array_equal(out, expected))
    ok_old = bool(jnp.array_equal(out, old_out))
    print(f"[split mode {mode}] COMPILED; exact match vs xla={ok_xla}, vs old kernel={ok_old}")
    if ok_xla and ok_old:
      working[mode] = fn

  if not working:
    print("\nNo split mode both compiled and matched -- stopping before timing/trace.")
    return

  # Step 2: same-round timing, old vs new vs XLA (int32).
  impls = {"xla_int32": xla_gather, "sc_old_both_cores_full_grid": functools.partial(
      sparsecore_gather_whole_row_w8, window_size=8)}
  for mode, fn in working.items():
    impls[f"sc_split_{mode}"] = fn
  print("\n" + "=" * 78 + "\nTiming (int32, real scale): xla vs old vs split\n" + "=" * 78)
  _run_comparison(impls, args, label="int32-core-split")

  # Step 3: device trace -> verify the two cores' work does not overlap.
  runs = {"xla_int32_gather": (jax.jit(_named("xla_int32_gather", xla_gather)), x_int32),
          "sc_old_full_grid": (jax.jit(_named(
              "sc_old_full_grid", functools.partial(sparsecore_gather_whole_row_w8, window_size=8))), x_int32)}
  for mode, fn in working.items():
    nm = f"sc_split_{mode}"
    runs[nm] = (jax.jit(_named(nm, fn)), x_int32)
  print("\n" + "=" * 78 + "\nDevice trace: per-SparseCore grid coverage\n" + "=" * 78)
  dt.capture_trace_runs(runs, padded_token_idx, valid_mask, trace_dir=TRACE_DIR_SPLIT)
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir=TRACE_DIR_SPLIT)

  n_steps_old = num_indices // 8
  pad_quantum = 8 * 2 * 16
  n_steps_new = (num_indices + (-num_indices) % pad_quantum) // 8
  print(
      f"\nEXPECTED (for reading the trace above): old kernel grid={n_steps_old} steps -> "
      f"ep_run_kernel per SparseCore={n_steps_old}, per call across both SparseCores={2 * n_steps_old} "
      f"(both cores run everything). Split kernel padded grid={n_steps_new} steps -> per SparseCore="
      f"{n_steps_new // 2}, per call across both={n_steps_new} (disjoint halves, no duplication)."
  )


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  if pltpu.get_tpu_info().sparse_core is None:
    print("No SparseCore on this TPU -- cannot run this experiment here.")
    raise SystemExit(1)
  run_core_split_experiment()
