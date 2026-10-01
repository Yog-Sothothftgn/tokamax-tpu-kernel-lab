"""Group A diagnostic (2026-09-30): WHY does `gather_window_size=8` fail to
compile, specifically -- isolating one variable at a time, per explicit
user request, BEFORE any further performance comparison (Group B is a
separate, later, not-yet-written file). This file makes no speed claim at
all; it only answers "does this compile, is it correct if so, and if not,
what exactly does the compiler say".

Context: `sparsecore_gather_prototype.py` found that a single-chunk
SparseCore gather at `value_dim=128` compiles and is correct at
`gather_window_size=128` (the guide's own literal value) but fails at
smaller windows (32, 8) with `'sc_tpu.enqueue_transfer' op Not
implemented: Source and target leading tiles have different trailing
dimensions`. That finding conflated several things that could each matter
on their own: the window size itself, the 2D `(1, W)` index-array shape
convention used there (vs. the guide's OWN bf16-packed example, which uses
a flat 1D `(W,)` convention instead), and whether the failure is in the
basic index-array pipelining or specifically in the indirect gather. This
file isolates each of those.

**A1**: vary ONLY the window size `W` (128, 64, 32, 16, 8), holding
`chunk_width=value_dim=128` fixed (deliberately narrow -- no column
chunking at all here, so VMEM capacity cannot be a confound; that's
already characterized separately in `sparsecore_gather_prototype.py`).
Compile+correctness only, full error text captured and printed for every
failure (not a truncated last line).

**A2**: for `W=8` (expected to fail) and `W=128` (known-working control),
decompose into sub-steps, changing exactly ONE thing per test:
  1. Index-copy-only: pipeline ONLY the index array through VMEM and write
     it straight back out -- no `x_hbm`, no indirect addressing at all.
     Isolates "can `emit_pipeline`/`BlockSpec` even move a `(., W)`-shaped
     int32 array at this window size" from "does the GATHER itself work".
  2. The actual gather (adds indirect addressing on top of step 1).
  3. If step 1 fails: redo step 1 with the OTHER index-shape convention
     (1D `(W,)` instead of 2D `(1, W)`, or vice versa) -- isolates shape
     convention from window size specifically for the plain index copy.
     (This file also redoes step 2 under both conventions, for the same
     reason, as an extra cross-check beyond the minimum requested.)

Fixed test data throughout: a 512x128 int32 table (`jnp.arange`, every
element distinct) and a 256-entry index array built to deliberately
include BOTH out-of-order and duplicate entries (forced, not left to
chance via plain `jax.random.randint`) -- so a real indexing bug cannot
hide behind "the random indices happened to already be sorted/unique".
256 is evenly divisible by every window size tested.

To run (real v6e VM only -- SparseCore ops have no CPU interpret-mode
stub, same as sparsecore_gather_prototype.py). Run with
`JAX_TRACEBACK_FILTERING=off` and tee to a log file -- `traceback.
format_exception` only re-prints whatever frames are already on the raised
exception, and JAX filters its own internal frames out of that by default;
without disabling that filter, the "full error" this file prints may still
be missing the actual internal frame that explains the failure:

  JAX_TRACEBACK_FILTERING=off \\
  python3 -u sparsecore_gather_window_size_diagnosis.py \\
  2>&1 | tee sparsecore_window_diagnosis.log
"""

import traceback

import jax
import jax.numpy as jnp

BATCH_SIZE = 512
VALUE_DIM = 128
NUM_INDICES = 256


def _make_test_data(seed: int = 0):
  """Distinct-valued int32 table + an index array with GUARANTEED
  out-of-order and duplicate entries (not left to chance)."""
  x = jnp.arange(BATCH_SIZE * VALUE_DIM, dtype=jnp.int32).reshape(BATCH_SIZE, VALUE_DIM)
  key = jax.random.key(seed)
  indices = jax.random.randint(key, (NUM_INDICES,), 0, BATCH_SIZE, jnp.int32)
  indices = indices.at[1].set(indices[0])  # forced duplicate, adjacent
  indices = indices.at[-1].set(indices[0])  # forced duplicate, far away
  mid = NUM_INDICES // 2
  # An explicit, unambiguous descending run -- reversing whatever random
  # values happened to already be there is NOT the same as forcing
  # out-of-order (per review: a reversed random slice is still just
  # "some order", not provably non-monotonic to a reader checking this by
  # hand). 16 strictly descending small integers, valid row indices into
  # the 512-row table either way.
  indices = indices.at[mid - 8:mid + 8].set(jnp.arange(15, -1, -1, dtype=jnp.int32))
  return x, indices


def _full_error(e: Exception) -> str:
  return "".join(traceback.format_exception(type(e), e, e.__traceback__))


def _describe_blocks(window_size: int, indices_2d: bool, *, is_echo: bool) -> tuple[tuple, tuple]:
  """Prints the exact BlockSpec shapes a given call uses, so that two
  failures with similar-looking error TEXT can be checked against which
  tiled memref they actually name, per review: `index_copy_only`'s INPUT
  transfer uses the identical indices BlockSpec shape as `gather_once`'s
  input transfer (same `(1, W)` or `(W,)` indices tile either way) -- but
  `index_copy_only`'s OUTPUT transfer reuses that SAME small indices-shaped
  block, while `gather_once`'s OUTPUT transfer is a completely different,
  much wider `(W, VALUE_DIM)` block. Two errors that look identical as
  strings are only good evidence of a SHARED cause if the memref shapes
  they name match this breakdown (e.g. both naming the `(1, W)`-or-`(W,)`
  INPUT indices tile) -- if one names the input tile and the other names a
  `(W, VALUE_DIM)` output tile, they are NOT the same failure and should
  not be treated as confirming each other.
  """
  in_shape = (1, window_size) if indices_2d else (window_size,)
  out_shape = in_shape if is_echo else (window_size, VALUE_DIM)
  kind = "echo (index-copy-only)" if is_echo else "gather"
  print(f"    [{kind}] input(indices) block shape={in_shape}, output block shape={out_shape}")
  return in_shape, out_shape


def _vector_mesh():
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc
  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this diagnostic"
  return plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")


def gather_once(
    x: jax.Array,
    indices: jax.Array,
    window_size: int,
    indices_2d: bool = True,
    materialize_index: bool = False,
) -> jax.Array:
  """A2 step 2 (and A1's only test): the actual indirect gather, at
  `chunk_width=VALUE_DIM` (the full row -- table is only 128 wide here by
  construction). `indices_2d` selects between the 2D `(1, W)` convention
  `sparsecore_gather_prototype.py` used (reshape indices, `.at[0]` inside
  the kernel) and the guide's OWN bf16-packed example's flat 1D `(W,)`
  convention.

  **`materialize_index` default is `False` (pass a VMEM REF to
  `sync_copy`, e.g. `i_vmem.at[0]` / `i_vmem` -- matching the guide's own
  int32 gather example) for BOTH shape conventions** -- an earlier version
  of this file used `i_vmem.at[0]` (a ref) for the 2D case but `i_vmem[...]`
  (a MATERIALIZED array read) for the 1D case, which meant the "1D vs 2D"
  comparison was actually changing TWO things at once (shape AND whether a
  ref or a read-out array gets passed to `sync_copy`) -- caught by review
  before this file was ever run: a 1D failure couldn't have been
  attributed to shape alone. Fixed so both conventions pass a ref by
  default; `materialize_index=True` preserves the original "read the array
  out first" behavior as its own SEPARATE, clearly-labeled experiment
  (see `run_a2`'s extra cross-check), not mixed into the main shape
  comparison.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  vector_mesh = _vector_mesh()
  num_indices = indices.shape[0]
  assert num_indices % window_size == 0

  if indices_2d:
    indices_in = indices.reshape((1, num_indices))
    in_block = pl.BlockSpec((1, window_size), index_map=lambda i: (0, i))
  else:
    indices_in = indices
    in_block = pl.BlockSpec((window_size,), index_map=lambda i: (i,))

  @pl.kernel(out_type=jax.ShapeDtypeStruct((num_indices, VALUE_DIM), x.dtype), mesh=vector_mesh)
  def kernel(x_hbm, i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      if indices_2d:
        idx = i_vmem[0] if materialize_index else i_vmem.at[0]
      else:
        idx = i_vmem[...] if materialize_index else i_vmem
      pltpu.sync_copy(x_hbm.at[idx], o_vmem)

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // window_size,),
        in_specs=[in_block],
        out_specs=[pl.BlockSpec((window_size, VALUE_DIM), index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  return jax.jit(kernel)(x, indices_in)


def index_copy_only(indices: jax.Array, window_size: int, indices_2d: bool = True) -> jax.Array:
  """A2 step 1: pipelines ONLY the index array through VMEM and writes it
  straight back out -- no `x_hbm`, no `sync_copy`, no indirect addressing
  at all.

  **Interpretation caveat, per review -- do not over-read a failure or
  success here in isolation**: this echo round-trips index -> HBM -> VMEM
  -> HBM again, which is a DIFFERENT (and in one respect more complex, in
  one respect simpler) path than the gather: it adds a VMEM-to-HBM WRITE
  of the index array that the gather never does (the gather's indices are
  read-only, used only to address `x_hbm`), while never touching `x_hbm`
  at all.
    - echo SUCCEEDS: the basic index HBM<->VMEM round-trip works at this
      window size/shape -- informative, but does NOT by itself prove the
      gather's indirect addressing will also work.
    - echo FAILS: the failure could be in the read-in, the VMEM op, or the
      NEW write-out path the gather doesn't have -- do NOT conclude "the
      gather's index input is broken" from this alone; check which op/pass
      the error names.
    - echo AND gather fail with the SAME underlying op/error: THAT is the
      strong signal of a shared input-layout problem, not either failure
      alone.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu

  vector_mesh = _vector_mesh()
  num_indices = indices.shape[0]
  assert num_indices % window_size == 0

  if indices_2d:
    indices_in = indices.reshape((1, num_indices))
    block = pl.BlockSpec((1, window_size), index_map=lambda i: (0, i))
    out_type = jax.ShapeDtypeStruct((1, num_indices), indices.dtype)
  else:
    indices_in = indices
    block = pl.BlockSpec((window_size,), index_map=lambda i: (i,))
    out_type = jax.ShapeDtypeStruct((num_indices,), indices.dtype)

  @pl.kernel(out_type=out_type, mesh=vector_mesh)
  def kernel(i_hbm, o_hbm):
    def body(i_vmem, o_vmem):
      o_vmem[...] = i_vmem[...]

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // window_size,),
        in_specs=[block],
        out_specs=[block],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  return jax.jit(kernel)(indices_in)


def run_a1() -> list[tuple[int, bool, bool | None, str | None]]:
  print("=" * 78)
  print("A1: vary ONLY window size W (chunk_width=128 fixed, full row, no column chunking)")
  print("=" * 78)
  x, indices = _make_test_data()
  expected = jnp.take(x, indices, axis=0)
  results = []
  for w in (128, 64, 32, 16, 8):
    print(f"[A1] W={w:4d}:")
    _describe_blocks(w, indices_2d=True, is_echo=False)
    try:
      out = gather_once(x, indices, w, indices_2d=True)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out, expected))
      print(f"[A1] W={w:4d}: COMPILED, correct={correct}")
      results.append((w, True, correct, None))
    except Exception as e:  # noqa: BLE001 -- capturing the real, full error is the point
      err = _full_error(e)
      print(f"[A1] W={w:4d}: COMPILE/RUN FAILED -- full error below\n{err}")
      results.append((w, False, None, err))
  print("\n[A1 summary]")
  for w, compiled, correct, _err in results:
    print(f"  W={w:4d}: compiled={compiled} correct={correct}")
  return results


def run_a2() -> None:
  print("\n" + "=" * 78)
  print("A2: decompose W=8 (expected failing) vs W=128 (known-working control)")
  print("=" * 78)
  x, indices = _make_test_data()
  expected = jnp.take(x, indices, axis=0)

  for w in (128, 8):
    print(f"\n--- W={w} ---")

    # Step 1: index-copy-only, 2D (1, W) convention (matches
    # sparsecore_gather_prototype.py's own convention) -- the ONLY
    # variable relative to the known-working W=128 control is W itself.
    _describe_blocks(w, indices_2d=True, is_echo=True)
    try:
      out = index_copy_only(indices, w, indices_2d=True)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out.reshape(-1), indices))
      print(f"[A2 step1: index-copy-only, 2D (1,{w})] COMPILED, correct={correct}")
      # A correctness FAILURE is still a failure for this check's purpose
      # -- per review, "compiled but produced wrong data" must not count
      # as "step 1 OK" (an earlier version of this file set this to True
      # unconditionally whenever the call didn't raise, regardless of
      # `correct`).
      step1_2d_ok = correct
    except Exception as e:  # noqa: BLE001
      print(f"[A2 step1: index-copy-only, 2D (1,{w})] FAILED -- full error below\n{_full_error(e)}")
      step1_2d_ok = False

    # Step 3 (only because step 1 is being checked either way here, for
    # BOTH w values, to see whether shape convention matters even at the
    # KNOWN-WORKING w=128): same index-copy-only test, but the flat 1D
    # (W,) convention instead -- the ONE thing that changes relative to
    # the step-1 attempt just above.
    _describe_blocks(w, indices_2d=False, is_echo=True)
    try:
      out = index_copy_only(indices, w, indices_2d=False)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out, indices))
      print(f"[A2 step3: index-copy-only, 1D ({w},)] COMPILED, correct={correct}")
    except Exception as e:  # noqa: BLE001
      print(f"[A2 step3: index-copy-only, 1D ({w},)] FAILED -- full error below\n{_full_error(e)}")

    if not step1_2d_ok:
      print(f"[A2 note] step 1 (2D) failed at W={w} -- step 3's 1D result above is the "
            f"direct comparison this case needs.")

    # Step 2: the actual gather (indirect addressing), 2D convention --
    # interpret this relative to step 1's outcome (did it fail at the SAME
    # stage as the plain index copy, or only here, once gather is added).
    _describe_blocks(w, indices_2d=True, is_echo=False)
    try:
      out = gather_once(x, indices, w, indices_2d=True)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out, expected))
      print(f"[A2 step2: gather, 2D (1,{w})] COMPILED, correct={correct}")
    except Exception as e:  # noqa: BLE001
      print(f"[A2 step2: gather, 2D (1,{w})] FAILED -- full error below\n{_full_error(e)}")

    # Extra cross-check beyond the minimum requested: the gather under the
    # 1D convention too, for the same reason step 3 exists for step 1 --
    # isolates shape convention from window size for the GATHER
    # specifically, not just the plain index copy.
    _describe_blocks(w, indices_2d=False, is_echo=False)
    try:
      out = gather_once(x, indices, w, indices_2d=False)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out, expected))
      print(f"[A2 extra: gather, 1D ({w},)] COMPILED, correct={correct}")
    except Exception as e:  # noqa: BLE001
      print(f"[A2 extra: gather, 1D ({w},)] FAILED -- full error below\n{_full_error(e)}")

    # Separate, explicitly-labeled experiment (per review: keep this OUT
    # of the main shape comparison above) -- the ORIGINAL "read the index
    # out as a materialized array" behavior, now isolated as its own
    # single-variable test (materialize_index=True vs False, 1D shape held
    # fixed) rather than conflated with the 1D-vs-2D comparison.
    try:
      out = gather_once(x, indices, w, indices_2d=False, materialize_index=True)
      jax.block_until_ready(out)
      correct = bool(jnp.array_equal(out, expected))
      print(f"[A2 separate experiment: gather, 1D ({w},), materialized index array] "
            f"COMPILED, correct={correct}")
    except Exception as e:  # noqa: BLE001
      print(f"[A2 separate experiment: gather, 1D ({w},), materialized index array] "
            f"FAILED -- full error below\n{_full_error(e)}")


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  from jax.experimental.pallas import tpu as pltpu
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    print("No SparseCore on this TPU -- cannot run this diagnostic here.")
    raise SystemExit(1)
  print(f"sparse_core info: {sc_info}")

  run_a1()
  run_a2()

  print(
      "\n" + "=" * 78 +
      "\nREADING THIS LOG: if two failures show textually similar error "
      "messages, do not treat that alone as 'same root cause'. Each test "
      "above printed its own input(indices)/output block shapes right "
      "before running -- check that BOTH errors' messages name the SAME "
      "one of those (e.g. both naming the small (1,W)-or-(W,) INPUT "
      "indices tile) before concluding they share a cause. An echo "
      "failure naming its (1,W)-shaped OUTPUT tile and a gather failure "
      "naming its (1,W)-shaped INPUT tile would LOOK similar but are "
      "actually about different transfers -- only an input-vs-input or "
      "output-vs-output match (with gather's much wider (W,VALUE_DIM) "
      "output block correctly distinguished from the indices-shaped "
      "block) is real evidence of a shared cause.\n" + "=" * 78
  )
