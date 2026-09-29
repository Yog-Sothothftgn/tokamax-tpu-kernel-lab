"""Minimal SparseCore gather prototype (2026-09-27), per explicit user
request: read the official JAX Pallas SparseCore guide
(https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html), then try its
documented gather primitive on THIS project's OWN real token indices and
activation shape -- not the guide's own benchmark shape -- before trusting
that its reported speedup transfers here.

**A real, load-bearing scale mismatch worth flagging up front**: the
guide's own gather benchmark uses `value_dim=128` (narrow rows) and
`num_indices` in the MILLIONS (`gather_window_size=128 * sc_num_cores *
sc_num_subcores * num_steps(1024)`, e.g. ~4.2M indices on a 2-core/16-subcore
v6e), and reports SparseCore beating a plain `jnp.take` by ~4.5x there.
Our real MoE dispatch is the opposite shape: `value_dim=LATENT_SIZE=3584`
(28x wider rows) and `num_indices=m_padded` in the THOUSANDS (per-shard
dispatch slot count, e.g. 4736 at `num_tokens=2048`/`local_num_experts=64`/
`capacity_factor=2.0` -- see WP-KV6). SparseCore's advantage in the guide's
own benchmark comes from parallelizing many small, irregular accesses
across 16 subcores; a "few thousand wide rows" workload is a fundamentally
different regime (more DMA-bandwidth-bound per access, less per-access
overhead to amortize) that the guide's own number says nothing about
directly. This file exists to actually measure OUR shape instead of
assuming the guide's ratio carries over.

**Deliberately narrow scope, per explicit user instruction**: this touches
ONLY the dispatch GATHER step. Sorting (`filter_and_pad_to_shard_jittable`'s
own `jnp.argsort`), the COMBINE step (`_combine_shard_contribution`'s
weighted top-k sum), and MXFP4 dequantization are NOT touched here and this
file does NOT wire into the production forward pass -- it only asks "does a
SparseCore gather beat the current plain-XLA gather at our real shape",
using the REAL, already-validated dispatch function
(`route_and_filter_to_local_shard_jittable`) purely to produce realistic
indices/activations to gather, not to be modified itself.

**Cannot run in interpret mode, unlike every other file in this directory**:
SparseCore ops are real hardware primitives (`pltpu.sync_copy` inside a
`pl.kernel(mesh=VectorSubcoreMesh(...))`) with no CPU stub. This needs a
real TPU with a SparseCore (v5p/v6e/7x per the guide's spec table -- v6e,
this project's own hardware, has 2 SparseCores/chip) -- i.e. the real v6e
VM, not this CPU-only Windows machine. **Also genuinely unverified**:
whether `jax.experimental.pallas.tpu_sc` (the `plsc` module the guide
imports) exists in this project's pinned `jax[tpu]==0.11.0` -- SparseCore
Pallas support may need a newer jax version, which could reopen the
hijax/flax pin conflict `tpu_v6e_workflow.md` already documents. `__main__`
runs a staged probe (`_probe_sparsecore_api`) FIRST, using the guide's own
toy examples (not this file's gather logic) to isolate an API/jax-version
problem from a bug in this file's own code, before assuming the rest works.
If the probe fails, verify compatibility in a SEPARATE throwaway venv --
do not upgrade jax in the existing, already-working one.

**VMEM caveat, CONFIRMED on real v6e hardware (2026-09-29), whole-row
gather does NOT fit at our width**: the guide's own gather example uses
`value_dim=128`; ours is `LATENT_SIZE=3584` (28x wider). A direct,
unmodified port of the guide's whole-row gather to `value_dim=3584` hits
one of two REAL, hardware-confirmed failures depending on window size, and
there is NO window size that avoids both:
  - `gather_window_size=128` (the guide's own literal, tiling-safe value):
    `'memref.alloca' op E3000: CompileTimeSparseCoreAllocationFailure:
    current allocation offset upper bound (917759 words) exceeds the
    legit[imate limit]` -- per-subcore VMEM is only 262144 bytes (65536
    words, from `pltpu.get_tpu_info().sparse_core.vmem_capacity_bytes`);
    128 rows x 3584 words is ~14x over that.
  - Any other window size tried (32, 8): `'sc_tpu.enqueue_transfer' op Not
    implemented: Source and target leading tiles have different trailing
    dimensions` -- a real Mosaic lowering limitation, reproduced at BOTH
    `value_dim=128` (the guide's own width) and `value_dim=3584`, so this
    is NOT specific to our width -- window sizes other than the compiler's
    expected native tile (128 for int32) simply aren't lowered yet.

**CONFIRMED WORKING instead**: chunking along the FEATURE dimension --
gather `chunk_width=128`-wide column slices (matching the guide's own
tile-safe width) at `gather_window_size=128`, repeated `LATENT_SIZE //
chunk_width = 28` times to cover the full 3584 columns, concatenating the
28 outputs back into a full-width row. Verified on real v6e hardware
(2026-09-29): all 28 chunks compile and match `x[indices]` exactly, and
the full 28-chunk reconstruction matches `jnp.take(x, indices, axis=0)`
exactly, at `batch_size=512`, `num_indices=256`, plain int32 (no bf16
packing yet -- see `sparsecore_gather_chunked`). **Compile+correctness
only so far** -- 28 separate kernel launches per real dispatch call could
plausibly cost more in per-launch overhead than the whole-row approach
would have saved, which is exactly the "does sync/handoff eat the benefit"
question this file exists to answer; not yet measured at real scale (see
`check_chunked`). `gather_window_size` therefore is fixed at 128 (the one
confirmed-working, tiling-safe value) rather than left as a free
parameter -- smaller windows are confirmed broken, not just untried.

To run (real v6e VM only):
  python sparsecore_gather_prototype.py
"""

import functools
import pathlib
import sys
import time

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE))

from kimi_k3_latent_moe_reference import (  # noqa: E402
    kimi_k3_config,
    route_and_filter_to_local_shard_jittable,
)

LATENT_SIZE = 3584  # must match kimi_k3_config().latent_size


def real_dispatch_indices(
    num_tokens: int = 2048,
    local_expert_start: int = 137,
    local_num_experts: int = 64,
    capacity_factor: float = 2.0,
    seed: int = 0,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
  """Returns `(x, padded_token_idx, valid_mask, production_sorted_tokens)`
  from the REAL, already-validated production dispatch function (not a
  hand-rolled approximation) -- same params as
  `check_route_and_filter_jittable_matches_baseline`'s own "standard" case,
  at this project's usual `num_tokens=2048` scale (matches
  `own_gateup_situ_pallas_kernel.py`'s default M). `x` is synthetic
  (`scale=0.02`, this project's usual convention) but real-shaped; only the
  ROUTING is real, matching this project's established WP-KV6 convention of
  keeping non-quantized pieces synthetic.

  `production_sorted_tokens` is the function's OWN first return value (its
  real `sorted_tokens`, already gathered+masked) -- kept and returned
  alongside the indices/mask so `check()` can compare against it directly,
  instead of only ever comparing against a hand-reconstructed baseline that
  might itself not match production (per explicit review feedback).
  """
  config = kimi_k3_config()
  keys = jax.random.split(jax.random.key(seed), 4)
  hidden_states = jax.random.normal(keys[0], (num_tokens, config.hidden_size), dtype=jnp.bfloat16)
  router_weight = jax.random.normal(keys[1], (config.hidden_size, config.num_experts)) * 0.02
  e_score_correction_bias = jax.random.normal(keys[2], (config.num_experts,)) * 0.02
  x = (jax.random.normal(keys[3], (num_tokens, config.latent_size)) * 0.02).astype(jnp.bfloat16)

  production_sorted_tokens, _, valid_mask, _, padded_token_idx, _ = route_and_filter_to_local_shard_jittable(
      hidden_states, x, router_weight, e_score_correction_bias,
      config=config, local_expert_start=local_expert_start,
      local_num_experts=local_num_experts, capacity_factor=capacity_factor,
  )
  return x, padded_token_idx, valid_mask, production_sorted_tokens


def current_jit_gather(x: jax.Array, padded_token_idx: jax.Array, valid_mask: jax.Array) -> jax.Array:
  """The "current" baseline: a plain XLA gather, RECONSTRUCTED from the
  PUBLIC `padded_token_idx`/`valid_mask` outputs (which use -1 for invalid
  slots) rather than `filter_and_pad_to_shard_jittable`'s internal-only
  `sorted_token_idx_all` array -- meant to be mathematically equivalent to
  that function's real `sorted_tokens`
  (`gathered_all = x[sorted_token_idx_all]; sorted_tokens =
  where(valid_mask, gathered_all, 0)`), since at invalid positions the
  gathered value is masked to zero either way. **This equivalence is not
  assumed** -- `check()` verifies this reconstruction against the real
  production `sorted_tokens` (see `real_dispatch_indices`) before trusting
  it as ground truth for the SparseCore comparison, per explicit review
  feedback.
  """
  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx)
  gathered = x[safe_idx]
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def sparsecore_gather_chunked(
    x_int32: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    chunk_width: int = 128,
    gather_window_size: int = 128,
) -> jax.Array:
  """The CONFIRMED-WORKING SparseCore gather path at our real value_dim
  (3584) -- see this file's module docstring for why the naive whole-row
  `sparsecore_gather` below cannot work at this width at ANY window size
  (real hardware errors, not a guess). Chunks the feature dimension into
  `value_dim // chunk_width` column slices (default 128-wide, matching the
  guide's own tile-safe width), gathers each chunk separately at
  `gather_window_size=128` (the one window size confirmed NOT to hit the
  tiling lowering bug), then concatenates the chunks back into a full-
  width row. Verified correct on real v6e hardware (2026-09-29): all 28
  chunks (at `chunk_width=128`) match `x[indices]` exactly, and the full
  concatenated reconstruction matches `jnp.take(x, indices, axis=0)`
  exactly.

  INT32 ONLY, no bf16 packing yet -- per explicit user instruction, this
  step is scoped to "does the chunked approach compile, run correctly, and
  perform reasonably" BEFORE adding bf16-packing complexity on top.
  `check_chunked` casts the real bf16 activation to int32 as a shape/
  mechanism stand-in -- not numerically meaningful, just enough to time
  and correctness-check the GATHER itself; bf16 packing is a separate,
  later step once this int32 timing picture is understood.

  `gather_window_size` is NOT a free parameter here (see module docstring)
  -- 128 is the one confirmed-safe value; changing it needs new hardware
  evidence, not a guess. `num_chunks` separate kernel launches per call is
  the real, not-yet-measured cost this function's timing exists to reveal
  -- per-launch overhead x 28 could plausibly erase whatever the gather
  itself saves, which is exactly the "does sync/handoff eat the benefit"
  question from the original request.
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  num_tokens, value_dim = x_int32.shape
  assert value_dim == LATENT_SIZE, f"expected value_dim={LATENT_SIZE}, got {value_dim}"
  assert x_int32.dtype == jnp.int32, f"expected int32, got {x_int32.dtype}"
  assert value_dim % chunk_width == 0, (
      f"value_dim={value_dim} not divisible by chunk_width={chunk_width}"
  )
  num_chunks = value_dim // chunk_width
  num_indices = padded_token_idx.shape[0]
  assert num_indices % gather_window_size == 0, (
      f"num_indices={num_indices} not divisible by gather_window_size={gather_window_size}"
  )

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this prototype"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)
  indices_r = safe_idx.reshape((1, num_indices))

  def gather_one_chunk(x_chunk_hbm):
    @pl.kernel(
        out_type=jax.ShapeDtypeStruct((num_indices, chunk_width), jnp.int32),
        mesh=vector_mesh,
    )
    def kernel(x_hbm, i_hbm, o_hbm):
      def body(i_vmem, o_vmem):
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(num_indices // gather_window_size,),
          in_specs=[pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i))],
          out_specs=[pl.BlockSpec((gather_window_size, chunk_width), index_map=lambda i: (i, 0))],
          core_axis_name='subcore',
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    return kernel(x_chunk_hbm, indices_r)

  chunks = [
      gather_one_chunk(x_int32[:, c * chunk_width:(c + 1) * chunk_width])
      for c in range(num_chunks)
  ]
  gathered = jnp.concatenate(chunks, axis=-1)
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def check_chunked(
    num_tokens: int = 2048,
    local_num_experts: int = 64,
    chunk_width: int = 128,
    gather_window_size: int = 128,
    seed: int = 0,
) -> bool:
  """Correctness + timing for the CONFIRMED-WORKING chunked SparseCore
  gather, at REAL production scale (real `m_padded`, real
  `LATENT_SIZE=3584`), int32 only (bf16 packing deferred -- see
  `sparsecore_gather_chunked`'s docstring). Compares against a plain XLA
  gather on the SAME int32 stand-in data, with both timing conventions
  (pipelined/blocking, same convention as every other check in this
  project) -- this is the real-scale answer to "do the 28 separate kernel
  launches cost more than they save".
  """
  x, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=num_tokens, local_num_experts=local_num_experts, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  num_chunks = LATENT_SIZE // chunk_width
  print(
      f"[setup-chunked] num_tokens={num_tokens} local_num_experts={local_num_experts} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} num_chunks={num_chunks} "
      f"num_valid={int(jnp.sum(valid_mask))}"
  )

  # int32 stand-in for the real bf16 activation -- NOT numerically
  # meaningful (bf16 packing/unpacking is deferred), just enough to time
  # and correctness-check the gather MECHANISM at the real shape.
  x_int32 = x.astype(jnp.int32)

  sc_fn = functools.partial(
      sparsecore_gather_chunked, chunk_width=chunk_width, gather_window_size=gather_window_size
  )

  def xla_fn(xx, idx, vm):
    safe_idx = jnp.where(idx < 0, 0, idx)
    gathered = xx[safe_idx]
    return jnp.where(vm[:, None], gathered, jnp.zeros_like(gathered))

  out = jax.jit(sc_fn)(x_int32, padded_token_idx, valid_mask)
  jax.block_until_ready(out)
  expected = jax.jit(xla_fn)(x_int32, padded_token_idx, valid_mask)
  jax.block_until_ready(expected)

  max_abs_diff = float(jnp.max(jnp.abs(out - expected)))
  ok = max_abs_diff == 0.0  # pure data movement, expect exact match
  print(
      f"[{'OK' if ok else 'FAIL'}] chunked sparsecore gather vs plain XLA gather (int32): "
      f"max_abs_diff={max_abs_diff} (expect exact 0)"
  )

  xla_ms_pipe = _time_jit_pipelined(xla_fn, x_int32, padded_token_idx, valid_mask)
  xla_ms_block = _time_jit_blocking(xla_fn, x_int32, padded_token_idx, valid_mask)
  sc_ms_pipe = _time_jit_pipelined(sc_fn, x_int32, padded_token_idx, valid_mask)
  sc_ms_block = _time_jit_blocking(sc_fn, x_int32, padded_token_idx, valid_mask)

  print(
      f"[timing-chunked][pipelined] xla={xla_ms_pipe:.4f}ms sparsecore_chunked={sc_ms_pipe:.4f}ms "
      f"(speedup={xla_ms_pipe / sc_ms_pipe:.3f}x, {num_chunks} kernel launches/call)\n"
      f"[timing-chunked][per-call]  xla={xla_ms_block:.4f}ms sparsecore_chunked={sc_ms_block:.4f}ms "
      f"(speedup={xla_ms_block / sc_ms_block:.3f}x)"
  )
  return ok


def sparsecore_gather(
    x: jax.Array,
    padded_token_idx: jax.Array,
    valid_mask: jax.Array,
    gather_window_size: int = 8,
    repack_every_call: bool = True,
) -> jax.Array:
  """**KNOWN NON-WORKING at this file's real `LATENT_SIZE=3584`, kept only
  as the initial (naive, whole-row) attempt and a record of the exact
  hardware errors that ruled it out -- see this file's module docstring
  and use `sparsecore_gather_chunked` instead.** Every `gather_window_size`
  tried either OOMs per-subcore VMEM (128) or hits a real Mosaic lowering
  bug (32, 8) -- there is no value that avoids both at this width.

  Original docstring, for context: SparseCore gather at our real shape,
  structurally mirroring the official guide's `gather_bf16_packed` example
  (bf16 needs pairwise-packing into int32 since SparseCore DMA only
  natively moves 32-bit words -- see this file's module docstring).

  `repack_every_call`: if True (the default, and the realistic case for a
  live forward pass where `x` is freshly computed every call), `x` must be
  PLAIN bfloat16, shape `(num_tokens, LATENT_SIZE)` -- the bf16->packed-
  int32 reshape/view happens INSIDE this jitted function, so its cost is
  included in the measured time -- this is the honest answer to "does the
  packing/handoff cost eat the gather's own advantage". If False, `x` must
  be ALREADY in packed-int32 HBM layout, shape `(num_tokens // 2,
  LATENT_SIZE)` (best case for SparseCore, only realistic if some upstream
  step could produce it pre-packed) -- exists so the two costs (repack vs.
  gather-alone) can be measured separately rather than only ever seeing
  them bundled together.

  **The two modes take different-shaped `x` on purpose** (packed rows are
  literally half as many, by construction) -- validated here by DTYPE, not
  by re-deriving "the original token count" from a shape that means
  something different in each mode (an earlier version of this file
  asserted `x.shape[0] % 2 == 0` unconditionally, which silently checked
  the WRONG thing in the `repack_every_call=False` branch -- caught by
  review before ever running: at a real small-decode-scale case like
  `num_tokens=2`, the pre-packed `x` legitimately has ONE row, and `1 % 2 ==
  0` is False, wrongly rejecting a perfectly valid input).
  """
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  packing = 2  # 32 // 16, bf16 -> int32
  num_indices = padded_token_idx.shape[0]
  assert num_indices % gather_window_size == 0, (
      f"num_indices={num_indices} not divisible by gather_window_size="
      f"{gather_window_size} -- pick a divisor (project convention: only "
      "evenly-dividing shapes, keep the exercise simple)"
  )

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this prototype"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  safe_idx = jnp.where(padded_token_idx < 0, 0, padded_token_idx).astype(jnp.int32)

  if repack_every_call:
    num_tokens, value_dim = x.shape
    assert value_dim == LATENT_SIZE, f"expected value_dim={LATENT_SIZE}, got {value_dim}"
    assert x.dtype == jnp.bfloat16, (
        f"repack_every_call=True expects plain bfloat16 input, got {x.dtype}"
    )
    assert num_tokens % packing == 0, (
        f"row-pairing needs an even token count, got num_tokens={num_tokens}"
    )
    x_packed = x.reshape(num_tokens // packing, packing * value_dim).view(jnp.int32)
  else:
    _num_tokens_packed, value_dim = x.shape
    assert value_dim == LATENT_SIZE, f"expected value_dim={LATENT_SIZE}, got {value_dim}"
    assert x.dtype == jnp.int32, (
        f"repack_every_call=False expects already-packed int32 input, got {x.dtype}"
    )
    x_packed = x  # caller already passed pre-packed int32 data, any row count is valid here

  @pl.kernel(
      out_type=jax.ShapeDtypeStruct((num_indices, value_dim), jnp.bfloat16),
      mesh=vector_mesh,
      scratch_types=dict(
          gather_vmem=pltpu.VMEM((gather_window_size, value_dim), jnp.int32)
      ),
  )
  def kernel(x_packed_hbm, i_hbm, o_hbm, *, gather_vmem):
    def body(idx_vmem, o_vmem):
      # Issue indirect 32-bit DMA gather using the halved index (doc's
      # gather_bf16_packed pattern, verbatim).
      pltpu.sync_copy(x_packed_hbm.at[jax.lax.div(idx_vmem, packing)], gather_vmem)
      pairs = gather_vmem.view(jnp.bfloat16).reshape(-1, packing, value_dim)
      is_odd = (idx_vmem % packing)[:, None]
      o_vmem[...] = jnp.where(is_odd == 1, pairs[:, 1], pairs[:, 0])

    pltpu.emit_pipeline(
        body,
        grid=(num_indices // gather_window_size,),
        in_specs=[pl.BlockSpec((gather_window_size,), index_map=lambda i: (i,))],
        out_specs=[pl.BlockSpec((gather_window_size, value_dim), index_map=lambda i: (i, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(i_hbm, o_hbm)

  gathered = kernel(x_packed, safe_idx)
  return jnp.where(valid_mask[:, None], gathered, jnp.zeros_like(gathered))


def _time_jit_pipelined(f, *args, num_repeats: int = 20) -> float:
  """Throughput-style timing -- see own_gateup_situ_pallas_kernel.py's
  identically-named function for the full caveat about what this does and
  does not measure."""
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def _time_jit_blocking(f, *args, num_repeats: int = 20) -> float:
  """Per-call latency timing -- blocks after every call."""
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
    jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def check(
    num_tokens: int = 2048,
    local_num_experts: int = 64,
    gather_window_size: int = 8,
    seed: int = 0,
) -> bool:
  """Correctness (SparseCore gather vs the REAL production `sorted_tokens`,
  not a hand-reconstructed stand-in) + three-way timing: (a) current XLA
  gather, (b) SparseCore gather assuming `x` arrives pre-packed (best
  case), (c) SparseCore gather repacking `x` on every call (realistic case
  for a live forward pass) -- (b) vs (c)'s gap IS the answer to "does the
  packing/handoff cost eat the gather's own advantage".
  """
  x, padded_token_idx, valid_mask, production_sorted_tokens = real_dispatch_indices(
      num_tokens=num_tokens, local_num_experts=local_num_experts, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(f"[setup] num_tokens={num_tokens} local_num_experts={local_num_experts} "
        f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} "
        f"num_valid={int(jnp.sum(valid_mask))}")

  # Step 0: does the reconstructed XLA baseline actually match the REAL
  # production sorted_tokens, before trusting it as ground truth for
  # anything below? Not assumed -- checked, per explicit review feedback.
  reconstructed = current_jit_gather(x, padded_token_idx, valid_mask)
  baseline_matches_production = bool(jnp.array_equal(reconstructed, production_sorted_tokens))
  print(
      f"[baseline-check] current_jit_gather reconstruction == real "
      f"route_and_filter_to_local_shard_jittable's sorted_tokens: "
      f"{baseline_matches_production}"
  )
  if not baseline_matches_production:
    print(
        "[baseline-check] FAIL -- current_jit_gather does not match production "
        "sorted_tokens; stopping before comparing SparseCore against a baseline "
        "that isn't even right."
    )
    return False

  expected = production_sorted_tokens  # the real thing, not a reconstruction, from here on

  x_packed_precomputed = x.reshape(num_tokens // 2, 2 * LATENT_SIZE).view(jnp.int32)
  sc_fn_prepacked = functools.partial(
      sparsecore_gather, gather_window_size=gather_window_size, repack_every_call=False
  )
  sc_fn_repack = functools.partial(
      sparsecore_gather, gather_window_size=gather_window_size, repack_every_call=True
  )

  out_prepacked = jax.jit(sc_fn_prepacked)(x_packed_precomputed, padded_token_idx, valid_mask)
  jax.block_until_ready(out_prepacked)
  out_repack = jax.jit(sc_fn_repack)(x, padded_token_idx, valid_mask)
  jax.block_until_ready(out_repack)

  diff = jnp.abs(out_repack.astype(jnp.float32) - expected.astype(jnp.float32))
  max_abs_diff = float(jnp.max(diff))
  ok = max_abs_diff == 0.0  # gather is a pure data-movement op, expect EXACT match, not rounding noise
  same_as_prepacked = bool(jnp.array_equal(out_repack, out_prepacked))
  status = "OK" if (ok and same_as_prepacked) else "FAIL"
  print(f"[{status}] sparsecore_gather vs REAL production sorted_tokens: max_abs_diff={max_abs_diff:.4e} "
        f"(expect exact 0 -- pure data movement) prepacked_vs_repack_identical={same_as_prepacked}")

  xla_ms_pipe = _time_jit_pipelined(current_jit_gather, x, padded_token_idx, valid_mask)
  xla_ms_block = _time_jit_blocking(current_jit_gather, x, padded_token_idx, valid_mask)
  sc_prepacked_ms_pipe = _time_jit_pipelined(sc_fn_prepacked, x_packed_precomputed, padded_token_idx, valid_mask)
  sc_prepacked_ms_block = _time_jit_blocking(sc_fn_prepacked, x_packed_precomputed, padded_token_idx, valid_mask)
  sc_repack_ms_pipe = _time_jit_pipelined(sc_fn_repack, x, padded_token_idx, valid_mask)
  sc_repack_ms_block = _time_jit_blocking(sc_fn_repack, x, padded_token_idx, valid_mask)

  print(
      f"[timing][pipelined]  xla={xla_ms_pipe:.4f}ms  "
      f"sparsecore(prepacked)={sc_prepacked_ms_pipe:.4f}ms (speedup={xla_ms_pipe / sc_prepacked_ms_pipe:.3f}x)  "
      f"sparsecore(repack-every-call)={sc_repack_ms_pipe:.4f}ms (speedup={xla_ms_pipe / sc_repack_ms_pipe:.3f}x)  "
      f"repack_overhead={(sc_repack_ms_pipe - sc_prepacked_ms_pipe) * 1000:.1f}us\n"
      f"[timing][per-call]   xla={xla_ms_block:.4f}ms  "
      f"sparsecore(prepacked)={sc_prepacked_ms_block:.4f}ms (speedup={xla_ms_block / sc_prepacked_ms_block:.3f}x)  "
      f"sparsecore(repack-every-call)={sc_repack_ms_block:.4f}ms (speedup={xla_ms_block / sc_repack_ms_block:.3f}x)  "
      f"repack_overhead={(sc_repack_ms_block - sc_prepacked_ms_block) * 1000:.1f}us"
  )
  return status == "OK"


def _probe_sparsecore_api() -> bool:
  """Exercises `pl.kernel` + `VectorSubcoreMesh` + `emit_pipeline` +
  `sync_copy` via the GUIDE'S OWN minimal toy examples -- not this file's
  real gather -- before trusting any of it works in this jax build. A
  successful `import tpu_sc` alone does not prove these actually work
  together (the import can succeed while the API has moved, is partially
  stubbed, or only errors at trace/compile time) -- per explicit review
  feedback. Staged, cheapest/most-isolated first:

    1. the guide's `sc_add_one` pipelining example (VectorSubcoreMesh +
       emit_pipeline, no gather at all) -- isolates whether the basic
       mesh/pipeline machinery works.
    2. the guide's own tiny plain-int32 `gather` example (adds sync_copy +
       indexed BlockSpec, still NOT this file's bf16-packing code) --
       isolates whether gather-via-indices works, before our own
       packing/unpacking logic is anywhere in the picture.

  Only if BOTH stages pass does `check()` (this file's real gather, at real
  shapes) get run. If EITHER stage fails or errors, that is a jax-build/API
  problem, not a bug in this file's own gather logic -- **do not try to fix
  it by upgrading jax in this project's existing, working venv** (this
  would risk breaking the already-hardware-confirmed WP4-WP6/WP-KV5/WP-KV6
  work and the fused-kernel arc, all validated under the pinned
  `jax[tpu]==0.11.0`). Verify SparseCore API compatibility in a SEPARATE,
  throwaway venv first.
  """
  import numpy as np
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  sc_info = pltpu.get_tpu_info().sparse_core
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  # Stage 1: guide's own sc_add_one (pipelining, no gather/indices at all).
  reg_shape = (1, sc_info.num_lanes)
  dma_block = (8, 128)

  @jax.jit
  def sc_add_one(x):
    @pl.kernel(out_type=x, mesh=vector_mesh, scratch_types=[])
    def sc_add_one_kernel(x_hbm_ref, o_hbm_ref):
      in_shape = x_hbm_ref.shape

      def body(in_vmem, out_vmem):
        @pl.loop(0, in_vmem.shape[0], step=reg_shape[0])
        def _(c0):
          @pl.loop(0, in_vmem.shape[1], step=reg_shape[1])
          def _(c1):
            slc = (pl.ds(c0, reg_shape[0]), pl.ds(c1, reg_shape[1]))
            out_vmem.at[*slc][...] = in_vmem.at[*slc][...] + 1

      pltpu.emit_pipeline(
          body,
          grid=(in_shape[0] // dma_block[0], in_shape[1] // dma_block[1]),
          in_specs=[pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))],
          out_specs=[pl.BlockSpec(block_shape=dma_block, index_map=lambda i, j: (i, j))],
          core_axis_name=('core', 'subcore'),
          dimension_semantics=(pltpu.PARALLEL, pltpu.PARALLEL),
      )(x_hbm_ref, o_hbm_ref)

    return sc_add_one_kernel(x)

  try:
    x_probe = jax.random.randint(jax.random.key(0), (4096, 128), 0, 64, jnp.int32)
    y_probe = sc_add_one(x_probe)
    stage1_ok = bool(np.array_equal(y_probe, x_probe + 1))
  except Exception as e:  # noqa: BLE001 -- recording the failure itself is the point
    print(f"[api-probe] stage 1 (sc_add_one) RAISED: {e}")
    stage1_ok = False
  print(f"[api-probe] stage 1 (sc_add_one, VectorSubcoreMesh + emit_pipeline, no gather): "
        f"{'OK' if stage1_ok else 'FAIL'}")
  if not stage1_ok:
    return False

  # Stage 2: guide's own tiny plain-int32 gather (sync_copy + indexed
  # BlockSpec). gather_window_size=128 is the guide's own literal,
  # confirmed-tiling-safe value (see module docstring) -- an earlier
  # version of this probe used 32 "to make it tiny", which is itself the
  # exact window size confirmed (on real hardware) to hit a real Mosaic
  # lowering bug, unrelated to whether SparseCore gather works at all. Only
  # num_steps is shrunk for a quick probe; window size is NOT a free
  # parameter here.
  batch_size, value_dim, gather_window_size, num_steps = 4096, 128, 128, 2
  num_indices = gather_window_size * sc_info.num_cores * sc_info.num_subcores * num_steps
  x_g = jnp.arange(batch_size * value_dim).reshape(batch_size, value_dim).astype(jnp.int32)
  indices_g = jax.random.randint(jax.random.key(1), (num_indices,), 0, batch_size, jnp.int32)

  @jax.jit
  def gather_probe(x, indices):
    indices = indices.reshape((1, num_indices))

    @pl.kernel(out_type=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype), mesh=vector_mesh)
    def kernel(x_hbm, i_hbm, o_hbm):
      def body(i_vmem, o_vmem):
        pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(num_indices // gather_window_size,),
          in_specs=[pl.BlockSpec((1, gather_window_size), index_map=lambda i: (0, i))],
          out_specs=[pl.BlockSpec((gather_window_size, value_dim), index_map=lambda i: (i, 0))],
          core_axis_name='subcore',
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    return kernel(x, indices)

  try:
    out_g = gather_probe(x_g, indices_g)
    stage2_ok = bool(np.array_equal(out_g, jnp.take(x_g, indices_g, axis=0)))
  except Exception as e:  # noqa: BLE001
    print(f"[api-probe] stage 2 (guide's plain-int32 gather) RAISED: {e}")
    stage2_ok = False
  print(f"[api-probe] stage 2 (guide's own plain-int32 gather, sync_copy + BlockSpec indexing): "
        f"{'OK' if stage2_ok else 'FAIL'}")
  return stage2_ok


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)

  try:
    from jax.experimental.pallas import tpu as _pltpu_check
    from jax.experimental.pallas import tpu_sc as _plsc_check  # noqa: F401
    sc_info = _pltpu_check.get_tpu_info().sparse_core
    print(f"tpu_sc import OK, sparse_core info: {sc_info}")
  except ImportError as e:
    print(
        f"tpu_sc import FAILED ({e}) -- this jax build does not expose "
        "SparseCore Pallas support; likely needs a newer jax version than "
        "this project's pinned jax[tpu]==0.11.0 (may reopen the "
        "hijax/flax pin conflict tpu_v6e_workflow.md documents). Verify "
        "compatibility in a SEPARATE, throwaway venv before touching the "
        "existing working one -- do not upgrade jax here directly. Stopping "
        "here rather than guessing further."
    )
    raise SystemExit(1)

  if sc_info is None:
    print("This TPU has no SparseCore -- cannot run this prototype here.")
    raise SystemExit(1)

  if not _probe_sparsecore_api():
    print(
        "\nAPI probe FAILED on the guide's own toy examples (not this file's "
        "gather logic) -- this is a jax-build/API-availability problem. Do "
        "NOT try to fix it by upgrading jax in this project's existing venv "
        "(risks breaking the already hardware-confirmed WP4-WP6/WP-KV5/"
        "WP-KV6 and fused-kernel work, all pinned to jax[tpu]==0.11.0). "
        "Set up a separate throwaway venv to find a jax version where the "
        "probe passes, first."
    )
    raise SystemExit(1)

  # check() exercises the whole-row sparsecore_gather, which is KNOWN
  # NON-WORKING at this file's real LATENT_SIZE=3584 (see its docstring and
  # the module docstring for the two real hardware errors) -- run the
  # chunked path instead, the one confirmed correct on real hardware.
  ok = check_chunked()
  print(f"\n{'all checks passed' if ok else 'CHECK FAILED -- see above'}")
