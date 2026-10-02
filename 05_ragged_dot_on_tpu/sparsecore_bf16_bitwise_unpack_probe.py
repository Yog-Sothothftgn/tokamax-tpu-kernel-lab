"""Isolated bf16 unpack probe (2026-10-02), per explicit user request:
BEFORE reconnecting to the real gather at all, test ONLY whether a
BITWISE unpack (shift + mask, then a SAME-WIDTH `uint16 -> bfloat16`
`.view()`) works inside a SparseCore kernel -- no gather, no routing
indices, no timing.

Context: `sparsecore_gather_group_b_comparison.py`'s real-hardware bf16
round hit a genuine, confirmed Mosaic limitation -- not an API-usage bug
like the two fixed before it -- when it tried
`gather_vmem[...].view(jnp.bfloat16)` (a WIDTH-CHANGING `int32 ->
bfloat16` bitcast, 32-bit to 16-bit) inside the SparseCore kernel body:

  NotImplementedError: Changing bitwidths not supported.
  (at _bitcast_convert_type_lowering_rule)

This is the FIRST time this whole investigation has actually exercised
the bf16 pack/unpack path on real SparseCore hardware -- every earlier
real-hardware run (A1/A2/A3, Group B round 1) used int32 only.

Hypothesis to test here, nothing else: a SAME-WIDTH bitcast (`uint16 ->
bfloat16`, 16-bit to 16-bit) might be implemented even though the
width-CHANGING one (`int32 -> bfloat16`) is not. If so, extracting each
packed int32 word's low/high 16 bits via ordinary bitwise ops (`&`,
`>>`), narrowing those to `uint16` (a standard integer truncate, not a
bitcast), and ONLY THEN bitcasting `uint16 -> bfloat16` might route
around the unimplemented 32->16 float bitcast entirely.

Test data: 16 known bf16 values -- positive, negative, zero, negative
zero, and simple binary fractions -- all EXACTLY representable in bf16
(powers of two / simple binary fractions), so there is no rounding
ambiguity: the unpacked output must match the originals BIT-FOR-BIT, not
just approximately.

Packing happens OUTSIDE the kernel (already confirmed to work fine --
packing via `.reshape`+`.view` is a plain XLA op, never inside the
SparseCore lowering path that hit the limitation above). Only the
UNPACK, inside the kernel body, is what's actually being tested here.

If this minimal probe passes: reconnect to the real `W=8` whole-row
gather, re-check full output + VMEM usage, and only THEN re-time (per
explicit user instruction -- correctness and compile-ability first,
performance last, don't skip steps).

To run (real v6e VM only):
  python sparsecore_bf16_bitwise_unpack_probe.py
"""

import jax
import jax.numpy as jnp


def bitwise_unpack_probe() -> bool:
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  sc_info = pltpu.get_tpu_info().sparse_core
  assert sc_info is not None, "No SparseCore on this TPU -- cannot run this probe"
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  test_values = jnp.array(
      [
          1.5, -2.25, 0.0, 0.125,
          -1.0, 7.5, -0.5, 100.25,
          -0.0, 0.03125, -128.0, 3.0,
          -3.0, 0.0625, -0.125, 1000.0,
      ],
      dtype=jnp.bfloat16,
  )
  n = test_values.shape[0]
  assert n % 2 == 0
  num_pairs = n // 2

  # Pack OUTSIDE the kernel (plain XLA op, not the thing being tested).
  # NOTE: .view(int32) on a (num_pairs, 2) bf16 array gives (num_pairs, 1)
  # int32 (view halves the LAST dim, not drops it) -- reshape to 1D to
  # match the kernel's 1D BlockSpec below. (First run hit a plain shape-
  # rank mismatch here, "safe_zip() argument 1 has length 2 but argument
  # 0 has length 1" -- a bug in this script, not a new Mosaic limitation.)
  packed = test_values.reshape(num_pairs, 2).view(jnp.int32).reshape(num_pairs)
  print(f"test_values: {test_values}")
  print(f"packed (int32): {packed}")

  @pl.kernel(
      out_type=jax.ShapeDtypeStruct((num_pairs, 2), jnp.bfloat16),
      mesh=vector_mesh,
  )
  def kernel(packed_hbm, o_hbm):
    def body(packed_vmem, o_vmem):
      packed_val = packed_vmem[...].astype(jnp.uint32)  # same-width (32->32) reinterpret, not the restricted case
      low16 = (packed_val & 0xFFFF).astype(jnp.uint16)  # standard integer narrow, not a bitcast
      high16 = (packed_val >> 16).astype(jnp.uint16)    # uint32 >> is a logical (zero-fill) shift
      low_bf16 = low16.view(jnp.bfloat16)   # SAME-WIDTH (16->16) bitcast -- the untested hypothesis
      high_bf16 = high16.view(jnp.bfloat16)
      o_vmem[:, 0] = low_bf16
      o_vmem[:, 1] = high_bf16

    pltpu.emit_pipeline(
        body,
        grid=(1,),  # single step -- no gather, no windowing, this is a pure unpack probe
        in_specs=[pl.BlockSpec((num_pairs,), index_map=lambda i: (0,))],
        out_specs=[pl.BlockSpec((num_pairs, 2), index_map=lambda i: (0, 0))],
        core_axis_name='subcore',
        dimension_semantics=(pltpu.PARALLEL,),
    )(packed_hbm, o_hbm)

  out = jax.jit(kernel)(packed)
  jax.block_until_ready(out)
  unpacked = out.reshape(n)
  print(f"unpacked (bf16): {unpacked}")

  # Bit-for-bit comparison, not approximate -- all test values are
  # exactly representable in bf16, so an exact match is the right bar.
  ok = bool(jnp.array_equal(unpacked, test_values))
  # Also compare raw bits directly (belt-and-suspenders: array_equal on
  # floats would already catch -0.0 == 0.0 as equal, which is CORRECT
  # numerically but would hide a real bit-level unpack bug for that one
  # value -- check the actual bit patterns too).
  bits_match = bool(jnp.array_equal(unpacked.view(jnp.uint16), test_values.view(jnp.uint16)))
  print(f"[bitwise-unpack-probe] value match (array_equal): {ok}")
  print(f"[bitwise-unpack-probe] bit-pattern match (incl. -0.0 vs 0.0): {bits_match}")
  return ok and bits_match


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  from jax.experimental.pallas import tpu as pltpu
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    print("No SparseCore on this TPU -- cannot run this probe here.")
    raise SystemExit(1)
  print(f"sparse_core info: {sc_info}")

  ok = bitwise_unpack_probe()
  print(f"\n{'PASS -- bitwise unpack works inside SparseCore kernel' if ok else 'FAIL -- see above'}")
