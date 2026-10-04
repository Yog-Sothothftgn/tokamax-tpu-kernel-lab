"""Experiment C: a Pallas TPU matmul that reads PACKED MXFP4 weights + E8M0
scales from HBM and dequantizes them to BF16 INSIDE the kernel, every call.

Scope (deliberately minimal): one expert, ONE matrix (gate, `w1`), no SiTU, no
up projection, no fusion with anything. Y = X @ W^T where the checkpoint stores
W as (out_features=N, in_features=K) and `X` is (M, K).

Checkpoint format (confirmed in mxfp4_dequant.py, not re-derived here):
  * `weight_packed` (N, K/2) uint8; low nibble = element 2j, high nibble = element
    2j+1 of the K axis; each nibble is FP4 E2M1 (sign bit 3, exponent bits 2:1,
    mantissa bit 0; magnitudes 0, .5, 1, 1.5, 2, 3, 4, 6).
  * `weight_scale` (N, K/32) uint8, E8M0 (biased exponent, value 2^(s-127)), one
    scale per 32 consecutive K elements (= 16 packed columns).
Orientation: gate matmul is `x @ deq(w1).T`, exactly as
own_gateup_situ_pallas_kernel.load_real_expert_gate_up uses it.

How the kernel avoids lane interleaving and lane-wise scale repetition (both
are awkward in Mosaic):
  * K is split into even/odd elements OUTSIDE the kernel on the ACTIVATIONS
    (x_even = x[:, 0::2], x_odd = x[:, 1::2]; an (M, K) elementwise relayout that is
    counted in every timed call). Then
        y = x_even @ W_lo^T + x_odd @ W_hi^T
    with W_lo / W_hi the dequantized low / high nibbles, so the weights never need
    to be interleaved. The summation order differs from a single K-long dot, so the
    f32 results can differ from XLA's by rounding before the final bf16 cast.
  * The per-group scale is expanded to per-packed-column with a tiny matmul by a
    0/1 expansion matrix E (E[g, j] = 1 iff j // 16 == g) built from iotas inside
    the kernel: (N_tile, G) @ (G, K/2). Each output has exactly one non-zero term
    and scales are powers of two, so this is exact in bf16 x bf16 -> f32.
  * Scales are zero-padded from G = K/32 (112 for K = 3584) to a multiple of 128
    columns ONCE, as weight-format preparation outside the timed region
    (`prepare_weights`); the padded groups multiply nothing.
Known, documented numerical difference from `dequantize_mxfp4`: scale byte 0
means 2^-127 there (a float32 denormal) and 0.0 here; scale bytes in real
checkpoints are checked to be >= 1 by the hardware script.

Only the WEIGHTS are read compressed. Nothing outside the kernel materialises a
full bf16 weight matrix; `inspect_compiled` checks the compiled module for that.

Local test (CPU, interpret mode, no TPU):  python mxfp4_online_dequant_matmul_pallas.py local
Hardware session:                          python mxfp4_online_dequant_matmul_pallas.py hw
"""

import functools
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from mxfp4_dequant import dequantize_mxfp4  # noqa: E402

LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072
GROUP = 32
HALF_GROUP = GROUP // 2


def _fp4_value(v):
  """E2M1 nibble (int32 0..15) -> float32, sign applied by setting the float's sign BIT so that FP4
  "-0" (nibble 8) stays -0.0. The first hardware run showed that `jnp.where(sign, -mag, mag)` produced
  +0.0 for exactly those elements (634,681 of 11,010,048, equal to the number of nibbles equal to 8 in
  the checkpoint)."""
  e = (v >> 1) & 3
  m = (v & 1).astype(jnp.float32)
  pow2 = jnp.where(e == 1, 1.0, jnp.where(e == 2, 2.0, 4.0))
  mag = jnp.where(e == 0, 0.5 * m, (1.0 + 0.5 * m) * pow2)
  sign_bit = (v & 8) << 28  # 0x80000000 when negative (int32 wrap-around)
  return jax.lax.bitcast_convert_type(jax.lax.bitcast_convert_type(mag, jnp.int32) | sign_bit, jnp.float32)


def _dequant_tile(p_ref, s_ref):
  """Returns (w_lo, w_hi) bf16 of shape (bn, K/2) for this tile."""
  p = p_ref[...].astype(jnp.int32)
  lo = p & 0xF
  hi = (p >> 4) & 0xF
  s = s_ref[...].astype(jnp.int32)
  scale_f = jax.lax.bitcast_convert_type(s << 23, jnp.float32)  # 2^(s-127) for 1 <= s <= 254
  gp, kh = s.shape[1], p.shape[1]
  g_idx = jax.lax.broadcasted_iota(jnp.int32, (gp, kh), 0)
  j_idx = jax.lax.broadcasted_iota(jnp.int32, (gp, kh), 1)
  expand = (j_idx // HALF_GROUP == g_idx).astype(jnp.bfloat16)
  scale_exp = jnp.dot(scale_f.astype(jnp.bfloat16), expand, preferred_element_type=jnp.float32)
  return ((_fp4_value(lo) * scale_exp).astype(jnp.bfloat16),
          (_fp4_value(hi) * scale_exp).astype(jnp.bfloat16))


def _matmul_kernel(xe_ref, xo_ref, p_ref, s_ref, o_ref):
  w_lo, w_hi = _dequant_tile(p_ref, s_ref)
  dn = (((1,), (1,)), ((), ()))  # contract the last dim of both: x @ w^T
  acc = jax.lax.dot_general(xe_ref[...], w_lo, dn, preferred_element_type=jnp.float32)
  acc += jax.lax.dot_general(xo_ref[...], w_hi, dn, preferred_element_type=jnp.float32)
  o_ref[...] = acc.astype(o_ref.dtype)


def _dequant_only_kernel(p_ref, s_ref, lo_ref, hi_ref):
  lo_ref[...], hi_ref[...] = _dequant_tile(p_ref, s_ref)


def _has_tpu() -> bool:
  return any(d.platform == "tpu" for d in jax.devices())


def prepare_weights(weight_packed: jax.Array, weight_scale: jax.Array):
  """Weight-format preparation, done ONCE outside any timed region: only pads
  the scale's group axis to a multiple of 128 columns. The packed bytes are
  passed through untouched."""
  n, g = weight_scale.shape
  gp = ((g + 127) // 128) * 128
  scale_p = jnp.pad(weight_scale, ((0, 0), (0, gp - g))) if gp != g else weight_scale
  return weight_packed, scale_p


def online_dequant_matmul(x: jax.Array, packed: jax.Array, scale_p: jax.Array, *, bm: int, bn: int) -> jax.Array:
  """x: (M, K) bf16; packed: (N, K/2) uint8; scale_p: (N, Gp) uint8 (from
  `prepare_weights`). Returns (M, N) bf16 = x @ dequant(W)^T."""
  m, k = x.shape
  n, kh = packed.shape
  assert k == 2 * kh and m % bm == 0 and n % bn == 0, (x.shape, packed.shape, bm, bn)
  gp = scale_p.shape[1]
  x_even, x_odd = x[:, 0::2], x[:, 1::2]
  return pl.pallas_call(
      _matmul_kernel,
      grid=(m // bm, n // bn),
      in_specs=[
          pl.BlockSpec((bm, kh), lambda i, j: (i, 0)),
          pl.BlockSpec((bm, kh), lambda i, j: (i, 0)),
          pl.BlockSpec((bn, kh), lambda i, j: (j, 0)),
          pl.BlockSpec((bn, gp), lambda i, j: (j, 0)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      compiler_params=pltpu.CompilerParams(dimension_semantics=("parallel", "parallel")) if _has_tpu() else None,
      interpret=not _has_tpu(),
  )(x_even, x_odd, packed, scale_p)


def dequant_only(packed: jax.Array, scale_p: jax.Array, *, bn: int):
  """Unit-test kernel: returns (W_lo, W_hi), each (N, K/2) bf16, where
  dequantize_mxfp4(...)[:, 0::2] == W_lo and [:, 1::2] == W_hi must hold bitwise."""
  n, kh = packed.shape
  gp = scale_p.shape[1]
  return pl.pallas_call(
      _dequant_only_kernel,
      grid=(n // bn,),
      in_specs=[pl.BlockSpec((bn, kh), lambda j: (j, 0)), pl.BlockSpec((bn, gp), lambda j: (j, 0))],
      out_specs=[pl.BlockSpec((bn, kh), lambda j: (j, 0)), pl.BlockSpec((bn, kh), lambda j: (j, 0))],
      out_shape=[jax.ShapeDtypeStruct((n, kh), jnp.bfloat16), jax.ShapeDtypeStruct((n, kh), jnp.bfloat16)],
      interpret=not _has_tpu(),
  )(packed, scale_p)


def reference_dequant_bf16(packed, scale) -> jax.Array:
  """(N, K) bf16 via the project's verified dequantize_mxfp4 (matches
  compressed-tensors bit for bit), the BF16 weights the pre-dequantized paths use."""
  return dequantize_mxfp4(packed, scale).astype(jnp.bfloat16)


def reference_matmul(x: jax.Array, w_deq_nk: jax.Array) -> jax.Array:
  """XLA matmul on PRE-DEQUANTIZED bf16 weights, W given as (N, K): x @ W^T, bf16 out."""
  return jnp.dot(x, w_deq_nk.T, preferred_element_type=jnp.float32).astype(x.dtype)


def check_scale_padding(scale: jax.Array, scale_p: jax.Array, kh: int) -> dict:
  """Valid range of the zero-padded scale and of the in-kernel expansion matrix:
    * the first G = K/32 columns of the padded scale equal the original scale bytes,
    * every padded column is zero,
    * every packed column j in [0, K/2) maps to exactly one REAL group j // 16 < G
      (E[g, j] = 1 iff j // 16 == g), and no padded group g >= G receives any column,
    * the original scale has no byte 0 (its in-kernel value would differ from the
      reference's 2^-127) and no byte 255 (inf / NaN in E8M0)."""
  g = scale.shape[1]
  gp = scale_p.shape[1]
  sc, sp = np.asarray(scale), np.asarray(scale_p)
  j = np.arange(kh)
  groups = j // HALF_GROUP
  covered = np.bincount(groups, minlength=gp)
  res = {
      "G_real": int(g), "G_padded": int(gp), "packed_cols": int(kh),
      "shape_ok": sp.shape == (sc.shape[0], gp) and gp % 128 == 0 and gp >= g,
      "real_columns_unchanged": bool(np.array_equal(sp[:, :g], sc)),
      "padded_columns_all_zero": bool(not sp[:, g:].any()),
      "all_packed_cols_map_to_real_group": bool(groups.max() < g and covered[:g].min() == HALF_GROUP
                                                 and covered[:g].max() == HALF_GROUP),
      "padded_groups_receive_no_column": bool(covered[g:].sum() == 0),
      "scale_byte_min": int(sc.min()), "scale_byte_max": int(sc.max()),
      "no_zero_or_255_bytes": bool(sc.min() >= 1 and sc.max() <= 254),
  }
  res["ok"] = all(v for k, v in res.items() if isinstance(v, bool))
  return res


def inspect_compiled(fn, args, n: int, k: int) -> dict:
  """Compiled-module check that the call's inputs are the compressed arrays and
  that no full-size bf16/f32 weight matrix exists as an XLA-level array."""
  text = jax.jit(fn).lower(*args).compile().as_text()
  full_shapes = [f"bf16[{n},{k}]", f"bf16[{k},{n}]", f"f32[{n},{k}]", f"f32[{k},{n}]"]
  return {
      "has_u8_packed_operand": f"u8[{n},{k // 2}]" in text,
      "full_weight_shapes_found": [s for s in full_shapes if s in text],
      "n_custom_calls": text.count("custom-call("),
      "hlo_chars": len(text),
  }


# ----------------------------------------------------------------------------
# Local CPU test (interpret mode)
# ----------------------------------------------------------------------------

def _random_mxfp4(key, n: int, k: int):
  k1, k2 = jax.random.split(key)
  packed = jax.random.randint(k1, (n, k // 2), 0, 256, dtype=jnp.int32).astype(jnp.uint8)
  scale = jax.random.randint(k2, (n, k // GROUP), 100, 155, dtype=jnp.int32).astype(jnp.uint8)
  return packed, scale


def local_test() -> bool:
  ok_all = True
  print(f"devices: {jax.devices()} (interpret mode: {not _has_tpu()})")

  # 1. dequant unit test: bit-exact vs the verified dequantizer.
  for n, k, bn in ((64, 512, 32), (128, 1024, 64)):
    packed, scale = _random_mxfp4(jax.random.key(n), n, k)
    _, scale_p = prepare_weights(packed, scale)
    lo, hi = dequant_only(packed, scale_p, bn=bn)
    ref = reference_dequant_bf16(packed, scale)
    ok = bool(jnp.array_equal(lo.view(jnp.uint16), ref[:, 0::2].view(jnp.uint16))) and bool(
        jnp.array_equal(hi.view(jnp.uint16), ref[:, 1::2].view(jnp.uint16)))
    print(f"[dequant bit-exact] n={n} k={k}: {'OK' if ok else 'FAIL'}")
    ok_all &= ok

  # 2. small matmul vs XLA on pre-dequantized weights.
  for m, n, k, bm, bn in ((16, 128, 512, 16, 64), (32, 256, 1024, 32, 128)):
    packed, scale = _random_mxfp4(jax.random.key(m + n), n, k)
    x = (jax.random.normal(jax.random.key(7), (m, k)) * 0.5).astype(jnp.bfloat16)
    _, scale_p = prepare_weights(packed, scale)
    out = online_dequant_matmul(x, packed, scale_p, bm=bm, bn=bn)
    ref = reference_matmul(x, reference_dequant_bf16(packed, scale))
    hp = jnp.dot(x.astype(jnp.float32), dequantize_mxfp4(packed, scale).T)
    d = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
    rel = float(jnp.max(d)) / (float(jnp.std(ref.astype(jnp.float32))) + 1e-12)
    d_hp = float(jnp.max(jnp.abs(out.astype(jnp.float32) - hp))) / (float(jnp.std(hp)) + 1e-12)
    ok = rel < 0.02 and not bool(jnp.any(~jnp.isfinite(out.astype(jnp.float32))))
    print(f"[matmul vs XLA(pre-dequantized bf16)] m={m} n={n} k={k}: max_abs/ref_std={rel:.5f} "
          f"(vs f32 reference {d_hp:.5f}) {'OK' if ok else 'FAIL'}")
    ok_all &= ok

  # 3. one real expert at real dimensions, if the weights are on disk.
  wdir = HERE / "wp_kv6_real_weights" / "layer1_expert0"
  if (wdir / "w1.weight_packed.npy").exists():
    packed = jnp.asarray(np.load(wdir / "w1.weight_packed.npy"))
    scale = jnp.asarray(np.load(wdir / "w1.weight_scale.npy"))
    assert packed.shape == (INTERMEDIATE_SIZE, LATENT_SIZE // 2) and scale.shape == (INTERMEDIATE_SIZE, LATENT_SIZE // GROUP)
    print(f"[real expert 0 w1] scale byte range {int(scale.min())}..{int(scale.max())} (min must be >= 1)")
    ok_all &= int(scale.min()) >= 1
    _, scale_p = prepare_weights(packed, scale)
    x = (jax.random.normal(jax.random.key(3), (16, LATENT_SIZE)) * 0.02).astype(jnp.bfloat16)
    out = online_dequant_matmul(x, packed, scale_p, bm=16, bn=256)
    ref = reference_matmul(x, reference_dequant_bf16(packed, scale))
    d = jnp.abs(out.astype(jnp.float32) - ref.astype(jnp.float32))
    rel = float(jnp.max(d)) / (float(jnp.std(ref.astype(jnp.float32))) + 1e-12)
    ok = rel < 0.02
    print(f"[real expert 0 gate, M=16, interpret] max_abs/ref_std={rel:.5f} {'OK' if ok else 'FAIL'}")
    ok_all &= ok
  else:
    print("[real expert 0] weights not on disk here; skipped")
  print("LOCAL TEST", "PASSED" if ok_all else "FAILED")
  return ok_all


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "local"
  if mode == "local":
    raise SystemExit(0 if local_test() else 1)
  else:
    raise SystemExit("hardware mode lives in real_mxfp4_hw_validation.py (experiment C section)")
