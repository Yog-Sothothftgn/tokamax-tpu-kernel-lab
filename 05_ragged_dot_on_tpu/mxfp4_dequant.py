"""WP-KV6 step 2 (2026-09-15/16): a JAX-native MXFP4 dequantization function
-- no PyTorch/`compressed-tensors` dependency at runtime, so this can run
anywhere this project's other CPU-only pieces run (and, later, on the v6e VM
without needing `compressed-tensors` installed there).

Algorithm confirmed from `compressed-tensors` v0.18.0's OWN source
(`compressors/nvfp4/helpers.py::unpack_fp4_from_uint8`,
`compressors/mx_utils.py::decompress_mx_scale`), not guessed or inferred from
`config.json` alone -- see project memory for the full research trail
(including real HTTP range-fetched tensor shapes from the actual
`moonshotai/Kimi-K3` HF repo, confirming `group_size=32` for the real
checkpoint).

Format: FP4 E2M1, 2 values packed per `uint8` byte (low nibble = first
unpacked value, high nibble = second -- confirmed from
`unpack_fp4_from_uint8`'s `stack((low, high), dim=1).flatten()` order, NOT
high-then-low). Scale is E8M0 (pure power-of-two, no mantissa): one `uint8`
per group of `group_size` consecutive columns (the reduction/K axis, matching
`nn.Linear.weight`'s `(out_features, in_features)` layout -- `in_features` is
the packed/grouped axis, `out_features` is not).
"""

import jax
import jax.numpy as jnp

_FP4_E2M1_TABLE = jnp.array([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=jnp.float32)


def dequantize_mxfp4(weight_packed: jax.Array, weight_scale: jax.Array) -> jax.Array:
  """Dequantizes one MXFP4-packed weight matrix to float32.

  Args:
    weight_packed: `(out_features, in_features // 2)` `uint8`.
    weight_scale: `(out_features, in_features // group_size)` `uint8` (E8M0
      biased exponent), where `group_size` is inferred as
      `in_features / weight_scale.shape[1]`.

  Returns:
    `(out_features, in_features)` `float32`.
  """
  out_features, packed_cols = weight_packed.shape
  in_features = packed_cols * 2
  num_groups = weight_scale.shape[1]
  group_size = in_features // num_groups

  low = weight_packed & 0x0F
  high = (weight_packed >> 4) & 0x0F
  combined = jnp.stack([low, high], axis=-1).reshape(out_features, in_features)

  sign = (combined & 0x08) != 0
  magnitude_idx = combined & 0x07
  values = _FP4_E2M1_TABLE[magnitude_idx] * jnp.where(sign, -1.0, 1.0)

  # E8M0 exponents are exactly representable in float32 (a pure power of
  # two, no mantissa) -- but compressed-tensors' own `decompress_mx_scale`
  # computes `2.0 ** exponent.to(bfloat16)`, i.e. deliberately (or as an
  # implementation side effect) rounds the exponent to bfloat16 BEFORE the
  # power, discarding precision that didn't need to be lost. Matched here
  # bit-for-bit (confirmed via `check_dequantize_mxfp4_matches_compressed_tensors`)
  # rather than using a more precise float32 computation, since a real
  # downstream consumer of this checkpoint format uses exactly this
  # (slightly lossy) library behavior as its own ground truth.
  scale_exponent_bf16 = (weight_scale.astype(jnp.int32) - 127).astype(jnp.bfloat16)
  scale_float = (2.0 ** scale_exponent_bf16).astype(jnp.float32)
  scale_expanded = jnp.repeat(scale_float, group_size, axis=1)

  return values * scale_expanded


def check_dequantize_mxfp4_matches_compressed_tensors(seed: int = 0) -> bool:
  """Verifies `dequantize_mxfp4` bit-for-bit against `compressed-tensors`'
  own real torch implementation, on synthetic random packed data -- this is
  the correctness gate for `dequantize_mxfp4` before it's ever used on a
  real downloaded checkpoint shard. No download needed: constructs valid
  random nibble/scale bytes directly (not derived from any real weight).

  Requires `torch` and `compressed_tensors` (a one-time `pip install
  compressed-tensors` in whatever environment runs this, no TPU/tokamax
  needed) -- NOT a runtime dependency of `dequantize_mxfp4` itself.
  """
  import torch  # noqa: PLC0415 -- optional, only needed for this cross-check
  from compressed_tensors.compressors.mx_utils import decompress_mx_scale  # noqa: PLC0415
  from compressed_tensors.compressors.nvfp4.helpers import unpack_fp4_from_uint8  # noqa: PLC0415

  out_features, in_features, group_size = 8, 64, 32
  num_groups = in_features // group_size
  rng = jax.random.key(seed)
  key_packed, key_scale = jax.random.split(rng)

  weight_packed = jax.random.randint(
      key_packed, (out_features, in_features // 2), 0, 256, dtype=jnp.uint32
  ).astype(jnp.uint8)
  weight_scale = jax.random.randint(
      key_scale, (out_features, num_groups), 100, 155, dtype=jnp.uint32
  ).astype(jnp.uint8)

  jax_out = dequantize_mxfp4(weight_packed, weight_scale)

  packed_t = torch.from_numpy(jax.device_get(weight_packed).copy())
  scale_t = torch.from_numpy(jax.device_get(weight_scale).copy())
  unpacked_t = unpack_fp4_from_uint8(packed_t, out_features, in_features, dtype=torch.float32)
  scale_float_t = decompress_mx_scale(scale_t).to(torch.float32)
  scale_expanded_t = scale_float_t.repeat_interleave(group_size, dim=1)
  torch_out = (unpacked_t * scale_expanded_t).numpy()

  max_abs_diff = float(jnp.max(jnp.abs(jax_out - jnp.asarray(torch_out))))
  ok = max_abs_diff == 0.0
  print(
      f"[mxfp4-dequant-check] max_abs_diff={max_abs_diff} "
      f"{'OK (exact match)' if ok else 'FAIL -- dequantize_mxfp4 diverges from compressed-tensors'}"
  )
  return ok


if __name__ == "__main__":
  ok = check_dequantize_mxfp4_matches_compressed_tensors()
  raise SystemExit(0 if ok else 1)
