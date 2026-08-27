# tokamax TPU kernel-lab

Pallas/Mosaic TPU kernel exercises built around close reading of
[openxla/tokamax](https://github.com/openxla/tokamax), leading up to a
hardware-validated implementation of Kimi K3's LatentMoE layer on TPU v6e via
`tokamax.ragged_dot`.

## Layout

| Directory | Content |
|---|---|
| [`01_pallas_basics/`](01_pallas_basics/) | Core Pallas concepts: `pallas_call`, `Ref`, `grid`, `BlockSpec`, `index_map` -- vector add, a tiled matmul, and a K-dimension-tiled matmul with a VMEM accumulator |
| [`04_own_kernel/`](04_own_kernel/) | A hand-written, V-dimension-tiled online-softmax Pallas kernel, checked against `jax.nn.softmax` |
| [`05_ragged_dot_on_tpu/`](05_ragged_dot_on_tpu/) | `tokamax.ragged_dot` (MoE grouped matmul) benchmark harness across shapes and group-size distributions (`xla` vs `mosaic`), plus a naive JAX reference implementation and a `tokamax.ragged_dot`-based implementation of Kimi K3's LatentMoE layer |
| [`06_kimi_k3_golden_validation/`](06_kimi_k3_golden_validation/) | Golden-test pipeline validating the LatentMoE implementation against the official Kimi K3 PyTorch model, stage by stage |

## Kimi K3 LatentMoE validation

`06_kimi_k3_golden_validation/` locks the official PyTorch source to a pinned
commit (hash-verified), generates golden intermediate/output bundles from it
(fp32 and bf16, at two scales), and cross-checks:

1. The naive JAX reference implementation against the golden bundles --
   matches on all 18 staged intermediates (router, dispatch, per-expert
   compute, combine, RMSNorm, shared expert, final output), both dtypes.
2. `tokamax.ragged_dot`'s `xla`, `mosaic` (v1), and `mosaic_tpu_v2`
   implementations against the same golden bundles, run on real TPU v6e
   hardware.

**Result on real v6e hardware**: `xla` and `mosaic_tpu_v2` match the official
PyTorch model exactly at fp32 (after correcting for TPU's reduced-precision
default matmul behavior), and match closely at bf16 (residual error
consistent with ordinary bf16 rounding, identical between the two
independent kernel implementations). `mosaic` (v1) is correctly skipped
below its 128-row tiling floor rather than silently miscomputing.

## Running

- `01_pallas_basics/` and `04_own_kernel/` run locally under `interpret=True`
  (no TPU needed) to check kernel semantics; real Mosaic/TPU-backend
  correctness and performance require a TPU VM (generation >= 5 for
  `mosaic_tpu`).
- `05_ragged_dot_on_tpu/` and the `xla`/`mosaic`/`mosaic_tpu_v2` checks in
  `06_kimi_k3_golden_validation/` require `jax` + `tokamax` on a real TPU VM.
- Generating golden bundles (`generate_pytorch_golden.py`) requires
  `torch` + `transformers` + `einops` (no TPU needed); it downloads the
  official Kimi K3 source at a pinned commit into
  `06_kimi_k3_golden_validation/official_kimi_k3/` (not redistributed here,
  since the license is not ours to redistribute -- see
  `validate_official_config.py`).

## References

- JAX Pallas TPU tutorial: https://docs.jax.dev/en/latest/pallas/tpu/index.html
- tokamax: https://github.com/openxla/tokamax
- Kimi K3: https://huggingface.co/moonshotai/Kimi-K3
