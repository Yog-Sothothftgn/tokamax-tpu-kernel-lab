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

### Result on real v6e hardware (`test_ragged_dot_against_pytorch_golden.py`)

All 8 (bundle x dtype x implementation) combinations tested against golden
bundles generated from the official model (`--bundle-set both --variant both
--implementation all`):

| Bundle | dtype | Implementation | Result |
|---|---|---|---|
| small (64/32/48) | fp32 | xla | ALL STAGES MATCH |
| small (64/32/48) | bf16 | xla | ALL STAGES MATCH (bit-exact) |
| mosaic (256/128/128) | fp32 | xla | ALL STAGES MATCH |
| mosaic (256/128/128) | fp32 | mosaic (v1) | SKIPPED -- dispatch rows `M=40 < 128` tiling floor |
| mosaic (256/128/128) | fp32 | mosaic_tpu_v2 | ALL STAGES MATCH |
| mosaic (256/128/128) | bf16 | xla | `expert_up_output` 1.95e-3 vs. 1e-3 tolerance (all other 12 stages OK) |
| mosaic (256/128/128) | bf16 | mosaic (v1) | SKIPPED -- same tiling floor reason |
| mosaic (256/128/128) | bf16 | mosaic_tpu_v2 | Same 1.95e-3 residual as xla above |

fp32 max abs diffs against the official model ranged from bit-exact to
~2.15e-6 across all 18 staged intermediates -- reaching this required
wrapping the whole computation in `jax.default_matmul_precision("highest")`,
since TPU's default matmul precision (`precision=None`) silently uses a
reduced-precision path for float32 inputs on the MXU (invisible on CPU,
confirmed via the compiled HLO). The bf16 `expert_up_output` residual is
assessed as ordinary bf16 rounding noise, not a logic bug: `xla` and
`mosaic_tpu_v2` -- two independent kernel implementations -- produce the
*identical* residual, and the mathematically equivalent fp32 run is exact.
`mosaic` (v1) is correctly skipped below its 128-row tiling floor rather than
silently miscomputing; `NotImplementedError`/skip is never counted as a pass.

Not yet done: WP-KV5 (full-dimension smoke test) and WP-KV6 (real,
MXFP4-quantized checkpoint validation); a bundle with `num_tokens >= 64` to
actually exercise Mosaic v1 end-to-end; digging the bf16 residual into the
operator level.

## Requirements

- **Pallas exercises** (`01_pallas_basics/`, `04_own_kernel/`): `jax` only.
  Run locally under `interpret=True` (no TPU needed) to check kernel
  semantics; real Mosaic/TPU-backend correctness and performance require a
  TPU VM (generation >= 5 for the `mosaic_tpu` backend).
- **`ragged_dot` benchmarking and validation** (`05_ragged_dot_on_tpu/`, and
  the `xla`/`mosaic`/`mosaic_tpu_v2` checks in
  `06_kimi_k3_golden_validation/`): `jax` (>= 0.11.0) +
  [`tokamax`](https://github.com/openxla/tokamax) (built from source, no
  pinned release) on a real TPU VM.
- **Golden bundle generation** (`generate_pytorch_golden.py`, no TPU needed):
  `torch` + `transformers` + `einops`. The bundles checked into this repo
  were generated with `torch==2.13.0+cpu` and `transformers==5.15.1` (see
  each bundle's `metadata.json` for the exact versions and the official
  source's pinned commit + SHA256). Generating a bundle downloads the
  official Kimi K3 source at that pinned commit into
  `06_kimi_k3_golden_validation/official_kimi_k3/` (not redistributed here,
  since its license is not ours to redistribute -- see
  `validate_official_config.py`).

## References

- JAX Pallas TPU tutorial: https://docs.jax.dev/en/latest/pallas/tpu/index.html
- tokamax: https://github.com/openxla/tokamax
- Kimi K3: https://huggingface.co/moonshotai/Kimi-K3
