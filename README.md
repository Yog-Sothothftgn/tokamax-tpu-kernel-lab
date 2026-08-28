# tokamax TPU kernel-lab

Pallas/Mosaic TPU kernel exercises built around close reading of
[openxla/tokamax](https://github.com/openxla/tokamax), leading up to a
hardware-validated, reduced-scale prototype of Kimi K3's LatentMoE layer on
TPU v6e using `tokamax.ragged_dot`.

## Current status

**Completed:**
- Official Kimi K3 source/config snapshot pinned and hash-verified.
- Golden bundles generated from the real official PyTorch model (fp32 + bf16,
  random weight init, reduced-scale configs -- see caveats below).
- Naive JAX-reference implementation validated against those bundles,
  fp32 and bf16, all 18 staged intermediates.
- `xla` and `mosaic_tpu_v2` (`tokamax.ragged_dot`) validated against the
  same bundles on real TPU v6e hardware, fp32.

**In progress / open:**
- bf16 on `xla`/`mosaic_tpu_v2` shows a tolerance-exceeding residual on 4 of
  18 intermediates (see Results below) -- assessed as likely bf16 precision
  noise, not yet confirmed via direct tensor-level comparison.
- `mosaic` (v1) has not actually executed a kernel yet on the results below --
  every bundle used so far has dispatch rows `M < 128`, below its tiling
  floor. A `num_tokens=64` ("mosaic_wide", `M=128`) bundle that can exercise
  it was added 2026-08-28 but not yet run on hardware.
- A latency benchmark across multiple batch sizes/sequence lengths
  (`run_latency_sweep`, added 2026-08-28) exists but hasn't been run on
  hardware yet -- the only latency numbers on record predate the
  SiTU-GLU/precision/routing fixes and are stale (see project history).
- Full-dimension (`num_experts=896`, `top_k=16`) validation.

**Not yet covered:**
- The real, MXFP4-quantized Kimi K3 checkpoint weights (bundles so far use
  random weight init).
- Multi-device/sharded expert dispatch.
- Any form of full-model deployment.

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
(fp32 and bf16, at two reduced scales -- both use random weight init, not the
real checkpoint), and cross-checks:

1. The naive JAX reference implementation against the golden bundles --
   matches on all 18 staged intermediates (router, dispatch, per-expert
   compute, combine, RMSNorm, shared expert, final output), both dtypes.
2. `tokamax.ragged_dot`'s `xla`, `mosaic` (v1), and `mosaic_tpu_v2`
   implementations against the same golden bundles, run on real TPU v6e
   hardware.

Both reduced-scale configs use `num_experts=8`, `top_k=2`, `num_shared_experts=1`
(the real model is `num_experts=896`, `top_k=16`) -- see `SMALL_CONFIG_KWARGS`
and `MOSAIC_CONFIG_KWARGS` in `generate_pytorch_golden.py`. The "mosaic"-named
bundle (256/128/128 hidden/latent/intermediate dims) was sized so K and N meet
Mosaic v1's 128-element tiling floor, but at its original `num_tokens=20`,
dispatch rows (`M = num_tokens x top_k = 20 x 2 = 40`) still fall below that
floor -- the name reflected intent, not confirmed v1 compatibility (see the
SKIPPED rows below). A third, "mosaic_wide" bundle (same dims, `num_tokens=64`
-> `M=128`, generated via `generate_pytorch_golden.py --config mosaic
--num-tokens 64`) was added 2026-08-28 to actually satisfy the floor -- not
yet run on hardware, so mosaic (v1)'s output has still never been checked
against real official-model ground truth as of this note.

### Result on real v6e hardware (`test_ragged_dot_against_pytorch_golden.py`)

Eight validation combinations (2 bundles x 2 dtypes x up to 3 implementations,
`mosaic` only attempted where dims allow) were run against golden bundles
generated from the official model. Of these: **4 passed** (fp32, `xla` and
`mosaic_tpu_v2`), **2 exceeded the current bf16 tolerance**, and **2 `mosaic`
(v1) cases did not execute a kernel** (`M=40` below its 128-row floor --
correctly counted as not-passed, not skipped-as-pass):

| Bundle | dtype | Implementation | Result |
|---|---|---|---|
| small (64/32/48) | fp32 | xla | ALL STAGES MATCH |
| small (64/32/48) | bf16 | xla | ALL STAGES MATCH (bit-exact) |
| "mosaic" (256/128/128) | fp32 | xla | ALL STAGES MATCH |
| "mosaic" (256/128/128) | fp32 | mosaic (v1) | NOT RUN -- dispatch rows `M=40 < 128` tiling floor |
| "mosaic" (256/128/128) | fp32 | mosaic_tpu_v2 | ALL STAGES MATCH |
| "mosaic" (256/128/128) | bf16 | xla | 4 of 18 stages exceed tolerance (below) |
| "mosaic" (256/128/128) | bf16 | mosaic (v1) | NOT RUN -- same tiling floor reason |
| "mosaic" (256/128/128) | bf16 | mosaic_tpu_v2 | Same 4 stages, same magnitudes as xla above |

fp32 max abs diffs against the official model ranged from bit-exact to
~2.15e-6 across all 18 staged intermediates -- reaching this required
wrapping the whole computation in `jax.default_matmul_precision("highest")`,
since TPU's default matmul precision (`precision=None`) silently uses a
reduced-precision path for float32 inputs on the MXU (invisible on CPU,
confirmed via the compiled HLO).

For bf16 on the "mosaic" bundle, 4 of 18 intermediates exceed the 1e-3
tolerance, both on `xla` and on `mosaic_tpu_v2`:

| Intermediate | max abs diff | tolerance |
|---|---|---|
| `expert_up_output` | 1.95e-3 | 1e-3 |
| `up_projection_output` | 7.81e-3 | 1e-3 |
| `final_output` | 7.81e-3 | 1e-3 |
| `normalized_output` | 1.56e-2 | 1e-3 |

The other 8 quantitative stages (plus 2 exact-match stages) are within
tolerance; the failing stages are downstream of `expert_up_output`, consistent
with one bf16 rounding difference propagating forward rather than 4
independent errors. `xla` and `mosaic_tpu_v2` report the *same* max abs diff
at each of these stages, which suggests a shared bf16/backend precision
effect rather than a `mosaic_tpu_v2`-specific bug -- but this is only an
inference from matching summary statistics; a direct same-device,
element-wise comparison of the two implementations' raw output tensors has
not been done yet.

`mosaic` (v1) has not been exercised end-to-end on any bundle generated so
far -- both bundles' dispatch row count is below its 128-row tiling floor, and
the harness correctly counts this as not-run rather than as a pass;
`NotImplementedError`/skip is never counted as a pass.

## Reproducing

```bash
cd 06_kimi_k3_golden_validation

# 1. Verify the official Kimi K3 source snapshot (downloads + hashes it).
python validate_official_config.py
python verify_official_snapshot.py

# 2. Generate golden bundles from the official PyTorch model (no TPU needed).
python generate_pytorch_golden.py --dtype fp32 --config small
python generate_pytorch_golden.py --dtype bf16 --config small
python generate_pytorch_golden.py --dtype fp32 --config mosaic
python generate_pytorch_golden.py --dtype bf16 --config mosaic
python generate_pytorch_golden.py --dtype fp32 --config mosaic --num-tokens 64  # "mosaic_wide" -- M=128, needed for mosaic (v1)
python generate_pytorch_golden.py --dtype bf16 --config mosaic --num-tokens 64

# 3. Validate the naive JAX reference against the bundles (no TPU needed).
python test_jax_reference_against_pytorch_golden.py --variant both

# 4. Validate tokamax.ragged_dot's xla/mosaic/mosaic_tpu_v2 against the same
#    bundles (requires a real TPU VM, generation >= 5).
python test_ragged_dot_against_pytorch_golden.py \
  --bundle-set both --variant both --implementation all

# 5. Latency across multiple batch sizes/sequence lengths (also needs a TPU VM).
cd ../05_ragged_dot_on_tpu
python kimi_k3_latent_moe_ragged_dot.py --latency-sweep
```

## Requirements

- **Pallas exercises** (`01_pallas_basics/`, `04_own_kernel/`): `jax` only.
  Run locally under `interpret=True` (no TPU needed) to check kernel
  semantics; real Mosaic/TPU-backend correctness and performance require a
  TPU VM (generation >= 5 for the `mosaic_tpu` backend).
- **`ragged_dot` benchmarking and validation** (`05_ragged_dot_on_tpu/`, and
  the `xla`/`mosaic`/`mosaic_tpu_v2` checks in
  `06_kimi_k3_golden_validation/`): `jax` (>= 0.11.0) +
  [`tokamax`](https://github.com/openxla/tokamax) on a real TPU VM. tokamax
  was installed by cloning `main` and building from source
  (`pip install -e ".[tpu,test]"`); the exact commit used for the hardware
  run recorded above was not pinned/recorded at the time, so it isn't stated
  here -- re-running against current `main` is the best available
  reproduction path until that's tracked down.
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
