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
- `xla`, `mosaic` (v1), and `mosaic_tpu_v2` (`tokamax.ragged_dot`) all
  validated against the same bundles on real TPU v6e hardware, fp32 --
  `mosaic` (v1) actually executed a kernel and matched for the first time
  2026-08-28, via the new "mosaic_wide" bundle (`M=128`, see below).
- Latency across multiple batch sizes/sequence lengths, real v6e hardware,
  2026-08-28 -- the first latency numbers measured against the corrected
  (post SiTU-GLU/precision-fix) architecture. `mosaic_tpu_v2` is the
  fastest implementation at every scale tested (see Latency below).
- The real 16-of-896-then-filtered-to-local-shard routing, dispatch, and
  weighted combine -- previously only tested in isolated pieces -- wired
  into one coherent forward pass and verified end-to-end against an
  unsharded reference, entirely locally (no TPU): summing every shard's
  contribution reproduces the unsharded computation's output exactly
  (`max_err=9.31e-10` at a small toy scale). See
  `check_sharded_forward_correctness` in
  `05_ragged_dot_on_tpu/kimi_k3_latent_moe_reference.py`.
- 10 local (CPU-only, no TPU needed) unit tests for that routing pipeline's
  edge cases -- empty shards, extreme skew, capacity overflow, padding
  contamination, `top_k=16`, global/local expert-id boundary off-by-ones,
  first/last shard, batch/seq invariance, fp32/bf16 dtype promotion, and
  the sharded-sum-vs-reference check across several configs (not just one)
  -- all pass. See `05_ragged_dot_on_tpu/test_sharded_routing_local.py`.
- A theoretical (lower-bound) memory budget estimate for the sharded
  pipeline at real Kimi K3 dims, across a wide batch/seq sweep, so
  OOM-prone shapes are flagged before ever attempting them on hardware --
  see `05_ragged_dot_on_tpu/memory_budget_estimate.py` and the Memory
  budget section below.

**In progress / open:**
- bf16 on `xla`/`mosaic` (v1)/`mosaic_tpu_v2` shows a tolerance-exceeding
  residual on 4 of 18 intermediates (see Results below) -- assessed as
  likely bf16 precision noise, not yet confirmed via direct tensor-level
  comparison.
- Full-dimension (`num_experts=896`, `top_k=16`) validation.
- A `tokamax.ragged_dot`-based (not naive-loop) version of the sharded
  forward pass above (`check_sharded_ragged_dot_correctness`), plus a
  latency sweep under the REAL routing distribution instead of the dense
  `single_chip_kimi_k3_config` simplification
  (`run_realistic_shard_latency_sweep`) -- both written 2026-08-28 but not
  yet run on hardware.
- **WP4 (SparseCore feasibility)**: a profiling harness splitting the
  forward pass into 4 stages -- A (router+projection, regular), B (dispatch
  indexing: reshape/filter/argsort/gather/bincount/padding, irregular), C
  (expert compute), D (combine, irregular) -- to measure whether B+D are
  actually a bottleneck, rather than assuming it. `profile_dispatch_vs_compute.py`
  runs A/B/D for real (Stage C is a dense-matmul stand-in, no tokamax
  needed) and is verified locally: on this machine's CPU, the irregular
  (B+D) share is 8.8% (num_tokens=128) shrinking to 3.7% (num_tokens=2048)
  -- explicitly a CPU methodology check, not a TPU finding (no MXU/VMEM on
  CPU to reflect real TPU behavior). `kimi_k3_latent_moe_ragged_dot.py`'s
  `profile_four_stages_wp4` is the same A/B/D functions plus the REAL
  `tokamax.ragged_dot` for Stage C -- written, not yet run on hardware.
- A direct, same-device, element-wise comparison of `xla`/`mosaic`/
  `mosaic_tpu_v2`'s bf16 outputs against EACH OTHER, across all 18 staged
  intermediates (`compare_bf16_implementations_direct.py`) -- every prior
  "shared bf16 precision effect" conclusion was only ever inferred from
  matching max-abs-diff *summary statistics* computed in separate runs
  against the golden reference, never a direct tensor diff between the
  implementations' own outputs in one run. Saves each implementation's full
  raw output bundle to `<name>_outputs.npz` so the tensors can be
  re-examined later without a TPU, and prints an explicit verdict: if every
  pair is bit-identical at the 4 previously-flagged stages, that's real
  evidence of a shared PyTorch-vs-TPU precision effect rather than a
  kernel-specific bug (three independently-coded kernels landing on the
  exact same bits by coincidence of a shared bug is far less likely than
  landing on the same bits via the same underlying bf16 MXU behavior).
  Written 2026-08-28, not yet run on hardware.
- **`run_v6e_experiment_suite.py`**: a one-shot entry point running the
  full battery above (env check, snapshot verification, sharded
  correctness, realistic-distribution latency, direct bf16 comparison, the
  full golden-validation battery, WP4 profiling) in a fixed order, each
  step as its own subprocess so one step's OOM/compile error/crash never
  blocks the rest. Writes `environment.json`, `summary.json`/`summary.csv`
  (status per step: `PASS`/`FAIL`/`UNSUPPORTED`/`OOM`/`COMPILE_ERROR`), and
  a full log per step. Verified locally that the harness mechanics work
  (every step correctly attempted and logged; the real steps all FAIL here
  since this machine has neither a TPU nor tokamax installed) -- the
  actual PASS/FAIL results are only meaningful once run on a v6e VM.
- **WP-KV6 (real checkpoint) prep**, without downloading the actual
  ~17GB-per-layer shard: the exact MXFP4 weight-field mapping for one MoE
  layer (confirmed against the real cached official source, not guessed --
  see `06_kimi_k3_golden_validation/prepare_real_checkpoint_layer.py`),
  which tensors need dequantization (routed-expert `w1`/`w2`/`w3` only),
  the extraction-script interface, the metadata/hash format, disk/memory
  requirements, and a real (not hand-rolled) MXFP4 dequantization round-trip
  verified locally against a small synthetic tensor using the actual
  `compressed-tensors` library.

**Not yet covered:**
- The real, MXFP4-quantized Kimi K3 checkpoint weights (bundles so far use
  random weight init) -- the extraction pipeline is designed and its
  dequantization step verified (see above), but the actual checkpoint shard
  has not been downloaded and `extract_moe_layer_from_shard` is not yet
  runnable.
- Actual multi-device/multi-chip execution (everything so far has run on a
  single chip; the sharded routing above proves the per-shard math is
  correct, not that multiple chips actually combine their contributions
  over a network).
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
dispatch rows (`M = num_tokens x top_k = 20 x 2 = 40`) still fell below that
floor -- the name reflected intent, not confirmed v1 compatibility. A third,
"mosaic_wide" bundle (same dims, `num_tokens=64` -> `M=128`, generated via
`generate_pytorch_golden.py --config mosaic --num-tokens 64`) was added
2026-08-28 to actually satisfy the floor, and run on real v6e hardware the
same day -- see Results below.

### Result on real v6e hardware (`test_ragged_dot_against_pytorch_golden.py`)

**"mosaic_wide" bundle (`num_tokens=64`, `M=128`), 2026-08-28 -- `mosaic` (v1)
executed a kernel and matched for the first time in this project:**

| dtype | Implementation | Result |
|---|---|---|
| fp32 | xla | ALL STAGES MATCH |
| fp32 | mosaic (v1) | **ALL STAGES MATCH** (first real v1 execution, not a skip) |
| fp32 | mosaic_tpu_v2 | ALL STAGES MATCH |
| bf16 | xla | 4 of 18 stages exceed tolerance (below) |
| bf16 | mosaic (v1) | Same 4 stages, same magnitudes as xla |
| bf16 | mosaic_tpu_v2 | Same 4 stages, same magnitudes as xla and mosaic (v1) |

**Earlier bundles** ("small" 64/32/48, and "mosaic" 256/128/128 at its
original `num_tokens=20`, `M=40`):

| Bundle | dtype | Implementation | Result |
|---|---|---|---|
| small (64/32/48) | fp32 | xla | ALL STAGES MATCH |
| small (64/32/48) | bf16 | xla | ALL STAGES MATCH (bit-exact) |
| "mosaic" (256/128/128, n=20) | fp32 | xla | ALL STAGES MATCH |
| "mosaic" (256/128/128, n=20) | fp32 | mosaic (v1) | NOT RUN -- dispatch rows `M=40 < 128` tiling floor |
| "mosaic" (256/128/128, n=20) | fp32 | mosaic_tpu_v2 | ALL STAGES MATCH |
| "mosaic" (256/128/128, n=20) | bf16 | xla | Same 4-stage residual as the mosaic_wide bf16 row above |
| "mosaic" (256/128/128, n=20) | bf16 | mosaic (v1) | NOT RUN -- same tiling floor reason |
| "mosaic" (256/128/128, n=20) | bf16 | mosaic_tpu_v2 | Same 4-stage residual |

fp32 max abs diffs against the official model ranged from bit-exact to
~3.0e-6 across all 18 staged intermediates (across both the n=20 and n=64
mosaic bundles) -- reaching this required wrapping the whole computation in
`jax.default_matmul_precision("highest")`, since TPU's default matmul
precision (`precision=None`) silently uses a reduced-precision path for
float32 inputs on the MXU (invisible on CPU, confirmed via the compiled HLO).

For bf16, the same 4 of 18 intermediates exceed the 1e-3 tolerance on every
run so far (n=20 and n=64 bundles alike), with the same max abs diffs across
**all three** implementations -- xla, mosaic (v1), and mosaic_tpu_v2:

| Intermediate | max abs diff | tolerance |
|---|---|---|
| `expert_up_output` | 1.95e-3 | 1e-3 |
| `up_projection_output` | 7.81e-3 | 1e-3 |
| `final_output` | 7.81e-3 | 1e-3 |
| `normalized_output` | 1.56e-2 | 1e-3 |

The other 8 quantitative stages (plus 2 exact-match stages) are within
tolerance; the failing stages are downstream of `expert_up_output`, consistent
with one bf16 rounding difference propagating forward rather than 4
independent errors. That three separate kernel implementations -- including
mosaic (v1), a completely different codebase from mosaic_tpu_v2 -- report the
*identical* residual at every failing stage is fairly strong evidence this is
a shared bf16/backend precision effect rather than a bug in any one of them,
but it is still an inference from matching summary statistics across runs,
not a direct same-device, element-wise comparison of the three
implementations' raw output tensors, which hasn't been done.

## Latency across batch size / sequence length (`run_latency_sweep`)

Real v6e hardware, 2026-08-28 -- single-chip-shard scale (`num_experts=64`,
`latent_size=3584`, `intermediate_size=3072`, bf16), heuristic config (no
autotuning -- see caveat below), full `latent_moe_forward_ragged_dot` forward
pass (not just the isolated expert FFN). `hidden_states` is
`(num_tokens, hidden_size)`, so `batch_size` and `seq_len` only ever enter
through their product; two pairs per `num_tokens` value are included
deliberately as a sanity check:

| batch | seq_len | num_tokens | xla (ms) | mosaic v1 (ms) | mosaic_tpu_v2 (ms) | peak_mem (MB) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 128 | 128 | 5.584 | 67.809 | 3.281 | 4694.35 |
| 1 | 512 | 512 | 5.868 | 111.052 | 4.142 | 4770.15 |
| 1 | 2048 | 2048 | 12.064 | 280.699 | 8.045 | 5293.22 |
| 2 | 1024 | 2048 | 12.047 | 280.806 | 8.042 | 5293.22 |
| 1 | 4096 | 4096 | 21.775 | 508.369 | 14.916 | 5990.66 |
| 4 | 1024 | 4096 | 21.768 | 508.581 | 14.834 | 5990.66 |

Observations:
- **The `(batch_size, seq_len)` sanity check passes**: `(1,2048)` vs.
  `(2,1024)` and `(1,4096)` vs. `(4,1024)` match within measurement noise on
  both latency and memory at every implementation -- confirms latency here
  really is a function of `num_tokens` alone, as the architecture implies,
  not something that would vary if the same total were split into more/fewer
  batches.
- **`mosaic_tpu_v2` is the fastest implementation at every scale tested**,
  30-41% faster than `xla` (e.g. 14.9ms vs. 21.8ms at `num_tokens=4096`).
  This is the first latency measurement against the corrected (post
  SiTU-GLU/float32-precision-fix) architecture -- the only prior number
  (`~34% faster than xla` at `num_tokens=2048`, from 2026-08-24) predated
  those fixes and was explicitly marked not reportable; this table
  supersedes it with a consistent finding across 4 scales, not just one.
- **`mosaic` (v1) is dramatically slower** -- 12-23x slower than `xla`,
  20-34x slower than `mosaic_tpu_v2`, consistent with the heuristic-config
  gap seen elsewhere in this project (WP1's DeepSeek-scale benchmark, WP-Kimi
  step 2's earlier single-shape run).
- Peak memory is dominated by the fixed per-expert weight footprint (~4.2GB
  at `num_experts=64`) rather than activations -- it grows only modestly
  (4.69GB -> 5.99GB) across a 32x increase in `num_tokens`.
- **Caveat**: heuristic config only, no autotuning -- autotuning this shape
  was confirmed impractically slow elsewhere in this project (~2400
  microbenchmarks, still not complete after 12+ minutes). `mosaic` (v1)'s
  gap in particular might narrow under a tuned config; this table should be
  read as "heuristic-config latency," not a ceiling on Mosaic's achievable
  performance.

## Memory budget (`memory_budget_estimate.py`)

Theoretical (lower-bound) memory estimate for the sharded pipeline at real
Kimi K3 dims (`local_num_experts=64`, bf16, `capacity_factor=2.0`), computed
without any TPU -- exact byte counts for every named tensor the pipeline
actually holds, deliberately excluding compiler-managed temporaries (upcast
copies, Pallas double-buffering, XLA scratch), so this is a genuine floor:
if the estimate already exceeds available HBM, the real run definitely
OOMs; staying under it is not a guarantee.

Weight memory is constant across shapes: **~4.46GB** (`expert_gate`/`up`/
`down` at ~1.33GB each dominate; router/projection/shared-expert weights
are a few hundred MB combined).

| Tokens | Local assignments | Weight memory | Activation memory | Padding memory | Estimated total |
|---:|---:|---:|---:|---:|---:|
| 128 | 146.3 | 4457.3MB | 26.1MB | 3.3MB | 4.38GB |
| 2,048 | 2,340.6 | 4457.3MB | 374.0MB | 32.8MB | 4.74GB |
| 8,192 | 9,362.3 | 4457.3MB | 1,492.0MB | 129.3MB | 5.89GB |
| 32,768 | 37,449.1 | 4457.3MB | 5,960.0MB | 513.5MB | 10.50GB |
| 131,072 | 149,796.6 | 4457.3MB | 23,828.0MB | 2,048.8MB | 28.94GB |
| 262,144 | 299,593.1 | 4457.3MB | 47,656.0MB | 4,097.5MB | **53.53GB -- OOM** |

(Full sweep, including the `(batch_size, seq_len)` pairs already used
elsewhere in this README, in `memory_budget_estimate.py`'s output --
`(1,2048)`/`(2,1024)` and `(1,4096)`/`(4,1024)` give identical rows, the
same batch/seq invariance already confirmed for latency.)

**Every shape actually used in this project's existing latency sweeps fits
comfortably** (well under 6GB up to `num_tokens=8192`); the estimate only
crosses the assumed 32GB v6e HBM ceiling somewhere between `num_tokens=
131,072` and `262,144` -- a useful upper bound for how far a future
batch/seq sweep could push before needing to worry about OOM, though the
real (non-lower-bound) threshold will be lower than this.

**Caveat**: the 32GB per-chip HBM figure is a public TPU v6e (Trillium)
spec, not independently confirmed by a real device query in this project --
`memory_budget_estimate.py`'s `__main__` prints the `jax.devices()[0].memory_stats()`
snippet needed to get the real number next time a TPU VM is available.

## Real checkpoint validation prep (`prepare_real_checkpoint_layer.py`)

Preparation for WP-KV6 (validating against the real, MXFP4-quantized Kimi
K3 checkpoint, not the random-init bundles used everywhere else) -- done
without downloading the actual checkpoint, so that step doesn't need
designing from scratch whenever the ~17GB-per-layer shard (or server
resources to hold it) become available.

**Weight field mapping**, confirmed directly against the cached, hash-verified
official source (`official_kimi_k3/modeling_kimi_linear.py`), not guessed:
for MoE layer `{i}` (`i >= 1` -- layer 0 is dense, not MoE), under
`model.layers.{i}.block_sparse_moe.`:

| Component | Key | Quantized? |
|---|---|---|
| Router weight | `gate.weight` | No (plain bf16) |
| Router bias | `gate.e_score_correction_bias` | No |
| Down projection | `routed_expert_down_proj.weight` | No |
| Up projection | `routed_expert_up_proj.weight` | No |
| RMSNorm scale | `routed_expert_norm.weight` | No |
| Shared experts | `shared_experts.{gate,up,down}_proj.weight` | No |
| Routed expert gate (w1) | `experts.{e}.w1.weight_packed` + `.weight_scale` | **Yes -- MXFP4** |
| Routed expert up (w3) | `experts.{e}.w3.weight_packed` + `.weight_scale` | **Yes -- MXFP4** |
| Routed expert down (w2) | `experts.{e}.w2.weight_packed` + `.weight_scale` | **Yes -- MXFP4** |

**Dequantization**: verified locally using the real
[`compressed-tensors`](https://github.com/vllm-project/compressed-tensors)
library (`MXFP4PackedCompressor`, format `mxfp4-pack-quantized`,
`num_bits=4`, `group_size=32`) -- not a hand-rolled bit-unpacker. A synthetic
round-trip test (random weight -> quantize -> pack -> the real library's
`decompress()` -> compare) confirms the calling convention is correct;
production-scale numerical accuracy against real trained weights is a
separate question this test doesn't (and can't, without the real weights)
answer.

**Disk / memory requirements** (from WP-KV1's already-confirmed real
measurements, 2026-08-26): one shard holds one full layer (~17GB), 96
shards total (~1.42TiB / ~1.56TB decimal). Dequantizing all 896 experts'
routed weights for one layer at once needs ~55GB in bf16 -- would OOM a
single chip, same figure already confirmed elsewhere in this project for
random-init weights. Restricting to this project's established
single-chip-shard scope instead:

| local_num_experts | Dequantized memory |
|---:|---:|
| 64 | ~3.9GB |
| 32 | ~2.0GB |
| 16 | ~1.0GB |

**Not yet done** (needs the real checkpoint): `extract_moe_layer_from_shard`'s
actual implementation (interface + exact logic written, raises
`NotImplementedError` until a real shard is available), and downloading a
shard at all -- this file only proves the pipeline it will run is
correctly designed and its dequantization step actually works.

## Reproducing

**Local routing unit tests** (no TPU needed, run this any time -- also the
first step `run_v6e_experiment_suite.py` runs, so a logic regression is
caught before spending real VM time on it):

```bash
cd 05_ragged_dot_on_tpu
python test_sharded_routing_local.py
python memory_budget_estimate.py
```

**Real checkpoint prep check** (no TPU needed, needs the `.venv_torch`-style
environment with `torch` + `compressed-tensors`, not the jax/tokamax one):

```bash
cd 06_kimi_k3_golden_validation
python prepare_real_checkpoint_layer.py
```

**One-shot TPU VM entry point**: once golden bundles exist (step 2 below,
no TPU needed), everything that needs a v6e VM can run in one command:

```bash
python run_v6e_experiment_suite.py --output-dir results/YYYY-MM-DD-v6e
```

See `run_v6e_experiment_suite.py`'s module docstring for exactly what it
runs and how results are structured. The step-by-step commands below are
what it runs internally, useful for running one piece in isolation or for
regenerating golden bundles (which it doesn't do itself).

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

# 5. Direct, same-device, element-wise comparison of the three
#    implementations' bf16 outputs against each other, across all 18
#    stages -- saves <name>_outputs.npz per implementation (needs a TPU VM).
python compare_bf16_implementations_direct.py --bundle-set mosaic_wide --output-dir bf16_direct_compare_outputs

# 6. Latency across multiple batch sizes/sequence lengths (also needs a TPU VM).
cd ../05_ragged_dot_on_tpu
python kimi_k3_latent_moe_ragged_dot.py --latency-sweep

# 7. WP4 profiling: 4-stage dispatch-vs-compute breakdown.
python profile_dispatch_vs_compute.py            # A/B/D for real, C is a dense-matmul stand-in (no TPU needed)
python kimi_k3_latent_moe_ragged_dot.py --wp4-profile  # same A/B/D, C is the REAL tokamax.ragged_dot (needs a TPU VM)
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
- **Real checkpoint prep** (`prepare_real_checkpoint_layer.py`, no TPU
  needed): `torch` + `compressed-tensors` (`compressed-tensors==0.18.0`
  tested). Actually running `extract_moe_layer_from_shard` will additionally
  need `safetensors` and the real checkpoint shard(s), neither available yet.

## References

- JAX Pallas TPU tutorial: https://docs.jax.dev/en/latest/pallas/tpu/index.html
- tokamax: https://github.com/openxla/tokamax
- Kimi K3: https://huggingface.co/moonshotai/Kimi-K3
