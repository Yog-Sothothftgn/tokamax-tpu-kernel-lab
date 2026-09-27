"""A from-scratch, hand-written Pallas TPU kernel for Kimi K3's gate/up
projection + SiTU-GLU activation -- written to actually satisfy Zifan's
"push to use raw Pallas and do kernel fusion" request, after the previous
attempt (`_local_shard_expert_ffn_ragged_dot_fused_gateup` in
`kimi_k3_latent_moe_ragged_dot.py`) turned out to just be a correct CALL
into tokamax's own already-built fusion feature -- zero Pallas code
written by this project, same category of shortcut Zifan originally
flagged ("you're just importing ops from tokamax").

**Deliberately scoped down from the real MoE case, per explicit user
direction (2026-09-20)**: ONE expert (dense, no ragged/grouped dispatch),
real Kimi K3 per-expert matrix dimensions (`latent_size=3584`,
`intermediate_size=3072`, confirmed from `kimi_k3_config()` in
`kimi_k3_latent_moe_reference.py`):

  X:              (M, 3584)
  W_gate, W_up:   (3584, 3072)
  Y (output):     (M, 3072)

This is the K-tiled matmul template this project already built and
verified in `01_pallas_basics/03_matmul_k_tiled.py` (Task A) -- extended
from ONE rhs matrix to TWO (gate and up, sharing the same lhs and the same
K-tiling schedule), with a fused activation applied to the two
accumulators on the LAST K step, instead of writing each matmul's result
back to HBM separately.

Kernel structure (grid = (m_tiles, n_tiles, k_tiles), K innermost/
"arbitrary" since the same (i,j) output block accumulates across K):
  1. Output tiling: each grid step (i, j, k) owns output block (i, j);
     `out_specs`' index_map ignores `k` (same convention as the K-tiled
     matmul template) -- the block is only actually WRITTEN on the last k.
  2. `in_specs` walk `x` along (i, k) and `w_gate`/`w_up` along (k, j) --
     the K-dimension reduction is loaded incrementally, tile by tile, not
     materialized as one whole (M, 3584) / (3584, 3072) block in VMEM.
  3. TWO VMEM scratch accumulators (`gate_acc_ref`, `up_acc_ref`), zeroed
     at `k_id == 0`, each accumulating its own partial matmul independently
     every K step.
  4. On the last K step: round each accumulator to `round_dtype` (bf16)
     first, THEN upcast to float32 to compute SiTU-GLU -- reproducing this
     project's established `_situ_and_mul` convention (see that function's
     docstring in `kimi_k3_latent_moe_reference.py`), not the raw fp32
     accumulator directly, so this kernel's numerics are directly
     comparable to the rest of this project's Kimi K3 pipeline.
  5. Only the ACTIVATED result is ever written to `o_ref` -- `gate_acc_ref`/
     `up_acc_ref` never touch HBM at all, at any K step.

Verified via `pl.pallas_call(..., interpret=True)` -- genuine Pallas
kernel semantics (grid iteration, BlockSpec indexing, scratch persistence
across the K loop, `pl.when` boundary conditions), not just a plain-JAX
formula check, runnable on this CPU-only Windows machine with NO real TPU
-- same technique `03_matmul_k_tiled.py`'s own `check()` already
established for this project. Real Mosaic-compiled TPU behavior (VMEM
budget at these real, much larger-than-`03_matmul_k_tiled.py`'s-toy-sizes
dimensions, actual tile-size tuning) still needs the real v6e TPU VM --
not yet run there as of this file's creation.

To run (local, interpret mode, no TPU needed):
  python own_gateup_situ_pallas_kernel.py
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Real Kimi K3 per-expert dimensions (kimi_k3_config()) and SiTU-GLU
# constants (config.json's activation_situ_beta/activation_situ_linear_beta).
LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072
SITU_BETA = 4.0
SITU_LINEAR_BETA = 25.0


def fused_gateup_situ_kernel(
    x_ref,
    wgate_ref,
    wup_ref,
    o_ref,
    gate_acc_ref,
    up_acc_ref,
    *,
    num_k_tiles: int,
    beta: float,
    linear_beta: float,
    round_dtype: jnp.dtype,
):
  """`gate_acc_ref`/`up_acc_ref`'s dtype controls accumulation precision --
  fp32 (this file's default) accumulates every K step in full precision;
  bf16 (see `fused_gateup_situ`'s `acc_dtype`) rounds the accumulator down
  to bf16 after EVERY K step, halving accumulator VMEM at the cost of
  compounding rounding error across `num_k_tiles` steps instead of once at
  the end -- a real precision-vs-memory tradeoff, not free, added
  2026-09-21 specifically to test whether it reopens tile-size options
  that hit the 32MB scoped-VMEM ceiling with fp32 accumulators (see this
  file's `__main__` config comments for the real OOMs that motivated this).
  """
  k_id = pl.program_id(2)

  @pl.when(k_id == 0)
  def _():
    gate_acc_ref[...] = jnp.zeros_like(gate_acc_ref)
    up_acc_ref[...] = jnp.zeros_like(up_acc_ref)

  partial_gate = jnp.dot(x_ref[...], wgate_ref[...], preferred_element_type=jnp.float32)
  partial_up = jnp.dot(x_ref[...], wup_ref[...], preferred_element_type=jnp.float32)
  gate_acc_ref[...] += partial_gate.astype(gate_acc_ref.dtype)
  up_acc_ref[...] += partial_up.astype(up_acc_ref.dtype)

  @pl.when(k_id == num_k_tiles - 1)
  def _():
    # Deliberately round to bf16 THEN upcast, matching this project's
    # established _situ_and_mul convention (see kimi_k3_latent_moe_reference.py)
    # -- not the raw fp32 accumulator directly. A no-op cast when
    # gate_acc_ref/up_acc_ref are already bf16 (round_dtype == acc_dtype).
    gate = gate_acc_ref[...].astype(round_dtype).astype(jnp.float32)
    up = up_acc_ref[...].astype(round_dtype).astype(jnp.float32)
    situ_gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
    bounded_up = linear_beta * jnp.tanh(up / linear_beta)
    activated = situ_gate * bounded_up
    o_ref[...] = activated.astype(o_ref.dtype)


def fused_gateup_situ(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    *,
    beta: float = SITU_BETA,
    linear_beta: float = SITU_LINEAR_BETA,
    round_dtype: jnp.dtype = jnp.bfloat16,
    acc_dtype: jnp.dtype = jnp.float32,
    bm: int = 128,
    bk: int = 512,
    bn: int = 512,
) -> jax.Array:
  """One expert's gate/up projection + SiTU-GLU, fused into a single
  hand-written Pallas kernel. `x`: (M, K). `w_gate`/`w_up`: (K, N). Returns
  (M, N) -- the ACTIVATED result only; gate/up never leave VMEM.

  `acc_dtype`: dtype of the two VMEM scratch accumulators. `float32`
  (default) matches the rest of this project's precision conventions;
  `bfloat16` halves accumulator VMEM (`bm*bn*2bytes*2` instead of
  `bm*bn*4bytes*2`) at the cost of rounding after every K step instead of
  only at the very end -- see `fused_gateup_situ_kernel`'s docstring.
  """
  m, k = x.shape
  k2, n = w_gate.shape
  assert k == k2, f"x/w_gate inner dims don't match: {x.shape} @ {w_gate.shape}"
  assert w_up.shape == w_gate.shape, (
      f"w_gate/w_up shape mismatch: {w_gate.shape} vs {w_up.shape}"
  )
  assert m % bm == 0 and n % bn == 0 and k % bk == 0, (
      "keep the exercise simple: only support shape/tile combinations that "
      "divide evenly for now (matches 03_matmul_k_tiled.py's own convention)"
  )

  num_k_tiles = k // bk
  has_tpu = any(d.platform == "tpu" for d in jax.devices())

  return pl.pallas_call(
      functools.partial(
          fused_gateup_situ_kernel,
          num_k_tiles=num_k_tiles,
          beta=beta,
          linear_beta=linear_beta,
          round_dtype=round_dtype,
      ),
      grid=(m // bm, n // bn, num_k_tiles),
      in_specs=[
          # x: row-block i fixed, walks the k-th K-tile.
          pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
          # w_gate: walks the k-th K-tile, column-block j fixed.
          pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
          # w_up: same walk as w_gate, independent weight array.
          pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
      ],
      # Same (i, j) output block "visited" every K step, only WRITTEN on
      # the last one (see pl.when in the kernel) -- identical convention to
      # 03_matmul_k_tiled.py.
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      scratch_shapes=[
          pltpu.VMEM((bm, bn), acc_dtype),  # gate_acc
          pltpu.VMEM((bm, bn), acc_dtype),  # up_acc
      ],
      compiler_params=(
          pltpu.CompilerParams(
              dimension_semantics=("parallel", "parallel", "arbitrary"),
          )
          if has_tpu
          else None
      ),
      interpret=not has_tpu,
  )(x, w_gate, w_up)


def _reference_gateup_situ(
    x: jax.Array,
    w_gate: jax.Array,
    w_up: jax.Array,
    beta: float,
    linear_beta: float,
    round_dtype: jnp.dtype,
) -> jax.Array:
  """Plain-JAX reference: same bf16-round-then-upcast convention as the
  kernel, but via ordinary (non-Pallas) matmuls -- what
  `_situ_and_mul(x @ w_gate, x @ w_up, ...)` computes in
  `kimi_k3_latent_moe_reference.py`, reproduced standalone here so this
  file has zero dependency on that module (or tokamax) for its own
  correctness check.
  """
  gate = (x @ w_gate).astype(round_dtype).astype(jnp.float32)
  up = (x @ w_up).astype(round_dtype).astype(jnp.float32)
  situ_gate = beta * jnp.tanh(gate / beta) * jax.nn.sigmoid(gate)
  bounded_up = linear_beta * jnp.tanh(up / linear_beta)
  return (situ_gate * bounded_up).astype(x.dtype)


def inspect_unfused_hlo(m: int = 2048, dtype=jnp.bfloat16) -> str:
  """Prints and returns the compiled HLO for `_reference_gateup_situ` (the
  "unfused" baseline) at real Kimi K3 dimensions -- step 2 of the
  ChatGPT-relayed debugging plan (2026-09-25): understand what the
  opponent actually does before assuming it's a naive two-separate-
  matmuls-plus-elementwise graph.

  Uses `.lower(...).compile().as_text()`, which compiles for whatever
  backend is actually active in this process (CPU here if no TPU is
  present, XLA:TPU on the real v6e VM) -- the FUSION DECISIONS shown here
  are for the ACTIVE backend, and CPU vs TPU's XLA backends can make
  genuinely different fusion/tiling choices for the same program. Read the
  printed text for: (a) how many `fusion` computations exist and what they
  contain (does the compiler group the two matmuls with the SiTU-GLU
  elementwise math into one fused kernel, or keep them as 3 separate
  ops?), (b) whether `gate`/`up` intermediates appear as materialized
  buffers between the matmul and the activation, or only inside a fusion
  region (never separately materialized). Re-run this on the real v6e VM
  for the TPU-specific answer -- this CPU-backend answer is a real, useful
  first look, not a substitute for it.
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kg, ku = jax.random.split(key, 3)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w_gate = (jax.random.normal(kg, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)
  w_up = (jax.random.normal(ku, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  unfused_fn = functools.partial(
      _reference_gateup_situ,
      beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA, round_dtype=jnp.bfloat16,
  )
  backend = jax.devices()[0].platform
  compiled = jax.jit(unfused_fn).lower(x, w_gate, w_up).compile()
  hlo_text = compiled.as_text()
  print(f"=== compiled HLO for _reference_gateup_situ, backend={backend}, m={m} ===")
  print(hlo_text)
  num_fusions = hlo_text.count("fusion(")
  num_dots = hlo_text.count(" dot(")
  print(
      f"=== summary: backend={backend} num_fusion_ops={num_fusions} "
      f"num_dot_ops={num_dots} ==="
  )
  return hlo_text


def _time_jit_pipelined(f, *args, num_repeats: int = 20) -> float:
  """Throughput-style timing (this project's established convention
  elsewhere, e.g. kimi_k3_latent_moe_ragged_dot.py's profiling functions):
  one warmup call (excludes compile time), then `num_repeats` calls
  submitted back-to-back with NO synchronization between them, one
  `block_until_ready` at the very end, total time / num_repeats.

  **2026-09-25, per a ChatGPT-relayed review the user brought back**: this
  measures pipelined THROUGHPUT, not per-call latency -- consecutive
  dispatches can overlap (the host issues call N+1 before call N's device
  work has finished), so the reported "ms" is not what a single, isolated,
  synchronous invocation would take. Kept alongside
  `_time_jit_blocking` (below) specifically so both conventions are
  reported and the difference between them is itself a measurement, not
  assumed away.
  """
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def _time_jit_blocking(f, *args, num_repeats: int = 20) -> float:
  """Per-call latency timing: blocks on `block_until_ready` after EVERY
  call, so each timed iteration genuinely waits for the previous one to
  finish before issuing the next -- the number a single synchronous
  request would actually see, unlike `_time_jit_pipelined`'s throughput
  number. Added 2026-09-25 alongside the pipelined version specifically to
  quantify how much of the previously-reported speedup gap was a
  measurement-methodology artifact vs. a real kernel difference.
  """
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
    jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def check(
    m: int,
    bm: int,
    bk: int,
    bn: int,
    dtype=jnp.bfloat16,
    acc_dtype: jnp.dtype = jnp.float32,
) -> bool:
  """Correctness check + fused-vs-unfused latency comparison. Correctness
  runs fine in interpret mode with no TPU -- mirrors 03_matmul_k_tiled.py's
  own check() structure. Uses real Kimi K3 per-expert dimensions
  (LATENT_SIZE=3584, INTERMEDIATE_SIZE=3072), only `m` (token count) and
  tile sizes vary across configs.

  The "unfused" baseline (`_reference_gateup_situ`) is plain, ordinary
  jax.jit-compiled XLA (two separate `x @ w` matmuls + a plain-JAX
  SiTU-GLU elementwise op) -- NOT tokamax's `ragged_dot`, since this
  kernel targets a single dense expert, not the ragged/grouped case
  `_local_shard_expert_ffn_ragged_dot`/`_local_shard_expert_ffn_ragged_dot_fused_gateup`
  in `kimi_k3_latent_moe_ragged_dot.py` already benchmark. Answers a
  narrower, more basic question than that file's comparison: does OUR
  hand-written kernel's fusion (no intermediate gate/up write to HBM) beat
  even a plain XLA-compiled unfused version at this dense (single-expert)
  scale, before ever bringing tokamax's own kernels or ragged dispatch
  into the picture at all.

  `acc_dtype=jnp.bfloat16` (2026-09-21 addition) uses a wider tolerance
  than the fp32-accumulator default -- accumulating in bf16 across
  `k // bk` K-steps genuinely compounds rounding error beyond the single
  round-at-the-end the fp32-accumulator kernel does, this is not just a
  test-calibration choice.

  **2026-09-25 fix, per a ChatGPT-relayed review**: the random key used to
  now depend on `hash((m, bm, bk, bn))` -- meaning every DIFFERENT tile
  config got DIFFERENT random `x`/`w_gate`/`w_up` data. The fused-vs-
  unfused comparison WITHIN one call was still apples-to-apples (both see
  the same data), but comparing SPEEDUP NUMBERS ACROSS different tile
  configs was comparing results on different underlying problems, not
  isolating the effect of tile choice alone. Fixed: the key now depends
  only on `m`, so every tile config at the same `m` sees byte-identical
  inputs.
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kg, ku = jax.random.split(key, 3)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w_gate = (jax.random.normal(kg, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)
  w_up = (jax.random.normal(ku, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  fused_fn = functools.partial(fused_gateup_situ, bm=bm, bk=bk, bn=bn, acc_dtype=acc_dtype)
  out = jax.jit(fused_fn)(x, w_gate, w_up)
  jax.block_until_ready(out)  # first call excluded from correctness check too, compile-only

  expected = _reference_gateup_situ(
      x, w_gate, w_up, SITU_BETA, SITU_LINEAR_BETA, jnp.bfloat16
  )
  diff = jnp.abs(out.astype(jnp.float32) - expected.astype(jnp.float32))
  max_abs_diff = float(jnp.max(diff))
  out_scale = float(jnp.std(expected.astype(jnp.float32))) + 1e-8
  relative_max_diff = max_abs_diff / out_scale
  has_nan = bool(jnp.any(jnp.isnan(out)))
  has_inf = bool(jnp.any(jnp.isinf(out)))
  tolerance = 0.05 if acc_dtype == jnp.float32 else 0.15
  ok = relative_max_diff < tolerance and not has_nan and not has_inf
  status = "OK" if ok else "FAIL"

  unfused_fn = functools.partial(
      _reference_gateup_situ,
      beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA, round_dtype=jnp.bfloat16,
  )

  fused_ms_pipe = _time_jit_pipelined(fused_fn, x, w_gate, w_up)
  unfused_ms_pipe = _time_jit_pipelined(unfused_fn, x, w_gate, w_up)
  speedup_pipe = unfused_ms_pipe / fused_ms_pipe

  fused_ms_block = _time_jit_blocking(fused_fn, x, w_gate, w_up)
  unfused_ms_block = _time_jit_blocking(unfused_fn, x, w_gate, w_up)
  speedup_block = unfused_ms_block / fused_ms_block

  print(
      f"[{status}] m={m} k={LATENT_SIZE} n={INTERMEDIATE_SIZE}"
      f" tile=(bm={bm},bk={bk},bn={bn}) acc_dtype={acc_dtype.__name__}"
      f" max_abs_diff={max_abs_diff:.4e} relative_max_diff={relative_max_diff:.4f}"
      f" tolerance={tolerance} has_nan={has_nan} has_inf={has_inf}\n"
      f"    [pipelined]  fused_ms={fused_ms_pipe:.4f} unfused_ms={unfused_ms_pipe:.4f} speedup={speedup_pipe:.3f}x\n"
      f"    [per-call]   fused_ms={fused_ms_block:.4f} unfused_ms={unfused_ms_block:.4f} speedup={speedup_block:.3f}x"
  )
  return ok


def compare_fused_configs_alternating(
    m: int = 2048,
    old_bm: int = 2048, old_bk: int = 512, old_bn: int = 1024,
    new_bm: int = 2048, new_bk: int = LATENT_SIZE, new_bn: int = 256,
    num_rounds: int = 12,
    dtype=jnp.bfloat16,
) -> dict:
  """Three-way controlled comparison (old fused config, new fused config,
  XLA baseline), per the user's explicit 2026-09-27 request: before
  reporting the ~97% no-K-tiling result as real, confirm it's stable under
  the same alternating-repeated-measurement discipline already used for
  the single-matmul K-tiling experiment (`own_single_matmul_pallas_kernel.py`'s
  `compare_bk_alternating`), not a one-shot number that could be noise or
  a lucky run.

  All three see the IDENTICAL input (same `m`, same fixed-per-m seed as
  `check()`). The round-robin order rotates every round (old->new->xla,
  new->xla->old, xla->old->new, repeating every 3 rounds) so no config is
  systematically favored by always running first (e.g. any residual
  warm-up effect) or last.

  Correctness is checked ONCE against the XLA reference per fused config
  (not per timing round -- a deterministic jit-compiled function's output
  doesn't change between calls), using the same relative-diff convention
  as `check()`.
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kg, ku = jax.random.split(key, 3)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w_gate = (jax.random.normal(kg, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)
  w_up = (jax.random.normal(ku, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  old_fn = jax.jit(functools.partial(fused_gateup_situ, bm=old_bm, bk=old_bk, bn=old_bn))
  new_fn = jax.jit(functools.partial(fused_gateup_situ, bm=new_bm, bk=new_bk, bn=new_bn))
  xla_fn = jax.jit(functools.partial(
      _reference_gateup_situ, beta=SITU_BETA, linear_beta=SITU_LINEAR_BETA, round_dtype=jnp.bfloat16,
  ))

  # Warm up (compile) all three before any timed round.
  old_out = old_fn(x, w_gate, w_up)
  new_out = new_fn(x, w_gate, w_up)
  xla_out = xla_fn(x, w_gate, w_up)
  jax.block_until_ready((old_out, new_out, xla_out))

  # Correctness reference: EAGER (not jax.jit-wrapped), matching check()'s
  # own established convention exactly -- 2026-09-27 fix, found by comparing
  # against check()'s own numbers for the identical (m, old_bm, old_bk,
  # old_bn) config: check() gets relative_max_diff~0.013 there (comfortably
  # under tolerance), but comparing against the JIT-compiled xla_fn instead
  # (as this function originally did) gave 0.0532 (just over 0.05) for BOTH
  # old and new configs -- identically, which is itself the tell that this
  # was a reference-computation artifact, not a real per-config numeric bug
  # (two genuinely different fused kernels landing on the exact same error
  # against a real ground truth would be an unlikely coincidence). jit vs.
  # eager execution of the same reference function can make different
  # internal fusion/precision decisions on real TPU hardware -- comparing
  # against the jitted version was comparing against a slightly different
  # (though not more "correct") computation than check()'s validated eager
  # baseline.
  expected = _reference_gateup_situ(
      x, w_gate, w_up, SITU_BETA, SITU_LINEAR_BETA, jnp.bfloat16
  )

  def _correctness(out, expected, label):
    diff = jnp.abs(out.astype(jnp.float32) - expected.astype(jnp.float32))
    max_abs_diff = float(jnp.max(diff))
    out_scale = float(jnp.std(expected.astype(jnp.float32))) + 1e-8
    relative_max_diff = max_abs_diff / out_scale
    has_nan = bool(jnp.any(jnp.isnan(out)))
    has_inf = bool(jnp.any(jnp.isinf(out)))
    ok = relative_max_diff < 0.05 and not has_nan and not has_inf
    print(
        f"[correctness: {label}] max_abs_diff={max_abs_diff:.4e} "
        f"relative_max_diff={relative_max_diff:.4f} has_nan={has_nan} has_inf={has_inf} "
        f"{'OK' if ok else 'FAIL'}"
    )
    return ok

  old_ok = _correctness(old_out, expected, f"old (bk={old_bk},bn={old_bn})")
  new_ok = _correctness(new_out, expected, f"new (bk={new_bk},bn={new_bn})")

  fns = {"old": old_fn, "new": new_fn, "xla": xla_fn}
  times = {"old": [], "new": [], "xla": []}
  order_cycle = [["old", "new", "xla"], ["new", "xla", "old"], ["xla", "old", "new"]]

  for round_idx in range(num_rounds):
    order = order_cycle[round_idx % 3]
    round_times = {}
    for name in order:
      t0 = time.perf_counter()
      jax.block_until_ready(fns[name](x, w_gate, w_up))
      round_times[name] = (time.perf_counter() - t0) * 1000
    for name in order:
      times[name].append(round_times[name])
    print(
        f"[round {round_idx}] order={order} "
        f"old={round_times['old']:.4f}ms new={round_times['new']:.4f}ms xla={round_times['xla']:.4f}ms"
    )

  def _stats(vals):
    n = len(vals)
    mean = sum(vals) / n
    var = sum((v - mean) ** 2 for v in vals) / n
    return {"mean": mean, "std": var ** 0.5, "min": min(vals), "max": max(vals)}

  stats = {name: _stats(vals) for name, vals in times.items()}
  print(
      f"\n[summary] m={m} num_rounds={num_rounds} "
      f"old=(bm={old_bm},bk={old_bk},bn={old_bn}) new=(bm={new_bm},bk={new_bk},bn={new_bn})\n"
      f"  old: mean={stats['old']['mean']:.4f}ms std={stats['old']['std']:.4f}ms "
      f"min={stats['old']['min']:.4f}ms max={stats['old']['max']:.4f}ms\n"
      f"  new: mean={stats['new']['mean']:.4f}ms std={stats['new']['std']:.4f}ms "
      f"min={stats['new']['min']:.4f}ms max={stats['new']['max']:.4f}ms\n"
      f"  xla: mean={stats['xla']['mean']:.4f}ms std={stats['xla']['std']:.4f}ms "
      f"min={stats['xla']['min']:.4f}ms max={stats['xla']['max']:.4f}ms\n"
      f"  new_vs_xla_speedup={stats['xla']['mean'] / stats['new']['mean']:.3f}x "
      f"old_vs_xla_speedup={stats['xla']['mean'] / stats['old']['mean']:.3f}x "
      f"new_vs_old_speedup={stats['old']['mean'] / stats['new']['mean']:.3f}x\n"
      f"  correctness: old={'OK' if old_ok else 'FAIL'} new={'OK' if new_ok else 'FAIL'}"
  )
  return {"times": times, "stats": stats, "old_ok": old_ok, "new_ok": new_ok}


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  if any(d.platform == "tpu" for d in jax.devices()):
    print("Real TPU detected -- fused/unfused timing below reflects actual hardware.")
  else:
    print(
        "No TPU detected -- running in interpret mode. Correctness numbers "
        "are still meaningful (genuine Pallas semantics), but timing/speedup "
        "numbers below are interpreter overhead, not real hardware performance."
    )
  configs = [
      # (m, bm, bk, bn) -- 3584 = 512*7 = 256*14 = 128*28; 3072 = 512*6 = 256*12 = 128*24
      (128, 128, 512, 512),
      (2048, 256, 512, 512),
      (2048, 128, 256, 256),
      # 2026-09-21, real hardware showed the above are ALL slower than the
      # unfused XLA baseline (0.07x-0.73x, not a speedup) -- suspected cause:
      # bn < INTERMEDIATE_SIZE means n_tiles > 1, and the grid's (i, j, k)
      # iteration order re-fetches the SAME x tile (which only depends on
      # (i, k), not j) once per j value -- pure redundant HBM traffic that
      # grows with n_tiles. bn=INTERMEDIATE_SIZE (n_tiles=1) eliminates this
      # entirely; testing whether that alone closes the gap.
      (2048, 128, 512, INTERMEDIATE_SIZE),
      (2048, 256, 512, INTERMEDIATE_SIZE),
      # 2026-09-21, second round: bn=INTERMEDIATE_SIZE improved things
      # (0.299x -> 0.474x) but still lost to unfused -- second suspected
      # redundant-reload source: w_gate/w_up's index_map depends on (k, j)
      # only, not i (the m-tile index), which is OUTERMOST/slowest-varying
      # in the grid -- so for m_tiles>1, the ENTIRE K-sweep of weight tiles
      # gets re-fetched once per m-tile, and weights (22MB each, full size)
      # are far more expensive to reload than x. Fewer, bigger bm ->
      # fewer m_tiles -> less redundant weight reload, at the cost of a
      # bigger VMEM accumulator (bm*bn*4bytes*2, two accumulators) -- testing
      # where that tradeoff actually lands on real hardware, not guessing.
      (2048, 512, 512, INTERMEDIATE_SIZE),
      # 2026-09-21, third round: bm=1024/bn=INTERMEDIATE_SIZE(3072) OOM'd on
      # real hardware -- "Scoped allocation with size 50.00M and limit 32.00M
      # exceeded scoped vmem limit by 18.00M." A REAL, confirmed scoped-VMEM
      # ceiling of 32MB for this kernel (not a guess). bm=512/bn=3072 (this
      # file's best working config so far, 0.771x) fits comfortably under
      # that. Total redundant-reload HBM traffic, given weight data (2 x
      # 3584x3072x2 bytes = 44.04MB) is reloaded once per m_tile and x data
      # (2048x3584x2 bytes = 14.68MB) is reloaded once per n_tile: minimizing
      # m_tiles matters far more than minimizing n_tiles, since weight reload
      # is ~3x more expensive per tile than x reload. These three configs
      # trade a bit more n_tiles for far fewer m_tiles, while staying under
      # the confirmed 32MB scoped-VMEM ceiling (rough estimate:
      # 10*bm*bn + 8*bk*bn + 4*bm*bk bytes, calibrated against the two data
      # points above -- not exact, real hardware will confirm or refute it).
      (2048, 1024, 512, 1536),  # m_tiles=2, n_tiles=2 -- est. ~24MB
      (2048, 2048, 512, 1024),  # m_tiles=1 (!), n_tiles=3 -- est. ~29MB
      (2048, 2048, 512, 768),   # m_tiles=1, n_tiles=4 -- est. ~23MB, safety margin if 1024 above is too tight
      # 2026-09-21, fourth round: bm=2048/bn=1024 (m_tiles=1, n_tiles=3) is
      # the best so far (0.879x-0.888x across runs, up from an initial
      # 0.07x-0.73x -- real progress, still short of actually beating
      # unfused). m_tiles is already at its floor (1); the remaining lever
      # is n_tiles.
      #
      # bm=2048/bn=1536 (n_tiles=2) CONFIRMED OOM on real hardware: "Scoped
      # allocation with size 46.00M and limit 32.00M exceeded... by 14.00M"
      # -- close to the ~42MB the rough estimate predicted.
      #
      # bm=2048/bk=896/bn=1024 (bigger K-tile at the current-best n config)
      # ALSO confirmed OOM: "Scoped allocation with size 38.00M and limit
      # 32.00M exceeded... by 6.00M". Together these two real OOMs show the
      # two fp32 accumulators alone (bm*bn*4bytes*2 = 2048*1024*4*2 = 16MB
      # at the current best config) already occupy roughly half the 32MB
      # ceiling -- there's little headroom left for a bigger bn OR a
      # bigger bk at bm=2048, regardless of which one is grown. Both
      # removed from this list rather than re-run; (2048, 2048, 512, 1024)
      # above is the practical ceiling this simple (one fixed tile size per
      # dimension, fp32 accumulators) tuning approach reaches -- 0.879x-
      # 0.888x across runs, up from an initial 0.07x-0.73x. Going further
      # would need a different lever (e.g. bf16 accumulators, at some
      # precision cost, to roughly halve accumulator VMEM and see if that
      # reopens the bn=1536/bk=896 options) rather than more of the same
      # fixed-tile-size search.
  ]
  results = [check(*c) for c in configs]
  assert all(results), (
      "some config's result is wrong -- go back and check the two-accumulator "
      "K-tiling logic / pl.when boundary conditions / bf16-round-then-upcast order"
  )
  print(f"all {len(results)} shape/tile configs passed")

  # 2026-09-21, fifth round: bf16 accumulators (see fused_gateup_situ_kernel's
  # docstring) halve accumulator VMEM -- testing whether that reopens the two
  # configs that OOM'd with fp32 accumulators (bn=1536, bk=896), plus the
  # current-best config as a same-shape sanity check (does bf16 accumulation
  # cost real speed/precision even where fp32 already fit?). If neither OOM'd
  # config now fits, or fits but doesn't beat the fp32-accumulator best
  # (0.879x-0.888x), this tuning approach has genuinely hit its ceiling and
  # the honest conclusion is "close but not a clean win," not something to
  # keep chasing.
  print("\n--- bf16-accumulator experiment ---")
  bf16acc_configs = [
      (2048, 2048, 512, 1024),  # current best, fp32-acc baseline for comparison
      (2048, 2048, 512, 1536),  # OOM'd with fp32 acc (46MB) -- does bf16 acc fit now?
      (2048, 2048, 896, 1024),  # OOM'd with fp32 acc (38MB) -- does bf16 acc fit now?
  ]
  bf16acc_results = [check(*c, acc_dtype=jnp.bfloat16) for c in bf16acc_configs]
  print(
      f"{sum(bf16acc_results)}/{len(bf16acc_results)} bf16-accumulator configs passed "
      "correctness (see speedup numbers above for whether any of this actually helped)"
  )

  # 2026-09-27, sixth round: back to the actual goal (the fused kernel, not
  # the single-matmul control experiment) -- own_single_matmul_pallas_kernel.py's
  # controlled bk comparison (10 alternating rounds, same bm/bn, only bk
  # varied) found a real, consistent ~32.6us device-side saving from
  # bk=LATENT_SIZE (no K-loop) over bk=512 (7 K-steps). Testing whether that
  # carries over to the TWO-accumulator fused kernel -- it has roughly 2x
  # the accumulator VMEM footprint of the single-matmul kernel at the same
  # (bm, bn), so the same bn=512 that worked there may not fit here at all;
  # bn=256 chosen as a conservative first attempt (est. ~23.6MB) rather than
  # re-discovering the ceiling one OOM at a time again.
  print("\n--- full-K (no K-tiling) fused-kernel experiment ---")
  fullk_ok = check(2048, 2048, LATENT_SIZE, 256)
  print(
      f"full-K fused-kernel config {'PASSED' if fullk_ok else 'FAILED'} -- "
      "see speedup numbers above; if this OOM'd instead, the single-matmul "
      "improvement does NOT carry over as-is to the two-accumulator case, "
      "and VMEM headroom (not the K-loop itself) is the next problem to solve."
  )
