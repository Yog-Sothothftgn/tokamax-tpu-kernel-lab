"""Control experiment (2026-09-25), per a ChatGPT-relayed debugging plan
the user brought back after `own_gateup_situ_pallas_kernel.py`'s hand-
written fused kernel came in behind the unfused XLA baseline (best found:
~0.88x, not a speedup) even after 5 rounds of real-hardware-guided tile
tuning: isolate whether the shortfall is in the BASIC matmul tiling/
scheduling itself, or specifically in the two-accumulator fusion/
activation stage, by testing the two pieces separately instead of only
ever measuring them combined.

This file tests ONLY `Y = X @ W` (ONE matmul, no gate/up, no fused
activation) at the SAME real Kimi K3 per-expert dimensions and the SAME
tile sizes already explored in `own_gateup_situ_pallas_kernel.py`, against
plain XLA-compiled `X @ W`. Two possible outcomes, each pointing somewhere
different:

  - If even this single matmul loses to XLA: the problem is in basic tile
    scheduling / data movement / accumulation -- nothing to do with fusion
    at all, and `own_gateup_situ_pallas_kernel.py`'s tuning direction
    (chasing redundant-reload reduction) was addressing a real but
    secondary effect on top of a more fundamental gap.
  - If this single matmul is competitive with (or beats) XLA, but the
    two-accumulator fused kernel still loses: the problem is specifically
    in the SECOND accumulator's extra VMEM pressure and/or the fused
    activation stage, not in this project's basic Pallas tiling ability.

Structurally identical to `01_pallas_basics/03_matmul_k_tiled.py` (Task
A) -- this is that exact template, just re-run at Kimi K3's real
dimensions and tile sizes instead of the toy shapes that file used, for a
true apples-to-apples comparison against the fused kernel's own tile
choices.

To run (local, interpret mode, no TPU needed for correctness; needs the
v6e TPU VM for real timing numbers):
  python own_single_matmul_pallas_kernel.py
"""

import functools
import time

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import tpu as pltpu

# Same real Kimi K3 per-expert dimensions as own_gateup_situ_pallas_kernel.py.
LATENT_SIZE = 3584
INTERMEDIATE_SIZE = 3072


def matmul_kernel(x_ref, w_ref, o_ref, acc_ref, *, num_k_tiles: int):
  """Identical structure to 03_matmul_k_tiled.py's own kernel -- ONE
  accumulator, ONE rhs matrix. See that file for the line-by-line
  explanation of the pl.when/scratch-accumulator pattern.
  """
  k_id = pl.program_id(2)

  @pl.when(k_id == 0)
  def _():
    acc_ref[...] = jnp.zeros_like(acc_ref)

  acc_ref[...] += jnp.dot(x_ref[...], w_ref[...], preferred_element_type=jnp.float32)

  @pl.when(k_id == num_k_tiles - 1)
  def _():
    o_ref[...] = acc_ref[...].astype(o_ref.dtype)


def pallas_matmul(
    x: jax.Array, w: jax.Array, *, bm: int = 128, bk: int = 512, bn: int = 512
) -> jax.Array:
  m, k = x.shape
  k2, n = w.shape
  assert k == k2, f"x/w inner dims don't match: {x.shape} @ {w.shape}"
  assert m % bm == 0 and n % bn == 0 and k % bk == 0, (
      "keep the exercise simple: only support shape/tile combinations that "
      "divide evenly for now"
  )

  num_k_tiles = k // bk
  has_tpu = any(d.platform == "tpu" for d in jax.devices())

  return pl.pallas_call(
      functools.partial(matmul_kernel, num_k_tiles=num_k_tiles),
      grid=(m // bm, n // bn, num_k_tiles),
      in_specs=[
          pl.BlockSpec((bm, bk), lambda i, j, kk: (i, kk)),
          pl.BlockSpec((bk, bn), lambda i, j, kk: (kk, j)),
      ],
      out_specs=pl.BlockSpec((bm, bn), lambda i, j, kk: (i, j)),
      out_shape=jax.ShapeDtypeStruct((m, n), x.dtype),
      scratch_shapes=[pltpu.VMEM((bm, bn), jnp.float32)],
      compiler_params=(
          pltpu.CompilerParams(dimension_semantics=("parallel", "parallel", "arbitrary"))
          if has_tpu
          else None
      ),
      interpret=not has_tpu,
  )(x, w)


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
  """Per-call latency timing -- blocks after every call. See
  own_gateup_situ_pallas_kernel.py's identically-named function."""
  f_jit = jax.jit(f)
  out = f_jit(*args)
  jax.block_until_ready(out)
  t0 = time.perf_counter()
  for _ in range(num_repeats):
    out = f_jit(*args)
    jax.block_until_ready(out)
  return (time.perf_counter() - t0) / num_repeats * 1000


def check(m: int, bm: int, bk: int, bn: int, dtype=jnp.bfloat16) -> bool:
  """Correctness + fused(pallas)-vs-XLA latency, both timing conventions.
  Fixed seed depends only on `m` (not tile params) -- same fix applied to
  own_gateup_situ_pallas_kernel.py's check() for the same reason: tile
  configs must be compared on IDENTICAL underlying data.
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kw = jax.random.split(key, 2)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w = (jax.random.normal(kw, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  pallas_fn = functools.partial(pallas_matmul, bm=bm, bk=bk, bn=bn)
  out = jax.jit(pallas_fn)(x, w)
  jax.block_until_ready(out)

  expected = (x @ w).astype(dtype)
  diff = jnp.abs(out.astype(jnp.float32) - expected.astype(jnp.float32))
  max_abs_diff = float(jnp.max(diff))
  out_scale = float(jnp.std(expected.astype(jnp.float32))) + 1e-8
  relative_max_diff = max_abs_diff / out_scale
  has_nan = bool(jnp.any(jnp.isnan(out)))
  has_inf = bool(jnp.any(jnp.isinf(out)))
  tolerance = 0.05
  ok = relative_max_diff < tolerance and not has_nan and not has_inf
  status = "OK" if ok else "FAIL"

  xla_fn = lambda xx, ww: (xx @ ww).astype(dtype)

  pallas_ms_pipe = _time_jit_pipelined(pallas_fn, x, w)
  xla_ms_pipe = _time_jit_pipelined(xla_fn, x, w)
  speedup_pipe = xla_ms_pipe / pallas_ms_pipe

  pallas_ms_block = _time_jit_blocking(pallas_fn, x, w)
  xla_ms_block = _time_jit_blocking(xla_fn, x, w)
  speedup_block = xla_ms_block / pallas_ms_block

  print(
      f"[{status}] m={m} k={LATENT_SIZE} n={INTERMEDIATE_SIZE}"
      f" tile=(bm={bm},bk={bk},bn={bn})"
      f" max_abs_diff={max_abs_diff:.4e} relative_max_diff={relative_max_diff:.4f}"
      f" has_nan={has_nan} has_inf={has_inf}\n"
      f"    [pipelined]  pallas_ms={pallas_ms_pipe:.4f} xla_ms={xla_ms_pipe:.4f} speedup={speedup_pipe:.3f}x\n"
      f"    [per-call]   pallas_ms={pallas_ms_block:.4f} xla_ms={xla_ms_block:.4f} speedup={speedup_block:.3f}x"
  )
  return ok


_TPU_V6E_PEAK_BF16_TFLOPS = 918.0  # same public spec expert_ffn_roofline.py already uses
_TPU_V6E_HBM_BANDWIDTH_GBPS = 1638.0


def single_matmul_roofline(m: int, k: int = LATENT_SIZE, n: int = INTERMEDIATE_SIZE) -> dict:
  """Theoretical compute-bound and memory-bandwidth-bound floors for ONE
  `(m,k) @ (k,n)` matmul, same methodology as `expert_ffn_roofline.py`'s
  `stage_c_roofline` (public TPU v6e specs, not independently verified
  hardware queries -- see that file's own caveat). Added 2026-09-26,
  per a ChatGPT-relayed request to answer "is the gap from waiting on
  data, or from compute/scheduling itself" with a real calculation
  instead of continuing to guess.
  """
  bf16_bytes = 2
  flops = 2 * m * k * n  # multiply-add counted as 2 ops
  bytes_moved = (m * k + k * n + m * n) * bf16_bytes  # read X, read W, write Y

  compute_bound_ms = flops / (_TPU_V6E_PEAK_BF16_TFLOPS * 1e12) * 1000
  memory_bound_ms = bytes_moved / (_TPU_V6E_HBM_BANDWIDTH_GBPS * 1e9) * 1000
  roofline_ms = max(compute_bound_ms, memory_bound_ms)
  return {
      "m": m, "k": k, "n": n,
      "flops": flops, "bytes_moved": bytes_moved,
      "compute_bound_ms": compute_bound_ms,
      "memory_bound_ms": memory_bound_ms,
      "roofline_ms": roofline_ms,
      "bound_type": "memory" if memory_bound_ms >= compute_bound_ms else "compute",
  }


def report_roofline_efficiency(
    m: int, pallas_ms: float, xla_ms: float, label: str = ""
) -> None:
  """Prints roofline efficiency for a REAL measured (pallas_ms, xla_ms)
  pair -- pass in real numbers from `check()`'s printed output (this
  function does no measuring itself, it only interprets numbers already
  measured on real hardware). Answers, with an actual calculation: is
  either implementation memory-bandwidth-bound at this shape (expect
  measured time close to memory_bound_ms if so), and how much of the
  pallas-vs-xla gap is a real efficiency difference vs. shared overhead
  both pay equally.
  """
  r = single_matmul_roofline(m)
  pallas_eff = r["roofline_ms"] / pallas_ms
  xla_eff = r["roofline_ms"] / xla_ms
  gap_us = (pallas_ms - xla_ms) * 1000
  print(
      f"[roofline{f' ({label})' if label else ''}] m={m} bound={r['bound_type']} "
      f"compute_floor={r['compute_bound_ms']:.4f}ms memory_floor={r['memory_bound_ms']:.4f}ms "
      f"xla_efficiency={xla_eff:.1%} pallas_efficiency={pallas_eff:.1%} "
      f"pallas_minus_xla_gap={gap_us:.1f}us "
      f"pallas_gap_from_ideal={(pallas_ms - r['roofline_ms']) * 1000:.1f}us "
      f"xla_gap_from_ideal={(xla_ms - r['roofline_ms']) * 1000:.1f}us"
  )


def inspect_pallas_matmul_hlo(m: int, bm: int, bk: int, bn: int, dtype=jnp.bfloat16) -> str:
  """Compiled HLO for our OWN pallas_matmul, mirroring
  own_gateup_situ_pallas_kernel.py's inspect_unfused_hlo -- but a Pallas
  kernel appears in the OUTER HLO graph as a single opaque
  `custom-call(..., custom_call_target="tpu_custom_call")`; this will NOT
  reveal the accumulator/K-loop structure INSIDE the kernel (that's
  compiled by Mosaic into its own lower-level representation, not part of
  the surrounding XLA HLO). What it DOES reveal: input/output layouts,
  any surrounding copy/bitcast ops XLA inserts around the custom-call, and
  (in the backend_config, same as the earlier OOM error messages) the
  actual scoped-VMEM size XLA/Mosaic allocated for this exact tile config.
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kw = jax.random.split(key, 2)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w = (jax.random.normal(kw, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  pallas_fn = functools.partial(pallas_matmul, bm=bm, bk=bk, bn=bn)
  backend = jax.devices()[0].platform
  compiled = jax.jit(pallas_fn).lower(x, w).compile()
  hlo_text = compiled.as_text()
  print(f"=== compiled HLO for pallas_matmul, backend={backend}, m={m}, tile=(bm={bm},bk={bk},bn={bn}) ===")
  print(hlo_text)
  return hlo_text


def profile_best_tile_device_trace(
    m: int = 2048,
    bm: int = 2048,
    bk: int = 512,
    bn: int = 1024,
    trace_dir: str = "/tmp/single_matmul_trace",
    num_repeats: int = 20,
    dtype=jnp.bfloat16,
) -> None:
  """Captures a real `jax.profiler` device trace for BOTH pallas_matmul
  and the XLA baseline at the SAME fixed (best) tile config -- step 3 of
  the ChatGPT-relayed plan (2026-09-26): "fix the best tile, compare
  compiled results AND device timeline; locate where it's slow before
  changing any more tile parameters."

  This does NOT parse the trace itself (XLA's trace format is a
  protobuf-based xplane, not simple text) -- it writes a real trace to
  `trace_dir`, viewable via:
    tensorboard --logdir=<trace_dir>
  then open the "Trace Viewer" tab (needs port-forwarding from the VM, or
  copying trace_dir back via gcloud compute scp, same as this project's
  established WP4 dispatch-attribution trace workflow --
  profile_dispatch_host_device_attribution in kimi_k3_latent_moe_ragged_dot.py
  already does exactly this pattern). Look for: how much of the total
  wall-clock is inside the `tpu_custom_call`/fusion op itself (device
  compute+DMA) vs. host-side dispatch, and whether DMA-copy ops show
  meaningful duration next to the matmul op (would indicate real data-
  movement cost, not just scheduling).
  """
  key = jax.random.key(hash(m) % (2**31))
  kx, kw = jax.random.split(key, 2)
  scale = 0.02
  x = (jax.random.normal(kx, (m, LATENT_SIZE)) * scale).astype(dtype)
  w = (jax.random.normal(kw, (LATENT_SIZE, INTERMEDIATE_SIZE)) * scale).astype(dtype)

  pallas_fn = jax.jit(functools.partial(pallas_matmul, bm=bm, bk=bk, bn=bn))
  xla_fn = jax.jit(lambda xx, ww: (xx @ ww).astype(dtype))

  # Warm up (excludes compile time from the trace) before tracing either.
  jax.block_until_ready(pallas_fn(x, w))
  jax.block_until_ready(xla_fn(x, w))

  print(f"Writing device trace to {trace_dir} ...")
  with jax.profiler.trace(trace_dir):
    with jax.profiler.TraceAnnotation("pallas_matmul_repeats"):
      for _ in range(num_repeats):
        out = pallas_fn(x, w)
      jax.block_until_ready(out)
    with jax.profiler.TraceAnnotation("xla_matmul_repeats"):
      for _ in range(num_repeats):
        out = xla_fn(x, w)
      jax.block_until_ready(out)
  print(
      f"Trace written to {trace_dir} -- view with `tensorboard --logdir={trace_dir}` "
      "(Trace Viewer tab), or copy the whole directory back via gcloud compute scp. "
      "Look for the pallas_matmul_repeats vs xla_matmul_repeats annotated regions and "
      "how much of each is device compute/DMA vs host dispatch gaps."
  )


if __name__ == "__main__":
  print("devices:", jax.devices())
  print("jax version:", jax.__version__)
  if any(d.platform == "tpu" for d in jax.devices()):
    print("Real TPU detected -- pallas/XLA timing below reflects actual hardware.")
  else:
    print(
        "No TPU detected -- running in interpret mode. Correctness numbers "
        "are still meaningful, but timing/speedup numbers below are "
        "interpreter overhead, not real hardware performance."
    )
  # Same tile points already explored for the FUSED (two-matmul) kernel in
  # own_gateup_situ_pallas_kernel.py, re-tested here on a SINGLE matmul --
  # if these numbers alone already lose to XLA, the fused kernel's
  # shortfall isn't (only) about fusion/redundant-reload; if these are
  # competitive, the two-accumulator/activation stage is the real culprit.
  configs = [
      (2048, 256, 512, 512),     # the fused kernel's early, badly-losing config
      (2048, 512, 512, INTERMEDIATE_SIZE),   # fused kernel's 0.77x-0.81x point
      (2048, 2048, 512, 1024),   # fused kernel's best point (0.88x)
      # 2026-09-26: device trace confirmed a real ~12.58us device-only gap
      # at the best config above, with DMA/copy time confirmed negligible
      # (~0.74us) -- ruling out data movement, but NOT yet ruling out the
      # K-loop (7 steps at bk=512) itself as the source. bk=LATENT_SIZE
      # (num_k_tiles=1, no accumulator loop at all -- the whole K reduction
      # happens in a single step) isolates that specific hypothesis
      # directly: if the device-only gap shrinks or disappears here, the
      # K-loop structure was a real contributor; if it doesn't change,
      # something else (single-shot MXU utilization, output write pattern)
      # is the cause.
      (2048, 2048, LATENT_SIZE, 1024),
  ]
  results = [check(*c) for c in configs]
  assert all(results), (
      "some config's result is wrong -- go back and check the single-accumulator "
      "K-tiling logic / pl.when boundary conditions"
  )
  print(f"all {len(results)} shape/tile configs passed")
