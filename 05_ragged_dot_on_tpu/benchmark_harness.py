"""WP1 starting point: a benchmark harness for tokamax.ragged_dot.

**Not reinvented from scratch.** After reading tokamax's own `benchmarking.py`,
`docs/benchmarking.md`, and `tokamax/benchmarks/ragged_dot.py`, most of what
WP1/WP2 ask for is already built into tokamax:

  - `tokamax.standardize_function` + `tokamax.benchmark`: handles input
    construction, separates compile time from steady-state execution timing
    (multiple measured iterations), and **returns `peak_memory_mb` (a real
    memory metric)** as part of the returned `BenchmarkData`. Tokamax's own
    docs recommend `method="xprof_hermetic"` on TPU (hardware-level timing via
    XProf, avoiding Python-overhead noise); plain wallclock timing is not
    recommended on TPU. (The docs' spelling is wrong, though — the actually
    registered key is `"hermetic_xprof"`, confirmed by a real `KeyError` on
    v6e; that's what the code below uses.)
  - `tokamax.generate_ragged_dot_group_sizes(m, num_groups, p, seed)`:
    generates all 5 required group distributions (uniform / mild skew / heavy
    skew / empty groups / single dominant) just by passing different `p`
    vectors — no need to hand-write distribution generators.
  - The "Large" shape doesn't need to be invented: `tokamax/benchmarks/
    ragged_dot.py` already ships a real MaxText DeepSeek-v3 shape
    (`lhs: (262144, 7168)`, `rhs: (256, 7168, 2048)`, `group_sizes: (256,)`,
    bf16), reused here directly (still worth confirming with Zifan that this
    is representative of what he cares about, but it's no longer a blocker).

The file itself is therefore thin: it just assembles the shape matrix x 5
distributions x {xla, mosaic} and calls `tokamax.benchmark` once per
combination, then prints/collects the results.

Also runs a correctness check (xla vs mosaic output diff, dtype-dependent
tolerance) before the performance sweep, on small/medium shapes only (the
Large/DeepSeek shape is skipped for correctness to avoid an expensive extra
pass over multi-GB arrays; its correctness is already covered more broadly by
tokamax's own test suite). Every mosaic run so far has hit "Autotuning cache
miss" (heuristic config, not tuned), and only `implementation="mosaic"` (v1)
was ever tested — pass --fair-baseline to run WP3.5.1: a cheap, single-shape
comparison of xla / mosaic v1 / mosaic v2, each heuristic AND tuned, before
committing to a full re-run of the matrix with a specific Mosaic version.

Usage:
  python benchmark_harness.py          # local: only implementation="xla",
                                        # automatically skips the Large case
                                        # (262144 rows is too slow/memory-heavy
                                        # locally). This only validates that
                                        # the harness code itself runs; it
                                        # does not represent real performance.
  (on real v6e) python benchmark_harness.py --real
                                        # real hardware: xla + mosaic, every
                                        # shape and distribution — this is
                                        # where the meaningful numbers come
                                        # from.
  python benchmark_harness.py --real --fair-baseline
                                        # also runs the WP3.5.1 fair-baseline
                                        # comparison (one shape only; costs
                                        # extra time)
"""

import dataclasses
import functools
import sys

import jax

# --- Environment workaround (2026-08-16) ---
# The installed flax (via qwix's transitive dependency on flax.nnx, which
# tokamax pulls in but we never actually use here) expects
# `jax.experimental.hijax.MutableHiType`, which the installed jax build
# doesn't provide (a real upstream version mismatch between jax and flax's
# latest releases, not something fixable by picking a different pinned
# version — both packages' latest releases were tried and are incompatible
# with each other). We never touch flax/qwix's quantization functionality,
# so a harmless stub lets the import chain complete without changing any
# behavior our code actually relies on.
import jax.experimental.hijax as _hijax  # noqa: E402

if not hasattr(_hijax, "MutableHiType"):

  class _MutableHiTypeStub:
    pass

  _hijax.MutableHiType = _MutableHiTypeStub

import jax.numpy as jnp  # noqa: E402
import tokamax  # noqa: E402


@dataclasses.dataclass(frozen=True)
class ShapeCase:
  name: str
  m: int
  k: int
  n: int
  g: int
  dtype: jnp.dtype = jnp.float32
  # Shapes too big/slow to run locally (e.g. real DeepSeek-v3 scale) are
  # flagged True and skipped by the local smoke test.
  skip_locally: bool = False


SHAPE_CASES = [
    ShapeCase("small", m=512, k=256, n=384, g=4),
    ShapeCase("medium_a", m=2048, k=512, n=512, g=8),
    ShapeCase("medium_b", m=4096, k=1024, n=1024, g=16),
    # Real shape, from the MaxText DeepSeek-v3 example in
    # tokamax/benchmarks/ragged_dot.py.
    ShapeCase(
        "deepseek_v3_maxtext",
        m=262144,
        k=7168,
        n=2048,
        g=256,
        dtype=jnp.bfloat16,
        skip_locally=True,
    ),
]


def _mild_skew_p(g: int) -> list[float]:
  p = [1.0] * g
  p[0] = 2.0  # Group 0 gets roughly 2x the average token count.
  s = sum(p)
  return [x / s for x in p]


def _heavy_skew_p(g: int) -> list[float]:
  p = [1.0] * g
  p[0] = float(g)  # Group 0 absorbs the vast majority of tokens.
  s = sum(p)
  return [x / s for x in p]


def _empty_groups_p(g: int) -> list[float]:
  half = max(g // 2, 1)
  p = [1.0] * half + [0.0] * (g - half)
  s = sum(p)
  return [x / s for x in p]


def _single_dominant_p(g: int) -> list[float]:
  p = [0.0] * g
  p[0] = 1.0
  return p


# Distribution name -> function producing a `p` vector; uniform uses p=None
# (tokamax's own default, evenly split).
DISTRIBUTIONS = {
    "uniform": lambda g: None,
    "mild_skew": _mild_skew_p,
    "heavy_skew": _heavy_skew_p,
    "empty_groups": _empty_groups_p,
    "single_dominant": _single_dominant_p,
}


def check_correctness(
    shape: ShapeCase,
    dist_name: str,
    *,
    seed: int = 0,
) -> bool:
  """Compares concrete xla vs mosaic outputs for one (shape, distribution).

  This is the WP2 correctness requirement (max abs/rel error, pass/fail) that
  `run_one`'s performance benchmarking doesn't cover on its own — benchmarking
  xla and mosaic separately never actually diffs their outputs against each
  other. Tolerance is dtype-dependent per WP2 ("don't use one threshold for
  all dtypes"): bf16 gets a looser tolerance than float32.
  """
  p = DISTRIBUTIONS[dist_name](shape.g)
  group_sizes = jnp.array(
      tokamax.generate_ragged_dot_group_sizes(
          m=shape.m, num_groups=shape.g, p=p, seed=seed
      ),
      dtype=jnp.int32,
  )

  key = jax.random.key(seed)
  k_lhs, k_rhs = jax.random.split(key)
  lhs = jax.random.normal(k_lhs, (shape.m, shape.k), dtype=jnp.float32).astype(
      shape.dtype
  )
  rhs = jax.random.normal(
      k_rhs, (shape.g, shape.k, shape.n), dtype=jnp.float32
  ).astype(shape.dtype)

  out_xla = tokamax.ragged_dot(lhs, rhs, group_sizes, implementation="xla")
  out_mosaic = tokamax.ragged_dot(
      lhs, rhs, group_sizes, implementation="mosaic"
  )

  out_xla32 = out_xla.astype(jnp.float32)
  out_mosaic32 = out_mosaic.astype(jnp.float32)
  abs_err = float(jnp.max(jnp.abs(out_xla32 - out_mosaic32)))
  denom = max(float(jnp.max(jnp.abs(out_xla32))), 1e-8)
  rel_err = abs_err / denom

  tol = 2e-2 if shape.dtype == jnp.bfloat16 else 1e-4
  ok = rel_err < tol
  status = "OK" if ok else "FAIL"
  print(
      f"[correctness][{status}] {shape.name} dist={dist_name} "
      f"abs_err={abs_err:.3e} rel_err={rel_err:.3e} tol={tol:.0e}"
  )
  return ok


def run_one(
    shape: ShapeCase,
    dist_name: str,
    implementation: str,
    *,
    seed: int = 0,
) -> tokamax.BenchmarkData:
  """Benchmarks one (shape, group distribution, implementation) combination."""
  p = DISTRIBUTIONS[dist_name](shape.g)
  group_sizes = jnp.array(
      tokamax.generate_ragged_dot_group_sizes(
          m=shape.m, num_groups=shape.g, p=p, seed=seed
      ),
      dtype=jnp.int32,
  )

  lhs = jax.ShapeDtypeStruct((shape.m, shape.k), shape.dtype)
  rhs = jax.ShapeDtypeStruct((shape.g, shape.k, shape.n), shape.dtype)

  fn = functools.partial(tokamax.ragged_dot, implementation=implementation)
  # lhs/rhs are abstract ShapeDtypeStructs and get randomly initialized;
  # group_sizes is a concrete array and is kept as-is — so the group
  # distribution is exactly what we specify, while activations are random but
  # reproducible (fixed seed).
  f_std, args = tokamax.standardize_function(fn, lhs, rhs, group_sizes, seed=seed)

  has_tpu = any(d.platform == "tpu" for d in jax.devices())
  # Recommended on TPU; falls back to the library default (wallclock) locally.
  # Note: tokamax's docs/benchmarking.md says "xprof_hermetic", but the actual
  # registered key in tokamax._src.benchmarking's TimingMethod Literal is
  # "hermetic_xprof" (reversed word order) — confirmed by a real KeyError on
  # real v6e hardware; trust the runtime behavior over the docs here.
  method = "hermetic_xprof" if has_tpu else None

  return tokamax.benchmark(jax.jit(f_std), args, method=method)


def run_fair_baseline(shape: ShapeCase, dist_name: str, *, seed: int = 0):
  """WP3.5.1: xla vs mosaic-v1 vs mosaic-v2, each heuristic AND tuned.

  Replaces the earlier `run_autotune_pilot`, which crashed with
  `TypeError: autotune() got an unexpected keyword argument 'lhs'` — read
  `tokamax/_src/autotuning/api.py` directly (not `docs/autotuning.md`, whose
  example is misleading) and found the real signature:

      autotune(f, *args, ignore_cache=False, all_implementations=False, ...)

  Two fixes versus the old code:
    1. `*args` must be POSITIONAL (`autotune(f, lhs, rhs, group_sizes)`), not
       keyword arguments matching f's parameter names.
    2. `all_implementations=True` tunes every implementation registered for
       the op (xla, mosaic v1, mosaic_tpu_v2, ...) in a single call — exactly
       what WP3.5.1 asks for ("compare xla / v1 / v2, default and tuned"), so
       there's no need to call `autotune` once per implementation.

  This runs on ONE shape only (not the full WP1 matrix) — autotuning is an
  exhaustive search and doing it across every shape/distribution would burn a
  lot of real TPU time before we even know which Mosaic version to prefer.

  Not verified locally (same constraint as the rest of this file, `tokamax`
  isn't importable in the local venv) — first run on hardware may still need
  a small fix, but the calling convention above is now grounded in the real
  source, not guessed from docs.
  """
  p = DISTRIBUTIONS[dist_name](shape.g)
  group_sizes = jnp.array(
      tokamax.generate_ragged_dot_group_sizes(
          m=shape.m, num_groups=shape.g, p=p, seed=seed
      ),
      dtype=jnp.int32,
  )
  key = jax.random.key(seed)
  k_lhs, k_rhs = jax.random.split(key)
  lhs = jax.random.normal(k_lhs, (shape.m, shape.k), dtype=jnp.float32).astype(
      shape.dtype
  )
  rhs = jax.random.normal(
      k_rhs, (shape.g, shape.k, shape.n), dtype=jnp.float32
  ).astype(shape.dtype)

  # Untuned (heuristic) numbers first, one per implementation, for direct
  # heuristic-vs-tuned comparison below.
  print(f"\n[fair-baseline] {shape.name}/{dist_name} — heuristic (untuned):")
  for impl in ("xla", "mosaic", "mosaic_tpu_v2"):
    try:
      bench = run_one(shape, dist_name, impl, seed=seed)
      _print_result(shape, dist_name, impl, bench)
    except NotImplementedError as e:
      print(f"[{shape.name}] dist={dist_name} impl={impl} SKIPPED: {e}")

  # `implementation=None` here doesn't matter much: `all_implementations=True`
  # below overrides it and tunes every registered implementation regardless.
  def f(lhs, rhs, group_sizes):
    return tokamax.ragged_dot(lhs, rhs, group_sizes, implementation=None)

  print(f"\n[fair-baseline] {shape.name}/{dist_name} — autotuning ALL implementations ...")
  autotune_result = tokamax.autotune(f, lhs, rhs, group_sizes, all_implementations=True)

  print(f"\n[fair-baseline] {shape.name}/{dist_name} — tuned results per implementation:")
  for bound_args, data in autotune_result.data:
    impl_name = type(bound_args.op).__name__
    try:
      best_config = data.fastest_config
      best = data[best_config]
      print(
          f"  {impl_name}: tuned median_exec={best.median_evaluation_time_ms:.4f}ms "
          f"peak_mem={best.peak_memory_mb:.2f}MB config={best_config}"
      )
    except Exception as e:  # noqa: BLE001 - reporting tuning failures, not raising
      print(f"  {impl_name}: FAILED to autotune ({e})")


def _print_result(shape: ShapeCase, dist_name: str, impl: str, bench):
  print(
      f"[{shape.name}] dist={dist_name} impl={impl} "
      f"compile={bench.compile_time_ms:.2f}ms "
      f"median_exec={bench.median_evaluation_time_ms:.4f}ms "
      f"peak_mem={bench.peak_memory_mb:.2f}MB"
  )


def main(real: bool, fair_baseline: bool):
  print("devices:", jax.devices())

  implementations = ["xla", "mosaic"] if real else ["xla"]
  cases = SHAPE_CASES if real else [c for c in SHAPE_CASES if not c.skip_locally]
  distributions = list(DISTRIBUTIONS) if real else ["uniform", "heavy_skew"]

  # Correctness first: cheap (small/medium shapes only, skips the Large/
  # DeepSeek case to avoid an expensive extra pass over a multi-GB array pair
  # purely for a correctness check — that shape's correctness is already
  # covered more broadly by tokamax's own test suite).
  print("\n--- correctness (xla vs mosaic) ---")
  correctness_cases = [c for c in cases if not c.skip_locally]
  correctness_ok = True
  for shape in correctness_cases:
    for dist_name in distributions:
      correctness_ok &= check_correctness(shape, dist_name)

  print("\n--- performance ---")
  for shape in cases:
    for dist_name in distributions:
      for impl in implementations:
        try:
          bench = run_one(shape, dist_name, impl)
        except NotImplementedError as e:
          print(f"[{shape.name}] dist={dist_name} impl={impl} SKIPPED: {e}")
          continue
        _print_result(shape, dist_name, impl, bench)

  if fair_baseline:
    print("\n--- WP3.5.1 fair baseline (one shape only, costs extra TPU time) ---")
    run_fair_baseline(cases[0], distributions[0])

  print(
      "\n"
      + (
          "Real-hardware mode: the numbers above are real v6e data."
          if real
          else "Local smoke test complete: this only confirms the harness "
          "code runs (implementation='xla', small shapes) — it does not "
          "represent real performance. Both mosaic and the Large shape "
          "must be re-run on v6e with --real."
      )
  )
  if not correctness_ok:
    print("WARNING: at least one correctness check FAILED — see above.")


if __name__ == "__main__":
  main(
      real="--real" in sys.argv,
      fair_baseline="--fair-baseline" in sys.argv,
  )
