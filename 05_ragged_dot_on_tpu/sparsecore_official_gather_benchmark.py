"""Reproduction of the OFFICIAL JAX Pallas SparseCore gather benchmark on
this project's TPU v6e, followed (separately) by a one-variable-at-a-time
grid. Purpose (per explicit user plan): first establish, on OUR machine, a
reference point where SparseCore gather is documented to be competitive,
then move toward our own workload one variable at a time. It is NOT a test
of whether "the environment is at fault"; a loss here would not prove that,
and a win here would not by itself prove that row width explains our losses.

Source of the reference code: the guide at
https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html (gather example +
benchmark cell). Quoted numbers there, which are for TPU 7x per the guide's
own output cells, NOT v6e: SparseCore 4.05 ms vs TensorCore `jnp.take`
18.1 ms (~4.5x) at batch_size=4096, value_dim=128, gather_window_size=128,
num_steps=1024 (num_indices = 128*2*16*1024 = 4,194,304, int32).

Variants (names are fixed; a name only ever describes the code it runs):

  xla_take                 jax.jit(lambda x, i: jnp.take(x, i, axis=0)),
                           verbatim from the guide's TensorCore cell.
  sc_official_verbatim     The guide's `gather` kernel as written:
                           (1, W) 2D index block, `x_hbm.at[i_vmem.at[0]]`,
                           emit_pipeline with core_axis_name='subcore' ONLY.
                           With only the subcore axis, both SparseCores run
                           the whole grid (each core redundantly) -- this is
                           what the guide's code does; we did not add that.
  sc_official_core_split   IDENTICAL to sc_official_verbatim except
                           core_axis_name=('core', 'subcore'), so the grid is
                           divided across both cores too. This is a
                           MODIFIED version and is never called "verbatim".
  sc_w8_1d_core_split      Our own earlier confirmed-working form: 1D ref
                           index block, window 8, tuple core axes. Used only
                           in the grid (it is the form that fits at wide
                           value_dim; window 128 does not).

Deviations from the guide's cell, all recorded in env.json:
  - x dtype is stated explicitly as int32 (the guide's `jnp.arange` gives
    int32 with x64 off); sc_num_cores/subcores come from the guide's
    literals (2, 16) and are asserted equal to `get_tpu_info().sparse_core`.
  - timing is NOT `%timeit`: explicit jit arguments, correctness first,
    10 rounds x 20 calls, rotated order, pipelined (at most 4 calls in
    flight, because each call here writes a 2 GiB output) and per-call.
  - the device trace uses 5 calls per candidate, synchronized per call.

Correctness is exact (int32 equality vs jnp.take) at a small scale first,
then at full scale, before any timing. Device trace is used only to confirm
that a SparseCore program actually ran and to read module-level device
spans; its nested ep_* breakdown is NOT interpreted here (nested pipelines
double-count in our analyzer).

To run (real v6e VM only, from 05_ragged_dot_on_tpu, venv active):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_official_gather_benchmark.py repro \
      2>&1 | tee sparsecore_official_repro.log
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_official_gather_benchmark.py grid \
      2>&1 | tee sparsecore_official_grid.log
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_official_gather_benchmark.py window       2>&1 | tee sparsecore_official_window.log
`window`: only the window size W (8..128) changes, with int32, value_dim=128,
4,194,304 indices, 1D ref indices, cross-core split all fixed; the official
W=128 2D-index version is kept as a reference, so "window size" and "1D vs 2D
index form at W=128" are separated.
Run `repro` first and look at it before running `grid`.
"""

import collections
import csv
import datetime
import json
import pathlib
import re
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))

import sparsecore_gather_device_trace as dt  # noqa: E402

RESULTS_DIR = pathlib.Path(__file__).parent / "sparsecore_official_bench_results"
TRACE_DIR = "/tmp/sparsecore_official_trace"
GUIDE_URL = "https://docs.jax.dev/en/latest/pallas/tpu/sparsecore.html"

# The guide's benchmark configuration.
BATCH_SIZE = 4096
VALUE_DIM = 128
GATHER_WINDOW_SIZE = 128
OFFICIAL_NUM_STEPS = 1024
GUIDE_SC_NUM_CORES, GUIDE_SC_NUM_SUBCORES = 2, 16
GUIDE_QUOTED_SC_MS, GUIDE_QUOTED_TC_MS = 4.05, 18.1  # TPU 7x, per the guide

MAX_INFLIGHT = 4


# ---------------------------------------------------------------------------
# Inputs and candidates
# ---------------------------------------------------------------------------

def make_inputs(batch_size: int, value_dim: int, num_indices: int):
  """Same construction as the guide: x = arange reshaped, indices =
  randint(key(0)). int32 stated explicitly."""
  x = jnp.arange(batch_size * value_dim, dtype=jnp.int32).reshape(batch_size, value_dim)
  indices = jax.random.randint(jax.random.key(0), (num_indices,), 0, batch_size, jnp.int32)
  return x, indices


def num_indices_for(num_steps: int, window: int = GATHER_WINDOW_SIZE) -> int:
  return window * GUIDE_SC_NUM_CORES * GUIDE_SC_NUM_SUBCORES * num_steps


def xla_take(x, indices):
  return jnp.take(x, indices, axis=0)


def make_sc_gather(variant: str, num_indices: int, value_dim: int):
  """Build a (x, indices) -> gathered function for one SparseCore variant."""
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  cfg = {
      "sc_official_verbatim": dict(window=128, index_2d=True, core_axis="subcore"),
      "sc_official_core_split": dict(window=128, index_2d=True, core_axis=("core", "subcore")),
      "sc_w8_1d_core_split": dict(window=8, index_2d=False, core_axis=("core", "subcore")),
  }.get(variant)
  if cfg is None:
    # "sc_1d_w<W>_core_split": 1D ref indices, window W, tuple core axes.
    m = re.fullmatch(r"sc_1d_w(\d+)_core_split", variant)
    assert m, f"unknown variant {variant}"
    cfg = dict(window=int(m.group(1)), index_2d=False, core_axis=("core", "subcore"))
  window, index_2d, core_axis = cfg["window"], cfg["index_2d"], cfg["core_axis"]
  assert num_indices % window == 0
  vector_mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  def gather(x, indices):
    if index_2d:
      idx = indices.reshape((1, num_indices))
      in_spec = pl.BlockSpec((1, window), index_map=lambda i: (0, i))
    else:
      idx = indices
      in_spec = pl.BlockSpec((window,), index_map=lambda i: (i,))

    @pl.kernel(out_type=jax.ShapeDtypeStruct((num_indices, value_dim), x.dtype), mesh=vector_mesh)
    def kernel(x_hbm, i_hbm, o_hbm):
      def body(i_vmem, o_vmem):
        if index_2d:
          pltpu.sync_copy(x_hbm.at[i_vmem.at[0]], o_vmem)
        else:
          pltpu.sync_copy(x_hbm.at[i_vmem], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(num_indices // window,),
          in_specs=[in_spec],
          out_specs=[pl.BlockSpec((window, value_dim), index_map=lambda i: (i, 0))],
          core_axis_name=core_axis,
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    return kernel(x, idx)

  return _rename(gather, variant)


def _rename(fn, name):
  def f(x, indices):
    return fn(x, indices)
  f.__name__ = name
  f.__qualname__ = name
  return f


# ---------------------------------------------------------------------------
# Environment record
# ---------------------------------------------------------------------------

def _git(*args):
  try:
    return subprocess.run(["git", *args], capture_output=True, text=True,
                          cwd=pathlib.Path(__file__).parent, timeout=10).stdout.strip()
  except Exception as e:  # noqa: BLE001
    return f"unavailable: {e}"


def record_environment(extra: dict | None = None) -> dict:
  from jax.experimental.pallas import tpu as pltpu
  sc = pltpu.get_tpu_info().sparse_core
  env = {
      "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "devices": [str(d) for d in jax.devices()],
      "device_kind": jax.devices()[0].device_kind,
      "jax": jax.__version__,
      "jaxlib": getattr(__import__("jaxlib"), "__version__", "?"),
      "sparse_core_info": repr(sc),
      "kernel_lab_commit": _git("rev-parse", "--short", "HEAD"),
      "kernel_lab_dirty": bool(_git("status", "--porcelain")),
      "guide_url": GUIDE_URL,
      "guide_quoted_ms": {"sparsecore": GUIDE_QUOTED_SC_MS, "tensorcore_take": GUIDE_QUOTED_TC_MS,
                          "hardware": "TPU 7x per the guide's output cells"},
      "guide_config": dict(batch_size=BATCH_SIZE, value_dim=VALUE_DIM,
                           gather_window_size=GATHER_WINDOW_SIZE, num_steps=OFFICIAL_NUM_STEPS),
      "deviations_from_guide": [
          "x dtype stated explicitly as int32 (guide's arange gives int32 with x64 off)",
          "sc_num_cores/subcores use the guide's literals (2, 16), asserted equal to get_tpu_info()",
          "timing: explicit jit args, rotated 10x20, pipelined (<=%d in flight) + per-call, not %%timeit" % MAX_INFLIGHT,
          "sc_official_core_split changes ONLY core_axis_name from 'subcore' to ('core','subcore')",
      ],
  }
  if extra:
    env.update(extra)
  RESULTS_DIR.mkdir(exist_ok=True)
  (RESULTS_DIR / "env.json").write_text(json.dumps(env, indent=2))
  print("[env]", json.dumps(env, indent=2))
  assert sc is not None, "No SparseCore on this TPU"
  if (sc.num_cores, sc.num_subcores) != (GUIDE_SC_NUM_CORES, GUIDE_SC_NUM_SUBCORES):
    print(f"[env] WARNING: sparse_core reports {sc.num_cores} cores x {sc.num_subcores} subcores, "
          f"guide literal is {GUIDE_SC_NUM_CORES} x {GUIDE_SC_NUM_SUBCORES}")
  return env


# ---------------------------------------------------------------------------
# Correctness
# ---------------------------------------------------------------------------

def check_correctness(variants: list[str], batch_size: int, value_dim: int, num_steps: int,
                      label: str) -> dict:
  """Exact int32 equality vs jnp.take. Returns {variant: bool|'ERROR'}."""
  n = num_indices_for(num_steps)
  x, idx = make_inputs(batch_size, value_dim, n)
  expected = jax.jit(xla_take)(x, idx)
  jax.block_until_ready(expected)
  # Non-degeneracy: x[r, c] = r*value_dim + c, so the reference must be
  # exactly idx*value_dim in column 0.
  assert bool(jnp.array_equal(expected[:, 0], idx * value_dim)), "reference is not the expected gather"
  results = {}
  for v in variants:
    try:
      out = jax.jit(make_sc_gather(v, n, value_dim))(x, idx)
      jax.block_until_ready(out)
      ok = bool(jnp.array_equal(out, expected))
      results[v] = ok
      print(f"[correctness {label}] {v}: {'OK (exact)' if ok else 'MISMATCH'}  "
            f"(num_indices={n}, value_dim={value_dim})")
      del out
    except Exception as e:  # noqa: BLE001 -- the real error is the point
      results[v] = "ERROR"
      print(f"[correctness {label}] {v}: FAILED to compile/run:\n"
            + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
  del expected
  return results


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------

def _time_pipelined(f, args, n):
  pending = collections.deque()
  t0 = time.perf_counter()
  for _ in range(n):
    pending.append(f(*args))
    if len(pending) > MAX_INFLIGHT:
      jax.block_until_ready(pending.popleft())
  while pending:
    jax.block_until_ready(pending.popleft())
  return (time.perf_counter() - t0) / n * 1000


def _time_blocking(f, args, n):
  t0 = time.perf_counter()
  for _ in range(n):
    jax.block_until_ready(f(*args))
  return (time.perf_counter() - t0) / n * 1000


def _med(vals):
  s = sorted(vals)
  k = len(s)
  return s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2


def time_rotated(fns: dict, args: tuple, label: str, rows: list, num_rounds=10, num_repeats=20,
                 point: dict | None = None) -> dict:
  """Rotated-order timing of already-correct candidates. fns[0] must be the
  XLA reference. Appends raw per-round rows to `rows`; returns summary."""
  names = list(fns)
  jitted = {n: jax.jit(f) for n, f in fns.items()}
  for f in jitted.values():
    jax.block_until_ready(f(*args))  # compile + warm
  pipe = {n: [] for n in names}
  blk = {n: [] for n in names}
  for r in range(num_rounds):
    rot = r % len(names)
    order = names[rot:] + names[:rot]
    for pos, n in enumerate(order):
      p = _time_pipelined(jitted[n], args, num_repeats)
      b = _time_blocking(jitted[n], args, num_repeats)
      pipe[n].append(p)
      blk[n].append(b)
      rows.append({**(point or {}), "label": label, "round": r, "order_pos": pos, "name": n,
                   "pipelined_ms": f"{p:.5f}", "per_call_ms": f"{b:.5f}"})
  ref = names[0]
  summary = {}
  for n in names:
    summary[n] = dict(
        pipelined_median_ms=_med(pipe[n]), pipelined_min_ms=min(pipe[n]), pipelined_max_ms=max(pipe[n]),
        per_call_median_ms=_med(blk[n]), per_call_min_ms=min(blk[n]), per_call_max_ms=max(blk[n]),
        speedup_vs_xla_pipelined=_med(pipe[ref]) / _med(pipe[n]),
        speedup_vs_xla_per_call=_med(blk[ref]) / _med(blk[n]),
    )
  print(f"\n[{label}] {num_rounds} rounds x {num_repeats} calls, rotated; speedup = xla_take latency / candidate latency")
  for n in names:
    s = summary[n]
    print(f"  {n:26s} pipelined {s['pipelined_median_ms']:.4f}ms (min {s['pipelined_min_ms']:.4f}, "
          f"max {s['pipelined_max_ms']:.4f}) x{s['speedup_vs_xla_pipelined']:.3f} | "
          f"per-call {s['per_call_median_ms']:.4f}ms (min {s['per_call_min_ms']:.4f}, "
          f"max {s['per_call_max_ms']:.4f}) x{s['speedup_vs_xla_per_call']:.3f}")
  return summary


def write_rows_csv(path: pathlib.Path, rows: list):
  if not rows:
    return
  keys = list(rows[0].keys())
  with open(path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=keys)
    w.writeheader()
    w.writerows(rows)
  print(f"[csv] wrote {path}")


# ---------------------------------------------------------------------------
# Trace (module spans + confirmation that a SparseCore program ran)
# ---------------------------------------------------------------------------

def run_trace(fns: dict, args: tuple, num_repeats: int = 5):
  jitted = {n: jax.jit(f) for n, f in fns.items()}
  for f in jitted.values():
    jax.block_until_ready(f(*args))
  print(f"Writing device trace to {TRACE_DIR} ...")
  with jax.profiler.trace(TRACE_DIR):
    for n, f in jitted.items():
      with jax.profiler.TraceAnnotation(f"{n}_repeats"):
        for _ in range(num_repeats):
          jax.block_until_ready(f(*args))
  print("NOTE: only module-level device spans and the presence of SparseCore programs are used "
        "from this analysis; the nested ep_* totals/busy%/gap numbers double-count nested "
        "pipelines and are NOT to be interpreted.")
  dt.analyze_trace(module_names=tuple(jitted.keys()), trace_dir=TRACE_DIR)


# ---------------------------------------------------------------------------
# Step 1: official reproduction
# ---------------------------------------------------------------------------

def run_repro(num_rounds: int = 10, num_repeats: int = 20):
  env = record_environment()
  sc_variants = ["sc_official_verbatim", "sc_official_core_split"]

  small = check_correctness(sc_variants, BATCH_SIZE, VALUE_DIM, num_steps=2, label="small, num_steps=2")
  full = check_correctness(sc_variants, BATCH_SIZE, VALUE_DIM, num_steps=OFFICIAL_NUM_STEPS,
                           label=f"official scale, num_steps={OFFICIAL_NUM_STEPS}")
  (RESULTS_DIR / "correctness.json").write_text(json.dumps({"small": small, "full": full}, indent=2))
  working = [v for v in sc_variants if small.get(v) is True and full.get(v) is True]
  if not working:
    print("No SparseCore variant passed correctness -- stopping before timing.")
    return

  n = num_indices_for(OFFICIAL_NUM_STEPS)
  x, idx = make_inputs(BATCH_SIZE, VALUE_DIM, n)
  fns = {"xla_take": _rename(xla_take, "xla_take")}
  for v in working:
    fns[v] = make_sc_gather(v, n, VALUE_DIM)

  rows = []
  summary = time_rotated(fns, (x, idx), "official-scale repro", rows, num_rounds, num_repeats,
                         point=dict(batch_size=BATCH_SIZE, value_dim=VALUE_DIM, num_indices=n))
  write_rows_csv(RESULTS_DIR / "repro_timing_raw.csv", rows)
  (RESULTS_DIR / "repro_summary.json").write_text(json.dumps(
      {"summary": summary, "guide_quoted_speedup_tpu7x": GUIDE_QUOTED_TC_MS / GUIDE_QUOTED_SC_MS}, indent=2))
  print(f"\n[reference] guide's quoted ratio (TPU 7x, its own harness): "
        f"{GUIDE_QUOTED_TC_MS / GUIDE_QUOTED_SC_MS:.2f}x")

  run_trace(fns, (x, idx))


# ---------------------------------------------------------------------------
# Step 2: grid (run only after looking at the repro). One variable at a time.
# ---------------------------------------------------------------------------

def _grid_point(batch_size, value_dim, num_indices, sc_variants, rows, all_summaries, label,
                num_rounds, num_repeats):
  x, idx = make_inputs(batch_size, value_dim, num_indices)
  expected = jax.jit(xla_take)(x, idx)
  jax.block_until_ready(expected)
  fns = {"xla_take": _rename(xla_take, "xla_take")}
  status = {}
  # Second XLA baseline: identical except mode="clip", i.e. without the
  # out-of-bounds fill handling that jnp.take's default mode adds (the repro
  # trace showed a separate `broadcast_select_fusion` next to the gather
  # fusion). Indices here are always in bounds, so outputs are identical.
  # Speedups below stay relative to `xla_take` (the guide's own baseline).
  f_clip = _rename(lambda a, i: jnp.take(a, i, axis=0, mode="clip"), "xla_take_clip")
  out = jax.jit(f_clip)(x, idx)
  jax.block_until_ready(out)
  status["xla_take_clip"] = "exact" if bool(jnp.array_equal(out, expected)) else "MISMATCH"
  if status["xla_take_clip"] == "exact":
    fns["xla_take_clip"] = f_clip
  del out
  for v in sc_variants:
    try:
      f = make_sc_gather(v, num_indices, value_dim)
      out = jax.jit(f)(x, idx)
      jax.block_until_ready(out)
      ok = bool(jnp.array_equal(out, expected))
      status[v] = "exact" if ok else "MISMATCH"
      if ok:
        fns[v] = f
      del out
    except Exception as e:  # noqa: BLE001
      status[v] = f"ERROR: {type(e).__name__}: {str(e).splitlines()[0][:160]}"
  del expected
  print(f"\n[grid point {label}] batch={batch_size} value_dim={value_dim} num_indices={num_indices} "
        f"correctness={status}")
  point = dict(batch_size=batch_size, value_dim=value_dim, num_indices=num_indices)
  if any(n.startswith("sc_") for n in fns):
    all_summaries.append({"point": point, "status": status,
                          "summary": time_rotated(fns, (x, idx), label, rows, num_rounds, num_repeats, point)})
  else:
    all_summaries.append({"point": point, "status": status, "summary": None})


def run_grid(num_rounds: int = 10, num_repeats: int = 20):
  record_environment({"grid": True})
  rows, summaries = [], []

  # Sweep A: ONLY num_indices changes (value_dim=128, batch=4096, int32).
  for steps in (1, 2, 4, 16, 64, 256, 1024):
    n = num_indices_for(steps)
    _grid_point(BATCH_SIZE, VALUE_DIM, n,
                ["sc_official_verbatim", "sc_official_core_split", "sc_w8_1d_core_split"],
                rows, summaries, f"A num_indices={n}", num_rounds, num_repeats)

  # Sweep B: ONLY value_dim changes (num_indices fixed, batch=4096, int32).
  # Window 128 does not fit VMEM beyond value_dim=128, so only the window-8
  # implementation (a single fixed implementation across the sweep) is used.
  for n in (num_indices_for(1), num_indices_for(16)):
    for vd in (128, 256, 512, 1024, 2048, 3584):
      _grid_point(BATCH_SIZE, vd, n, ["sc_w8_1d_core_split"], rows, summaries,
                  f"B num_indices={n} value_dim={vd}", num_rounds, num_repeats)

  write_rows_csv(RESULTS_DIR / "grid_timing_raw.csv", rows)
  (RESULTS_DIR / "grid_summary.json").write_text(json.dumps(summaries, indent=2))

  print("\n" + "=" * 100 + "\nGrid summary (speedup = xla_take / candidate; >1 means the SparseCore candidate is faster)")
  print(f"{'num_indices':>12} {'value_dim':>9}  {'candidate':24s} {'pipelined':>10} {'per-call':>10}")
  for s in summaries:
    p = s["point"]
    if s["summary"] is None:
      print(f"{p['num_indices']:>12} {p['value_dim']:>9}  (no working SC candidate: {s['status']})")
      continue
    for name, v in s["summary"].items():
      if name == "xla_take":
        continue
      print(f"{p['num_indices']:>12} {p['value_dim']:>9}  {name:24s} "
            f"{v['speedup_vs_xla_pipelined']:>9.3f}x {v['speedup_vs_xla_per_call']:>9.3f}x")


# ---------------------------------------------------------------------------
# Step 3: window-size sweep. Fixed: int32, value_dim=128, num_indices=4,194,304,
# 1D ref indices, cross-core split, same inputs/timing. Only the window W
# changes (8..128). The official W=128 2D-index version is kept as the
# reference, which also separates "1D vs 2D at W=128" from "window size".
# ---------------------------------------------------------------------------

def run_window_sweep(num_rounds: int = 10, num_repeats: int = 20):
  record_environment({"window_sweep": True})
  windows = (8, 16, 32, 64, 128)
  n = num_indices_for(OFFICIAL_NUM_STEPS)
  sc_variants = ["sc_official_core_split"] + [f"sc_1d_w{w}_core_split" for w in windows]

  small = check_correctness(sc_variants, BATCH_SIZE, VALUE_DIM, num_steps=2, label="small, num_steps=2")
  full = check_correctness(sc_variants, BATCH_SIZE, VALUE_DIM, num_steps=OFFICIAL_NUM_STEPS,
                           label=f"full scale, num_steps={OFFICIAL_NUM_STEPS}")
  (RESULTS_DIR / "window_correctness.json").write_text(json.dumps({"small": small, "full": full}, indent=2))
  working = [v for v in sc_variants if small.get(v) is True and full.get(v) is True]
  if not working:
    print("No SparseCore variant passed correctness -- stopping before timing.")
    return

  x, idx = make_inputs(BATCH_SIZE, VALUE_DIM, n)
  fns = {"xla_take": _rename(xla_take, "xla_take"),
         "xla_take_clip": _rename(lambda a, i: jnp.take(a, i, axis=0, mode="clip"), "xla_take_clip")}
  for v in working:
    fns[v] = make_sc_gather(v, n, VALUE_DIM)
  rows = []
  summary = time_rotated(fns, (x, idx), "window sweep", rows, num_rounds, num_repeats,
                         point=dict(batch_size=BATCH_SIZE, value_dim=VALUE_DIM, num_indices=n))
  write_rows_csv(RESULTS_DIR / "window_timing_raw.csv", rows)
  (RESULTS_DIR / "window_summary.json").write_text(json.dumps(summary, indent=2))

  clip_ms = summary["xla_take_clip"]["pipelined_median_ms"]
  print("\n[window sweep table] output bytes = num_indices*value_dim*4; speedup vs xla_take_clip")
  out_gb = n * VALUE_DIM * 4 / 1e9
  for v in working:
    s = summary[v]
    print(f"  {v:26s} pipelined {s['pipelined_median_ms']:.4f}ms  vs clip x{clip_ms / s['pipelined_median_ms']:.3f}  "
          f"output GB/s={out_gb / s['pipelined_median_ms'] * 1000:.0f}")

  trace_set = [v for v in ("sc_official_core_split", "sc_1d_w8_core_split", "sc_1d_w128_core_split") if v in working]
  run_trace({v: fns[v] for v in trace_set}, (x, idx))


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "repro"
  print(f"devices: {jax.devices()}  jax: {jax.__version__}  mode: {mode}")
  if mode == "repro":
    run_repro()
  elif mode == "grid":
    run_grid()
  elif mode == "window":
    run_window_sweep()
  else:
    raise SystemExit("usage: sparsecore_official_gather_benchmark.py [repro|grid|window]")
