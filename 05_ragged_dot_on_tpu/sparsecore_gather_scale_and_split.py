"""Two controlled experiments on the dispatch gather, per explicit plan:

STEP 1 (`sweep`): change ONLY the number of output indices, at the real bf16
width (3584), for XLA / our latest SparseCore version / Tokamax mosaic_tpu_v2,
each including its complete wrapper cost (packing, unpacking, masking). Same
table, same indices, same mask for every candidate. Indices are uniform random
in-range (so the known row-0 penalty of the production padding is NOT
present), mask is random with the production valid fraction (0.458); both are
fixed by seed. Table rows are fixed at 2048 so N is the only variable; a
separate small control changes only the table rows at one N.
  N = 4096, 16384, 65536, 262144, 524288 (each point in its OWN process, so a
  failure or an out-of-memory at the large end cannot affect other points).
  Candidates are imported unchanged from sparsecore_gather_tokamax_comparison.
  The `pipelined` convention keeps at most k calls in flight with
  k = clamp(4e9 // output_bytes, 1, 4), because outputs get large; k is
  recorded per row. Each candidate must reproduce the XLA reference bit for
  bit before it is timed. "A SparseCore program ran" is checked weakly by
  counting async offload `call-start` ops in the compiled HLO; run `trace` for
  a real device trace at one N.

STEP 2 (`split`): the SparseCore gather itself (int32 words, the packed
width 1792, random in-range indices), changing ONLY how the row is split into
column chunks: whole row, 2 chunks, 4 chunks, one kernel call each, each chunk
written straight to its final place in the output (no concatenate), with the
window W chosen so the double-buffered output block fits the 256 KiB VMEM.
Every variant is probed in its own process first (a SparseCore core halt
poisons all later programs of a process). Printed next to each measurement is
the prediction of the two-parameter fit from the earlier window sweep
(0.573 us per step + 20.8 GB/s per subcore); it is a fit to int32/128-wide
data, not a validated model, and is printed only so that disagreements are
visible.

To run (real v6e VM only, venv active, from 05_ragged_dot_on_tpu):
  python3 -u sparsecore_gather_scale_and_split.py sweep 2>&1 | tee sparsecore_scale_sweep.log
  python3 -u sparsecore_gather_scale_and_split.py trace 65536 2>&1 | tee sparsecore_scale_trace.log
  python3 -u sparsecore_gather_scale_and_split.py split 2>&1 | tee sparsecore_split.log
"""

import csv
import json
import pathlib
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparsecore_gather_tokamax_comparison as tc  # noqa: E402
import sparsecore_official_gather_benchmark as ob  # noqa: E402
from sparsecore_gather_bf16_colhalf import _named  # noqa: E402
import sparsecore_gather_device_trace as dt  # noqa: E402

RESULTS_DIR = _HERE / "sparsecore_scale_split_results"
WIDTH_BF16 = 3584
PACKED_WORDS = 1792
VALID_FRACTION = 0.458  # 2167 / 4736 in the real dispatch
SWEEP_NS = (4096, 16384, 65536, 262144, 524288)
DEFAULT_TABLE_ROWS = 2048
CANDS = ("xla_ref", "ours_unpack_outside", "tokamax_v2")


def _inflight_for(out_bytes: int) -> int:
  return int(max(1, min(4, 4_000_000_000 // out_bytes)))


def _append_csv(path: pathlib.Path, rows: list):
  if not rows:
    return
  RESULTS_DIR.mkdir(exist_ok=True)
  new = not path.exists()
  with open(path, "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    if new:
      w.writeheader()
    w.writerows(rows)


# ----------------------------------------------------------------------------
# Step 1
# ----------------------------------------------------------------------------

def make_scale_inputs(n: int, table_rows: int):
  kx, ki, km = jax.random.split(jax.random.key(0), 3)
  x = jax.random.normal(kx, (table_rows, WIDTH_BF16), jnp.float32).astype(jnp.bfloat16)
  x = x.at[3, :4].set(jnp.array(-0.0, jnp.bfloat16)).at[5, :4].set(jnp.array(0.0, jnp.bfloat16))
  idx = jax.random.randint(ki, (n,), 0, table_rows, jnp.int32)
  mask = jax.random.bernoulli(km, VALID_FRACTION, (n,))
  return x, idx, mask


def _sc_offload_ops(fn, args) -> int:
  try:
    text = jax.jit(fn).lower(*args).compile().as_text()
    return text.count("call-start")
  except Exception as e:  # noqa: BLE001
    return -1


def run_point(n: int, table_rows: int, rounds: int = 10, repeats: int = 20):
  from jax.experimental.pallas import tpu as pltpu
  assert pltpu.get_tpu_info().sparse_core is not None
  x, idx, mask = make_scale_inputs(n, table_rows)
  args = (x, idx, mask)
  out_bytes = n * WIDTH_BF16 * 2
  k = _inflight_for(out_bytes)
  print(f"\n##### point N={n} table_rows={table_rows} out={out_bytes / 1e9:.2f}GB inflight={k} "
        f"valid={int(mask.sum())} #####")
  fns_all = {"xla_ref": tc.reference_gather, "ours_unpack_outside": tc.ours_unpack_outside,
             "tokamax_v2": tc.tokamax_v2}
  expected = jax.jit(tc.reference_gather)(*args)
  jax.block_until_ready(expected)
  fns, status, offload = {}, {}, {}
  for name, fn in fns_all.items():
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
      ok = bool(jnp.array_equal(out, expected)) and tc._bits_equal(out, expected)
      status[name] = "exact" if ok else "MISMATCH"
      if ok:
        fns[name] = fn
        offload[name] = _sc_offload_ops(fn, args)
      del out
    except Exception as e:  # noqa: BLE001
      status[name] = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}"
  del expected
  print(f"[correctness] {status}  [offload call-start ops in compiled HLO] {offload}")
  if "xla_ref" not in fns or len(fns) < 2:
    print("need xla_ref + one more exact candidate; skipping timing for this point")
    return

  names = list(fns)
  jitted = {nm: jax.jit(f) for nm, f in fns.items()}
  for f in jitted.values():
    jax.block_until_ready(f(*args))

  def pipelined(f):
    import collections
    pend = collections.deque()
    t0 = time.perf_counter()
    for _ in range(repeats):
      pend.append(f(*args))
      if len(pend) > k:
        jax.block_until_ready(pend.popleft())
    while pend:
      jax.block_until_ready(pend.popleft())
    return (time.perf_counter() - t0) / repeats * 1000

  def blocking(f):
    t0 = time.perf_counter()
    for _ in range(repeats):
      jax.block_until_ready(f(*args))
    return (time.perf_counter() - t0) / repeats * 1000

  raw = []
  for r in range(rounds):
    rot = r % len(names)
    for pos, nm in enumerate(names[rot:] + names[:rot]):
      raw.append({"n": n, "table_rows": table_rows, "inflight": k, "round": r, "pos": pos, "candidate": nm,
                  "pipelined_ms": f"{pipelined(jitted[nm]):.5f}", "per_call_ms": f"{blocking(jitted[nm]):.5f}"})
  _append_csv(RESULTS_DIR / "scale_timing_raw.csv", raw)

  def med(nm, key):
    return ob._med([float(r[key]) for r in raw if r["candidate"] == nm])

  base_p, base_b = med("xla_ref", "pipelined_ms"), med("xla_ref", "per_call_ms")
  summary = []
  for nm in names:
    p, b = med(nm, "pipelined_ms"), med(nm, "per_call_ms")
    summary.append({"n": n, "table_rows": table_rows, "inflight": k, "candidate": nm, "pipelined_ms": f"{p:.4f}",
                    "per_call_ms": f"{b:.4f}", "speedup_vs_xla_pipelined": f"{base_p / p:.3f}",
                    "speedup_vs_xla_per_call": f"{base_b / b:.3f}", "offload_ops": offload.get(nm, "")})
    print(f"  N={n} rows={table_rows} {nm:20s} pipelined {p:.4f}ms (x{base_p / p:.3f}) | "
          f"per-call {b:.4f}ms (x{base_b / b:.3f}) | offload_ops={offload.get(nm)}")
  _append_csv(RESULTS_DIR / "scale_summary.csv", summary)


def run_sweep():
  RESULTS_DIR.mkdir(exist_ok=True)
  points = [(n, DEFAULT_TABLE_ROWS) for n in SWEEP_NS] + [(65536, 32768)]
  for n, rows in points:
    r = subprocess.run([sys.executable, __file__, "point", str(n), str(rows)])
    if r.returncode != 0:
      print(f"!!! point N={n} table_rows={rows} exited with code {r.returncode}")
  print("\nsummary CSV:", RESULTS_DIR / "scale_summary.csv")
  print((RESULTS_DIR / "scale_summary.csv").read_text() if (RESULTS_DIR / "scale_summary.csv").exists() else "(none)")


def run_trace(n: int):
  x, idx, mask = make_scale_inputs(n, DEFAULT_TABLE_ROWS)
  fns = {"xla_ref": tc.reference_gather, "ours_unpack_outside": tc.ours_unpack_outside, "tokamax_v2": tc.tokamax_v2}
  runs = {nm: (jax.jit(_named(nm, f)), x) for nm, f in fns.items()}
  dt.capture_trace_runs(runs, idx, mask, trace_dir="/tmp/sparsecore_scale_trace", num_repeats=5)
  print("NOTE: use module spans and SparseCore program names only; nested ep_* totals are not interpreted.")
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir="/tmp/sparsecore_scale_trace", top_n_ops=14)


# ----------------------------------------------------------------------------
# Step 2
# ----------------------------------------------------------------------------

def make_split_gather(n: int, d: int, chunks: int, window: int):
  """Pure int32 gather of `d` words per row in one kernel call. chunks == 1
  reuses the already-proven whole-row kernel; chunks > 1 uses a flattened 1D
  grid of (rows/window)*chunks steps, step s -> (row block s // chunks, column
  chunk s % chunks), copying the (window, d/chunks) piece straight to its
  final place in the output."""
  if chunks == 1:
    return ob.make_sc_gather(f"sc_1d_w{window}_core_split", n, d)
  from jax.experimental import pallas as pl
  from jax.experimental.pallas import tpu as pltpu
  from jax.experimental.pallas import tpu_sc as plsc

  assert d % chunks == 0 and n % window == 0
  cw = d // chunks
  total = (n // window) * chunks
  mesh = plsc.VectorSubcoreMesh(core_axis_name="core", subcore_axis_name="subcore")

  def gather(x, indices):
    @pl.kernel(out_type=jax.ShapeDtypeStruct((n, d), x.dtype), mesh=mesh)
    def kernel(x_hbm, i_hbm, o_hbm):
      def body(i_vmem, o_vmem):
        c = pl.program_id(0) % chunks
        pltpu.sync_copy(x_hbm.at[i_vmem, pl.ds(c * cw, cw)], o_vmem)

      pltpu.emit_pipeline(
          body,
          grid=(total,),
          in_specs=[pl.BlockSpec((window,), index_map=lambda s: (s // chunks,))],
          out_specs=[pl.BlockSpec((window, cw), index_map=lambda s: (s // chunks, s % chunks))],
          core_axis_name=("core", "subcore"),
          dimension_semantics=(pltpu.PARALLEL,),
      )(i_hbm, o_hbm)

    return kernel(x, indices)

  return ob._rename(gather, f"sc_split{chunks}_w{window}")


SPLIT_VARIANTS = {  # name -> (chunks, window); per-buffer bytes = window * (d/chunks) * 4
    "sc_whole_w8": (1, 8),
    "sc_whole_w16": (1, 16),
    "sc_split2_w16": (2, 16),
    "sc_split2_w32": (2, 32),
    "sc_split4_w32": (4, 32),
    "sc_split4_w64": (4, 64),
}


def _probe_split(name: str, n: int) -> bool:
  r = subprocess.run([sys.executable, __file__, "probe_split", name, str(n)], capture_output=True, text=True,
                     timeout=900)
  ok = r.returncode == 0 and "PROBE_RESULT exact" in r.stdout
  tail = (r.stdout + r.stderr).strip().splitlines()
  detail = next((ln for ln in tail if "PROBE_RESULT" in ln or "Error" in ln), tail[-1] if tail else "")
  print(f"[probe] {name} n={n}: {'OK (exact)' if ok else 'FAILED'} {'' if ok else detail[:300]}")
  return ok


def run_probe_split(name: str, n: int):
  chunks, window = SPLIT_VARIANTS[name]
  x = jnp.arange(2048 * PACKED_WORDS, dtype=jnp.int32).reshape(2048, PACKED_WORDS)
  idx = jax.random.randint(jax.random.key(1), (n,), 0, 2048, jnp.int32)
  expected = jax.jit(lambda a, i: jnp.take(a, i, axis=0, mode="clip"))(x, idx)
  out = jax.jit(make_split_gather(n, PACKED_WORDS, chunks, window))(x, idx)
  jax.block_until_ready(out)
  print("PROBE_RESULT exact" if bool(jnp.array_equal(out, expected)) else "PROBE_RESULT mismatch")


def run_split(rounds: int = 10, repeats: int = 20):
  ns = (5120, 65536)
  alive = {n: [nm for nm in SPLIT_VARIANTS if _probe_split(nm, n)] for n in ns}  # TPU untouched in this process
  RESULTS_DIR.mkdir(exist_ok=True)
  ob.RESULTS_DIR = RESULTS_DIR
  ob.record_environment({"split_experiment": True, "probed_ok": alive})
  x = jnp.arange(2048 * PACKED_WORDS, dtype=jnp.int32).reshape(2048, PACKED_WORDS)
  rows, summaries = [], []
  for n in ns:
    idx = jax.random.randint(jax.random.key(1), (n,), 0, 2048, jnp.int32)
    fns = {"xla_take_clip": ob._rename(lambda a, i: jnp.take(a, i, axis=0, mode="clip"), "xla_take_clip")}
    for nm in alive[n]:
      chunks, window = SPLIT_VARIANTS[nm]
      fns[nm] = make_split_gather(n, PACKED_WORDS, chunks, window)
    expected = jax.jit(fns["xla_take_clip"])(x, idx)
    ok_names = ["xla_take_clip"]
    for nm in list(fns)[1:]:
      exact = bool(jnp.array_equal(jax.jit(fns[nm])(x, idx), expected))
      print(f"[correctness n={n}] {nm}: {'exact' if exact else 'MISMATCH'}")
      if exact:
        ok_names.append(nm)
    fns = {nm: fns[nm] for nm in ok_names}
    summ = ob.time_rotated(fns, (x, idx), f"split n={n}", rows, rounds, repeats, dict(n=n, d=PACKED_WORDS))
    out_gb = n * PACKED_WORDS * 4 / 1e9
    print(f"[n={n}] output {out_gb * 1000:.1f} MB; fit prediction = steps_per_subcore * (0.573us + bytes_per_step/20.8GB/s)")
    for nm in fns:
      s = summ[nm]
      line = f"  {nm:16s} measured pipelined {s['pipelined_median_ms']:.4f}ms  out {out_gb / s['pipelined_median_ms'] * 1000:.0f} GB/s"
      if nm in SPLIT_VARIANTS:
        chunks, window = SPLIT_VARIANTS[nm]
        steps = (n // window) * chunks / 32
        step_bytes = window * (PACKED_WORDS // chunks) * 4
        pred = steps * (0.573 + step_bytes / 20.8e3) / 1000  # us -> ms; 20.8 GB/s = 20.8e3 B/us
        line += f" | fit predicts {pred:.4f}ms (steps/subcore={steps:.0f}, step={step_bytes / 1024:.0f} KiB)"
      print(line)
    summaries.append({"n": n, "summary": summ})
  ob.write_rows_csv(RESULTS_DIR / "split_timing_raw.csv", rows)
  (RESULTS_DIR / "split_summary.json").write_text(json.dumps(summaries, indent=2))


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
  if mode == "sweep":
    run_sweep()
  elif mode == "point":
    run_point(int(sys.argv[2]), int(sys.argv[3]))
  elif mode == "trace":
    run_trace(int(sys.argv[2]) if len(sys.argv) > 2 else 65536)
  elif mode == "split":
    run_split()
  elif mode == "probe_split":
    run_probe_split(sys.argv[2], int(sys.argv[3]))
  else:
    raise SystemExit("usage: sparsecore_gather_scale_and_split.py [sweep|trace N|split]")
