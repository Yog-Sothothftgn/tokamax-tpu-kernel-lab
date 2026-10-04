"""Experiment A: does Tokamax's SparseCore gather (implementation="mosaic_tpu_v2")
change the earlier negative result for the narrow-table reformulation?

Earlier finding (sparsecore_gather_scale_and_split.py `narrow`, local Pallas
kernel): turning the [rows, 1792] int32 table into a narrow [rows*14, 128] table
with expanded indices and the guide's full-narrow-row kernel gave no gain over
the wide-row kernels, and restoring the wide output cost a full extra pass.
Here the same reformulation is run through Tokamax's gather, to see whether that
result depends on the local Pallas implementation.

Data (fixed seeds, shared by every candidate):
  wide table  [2048, 1792] int32, element value 0x3F800000 + flat index, so that its
              float32 reinterpretation is finite, normal and distinct per element
  indices     N in {8192, 65536}, uniform in [0, 2048)
  narrow table [2048*14, 128] = wide table reshaped row-major
  expanded    index t -> 14*t + [0..13]  (N*14 = 114688 / 917504 entries)
DEVIATION FROM THE PLAN, measured from the upstream source (a669f05): Tokamax's
v2 ragged gather accepts only float32, bfloat16, int8 and int4 tables and raises
ValueError for int32. The Tokamax candidate therefore gathers the SAME BYTES as a
float32 table (a bitcast of the int32 table, prepared outside the timed region in
calibre 1 and inside the call in calibre 2). Equality is checked on raw 32-bit
patterns, never on float values. `start = 0`, `end = N*14` (the expanded length,
not N).

Candidates:
  xla_wide        jnp.take on the wide table (clip mode)           -> [N, 1792]
  local_narrow    guide-style kernel, window 128, grid split across both cores
                  (ob.make_sc_gather("sc_official_core_split"))     -> narrow
  tokamax_narrow  ragged_gather(..., implementation="mosaic_tpu_v2") -> narrow
  xla_narrow      jnp.take on the narrow table (clip mode)           -> narrow

Two measurement calibres, never mixed:
  1  PURE NARROW READ: narrow table and expanded indices are prepared before the
     timed region and passed as arguments; every narrow candidate returns the same
     [N*14, 128] output (Tokamax's float32 output is bit-compared as uint32).
     xla_wide is listed only as context (same bytes, different output shape).
     The earlier `narrow_raw` number (0.8533 ms at N=65536) INCLUDED the index
     expansion and must not be used as the pure-read baseline; local_narrow is
     re-measured here with the expansion outside the timing.
  2  COMPLETE REPLACEMENT: starts from the raw wide table and raw indices and counts
     layout preparation + index expansion + gather + restoring the [N, 1792]
     output; compared against the plain XLA wide gather.

Protocol: shape / dtype / bitwise equality first; all arrays are jit arguments;
compile + warm-up outside the timed region; 10 rounds x 20 calls, rotated order;
pipelined (<= 4 in flight) and per-call conventions both saved with raw rows.
Each candidate is first probed in its OWN process (a SparseCore core halt poisons
the rest of a process). A short device trace confirms that Tokamax really ran a
SparseCore program (program names on the TEC tracks, not an HLO string count).
If Tokamax shows no improvement, nothing larger is added in this round.

Run (v6e VM): python tokamax_narrow_gather_experiment.py 2>&1 | tee tokamax_narrow.log
"""

import inspect
import json
import pathlib
import subprocess
import sys
import traceback

import jax
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hw_session_common as hc  # noqa: E402
import sparsecore_official_gather_benchmark as ob  # noqa: E402

RESULTS_DIR = HERE / "hw_session_results"
ROWS, WIDTH, CH = 2048, 1792, 128
K14 = WIDTH // CH  # 14
NS = (8192, 65536)
NAMES = ("local_narrow", "tokamax_narrow", "xla_narrow")


def make_inputs(n: int):
  x = (jnp.arange(ROWS * WIDTH, dtype=jnp.int32) + jnp.int32(0x3F800000)).reshape(ROWS, WIDTH)
  idx = jax.random.randint(jax.random.key(1), (n,), 0, ROWS, jnp.int32)
  xn = x.reshape(ROWS * K14, CH)
  idx_exp = (idx[:, None] * K14 + jnp.arange(K14, dtype=jnp.int32)[None, :]).reshape(-1)
  xn_f32 = jax.lax.bitcast_convert_type(xn, jnp.float32)
  return x, idx, xn, idx_exp, xn_f32


def _expand(idx):
  return (idx[:, None] * K14 + jnp.arange(K14, dtype=jnp.int32)[None, :]).reshape(-1)


def _tokamax_gather(table_f32, idx_exp):
  from tokamax._src.ops.ragged_gather import api as rg_api  # lazily: import problems become candidate failures
  n = idx_exp.shape[0]
  start = jnp.zeros((1,), jnp.int32)
  end = jnp.full((1,), n, jnp.int32)  # the whole EXPANDED length
  return rg_api.ragged_gather(table_f32, idx_exp, start, end, implementation="mosaic_tpu_v2")


def build_fns(n: int):
  """Returns {calibre1: {...}, calibre2: {...}} of callables."""
  local_kernel = ob.make_sc_gather("sc_official_core_split", n * K14, CH)

  # calibre 1: args = (x, idx, xn, idx_exp, xn_f32)
  c1 = {
      "xla_narrow": lambda x, idx, xn, ie, xf: jnp.take(xn, ie, axis=0, mode="clip"),
      "local_narrow": lambda x, idx, xn, ie, xf: local_kernel(xn, ie),
      "tokamax_narrow": lambda x, idx, xn, ie, xf: _tokamax_gather(xf, ie),
      "xla_wide(context)": lambda x, idx, xn, ie, xf: jnp.take(x, idx, axis=0, mode="clip"),
  }
  # calibre 2: args = (x, idx)
  def c2_local(x, idx):
    return local_kernel(x.reshape(ROWS * K14, CH), _expand(idx)).reshape(n, WIDTH)

  def c2_tokamax(x, idx):
    xf = jax.lax.bitcast_convert_type(x.reshape(ROWS * K14, CH), jnp.float32)
    g = _tokamax_gather(xf, _expand(idx))
    return jax.lax.bitcast_convert_type(g, jnp.int32).reshape(n, WIDTH)

  def c2_xla_narrow(x, idx):
    return jnp.take(x.reshape(ROWS * K14, CH), _expand(idx), axis=0, mode="clip").reshape(n, WIDTH)

  c2 = {
      "xla_wide": lambda x, idx: jnp.take(x, idx, axis=0, mode="clip"),
      "local_narrow_full": c2_local,
      "tokamax_narrow_full": c2_tokamax,
      "xla_narrow_full": c2_xla_narrow,
  }
  return c1, c2


def _bits(a):
  return jax.lax.bitcast_convert_type(a, jnp.uint32) if a.dtype == jnp.float32 else a.astype(jnp.uint32)


def check_c1(name, out, exp_narrow, exp_wide, n):
  want = exp_wide if name.startswith("xla_wide") else exp_narrow
  if out.shape != want.shape:
    return False, f"shape {out.shape} != {want.shape}"
  if out.dtype not in (jnp.int32, jnp.float32):
    return False, f"dtype {out.dtype}"
  return bool(jnp.array_equal(_bits(out), want.astype(jnp.uint32))), "bits"


# ---- isolated probe ---------------------------------------------------------

def run_probe(name: str, n: int):
  x, idx, xn, ie, xf = make_inputs(n)
  c1, _ = build_fns(n)
  out = jax.jit(c1[name])(x, idx, xn, ie, xf)
  jax.block_until_ready(out)
  exp = jnp.take(xn, ie, axis=0, mode="clip")
  ok, how = check_c1(name, out, exp, None, n)
  print(f"PROBE_RESULT {'exact' if ok else 'mismatch:' + how} dtype={out.dtype} shape={tuple(out.shape)}")


def probe(name: str, n: int):
  r = subprocess.run([sys.executable, __file__, "probe", name, str(n)], capture_output=True, text=True, timeout=900)
  ok = r.returncode == 0 and "PROBE_RESULT exact" in r.stdout
  tail = (r.stdout + r.stderr).strip().splitlines()
  detail = next((ln for ln in tail if "PROBE_RESULT" in ln or "Error" in ln), tail[-1] if tail else "")
  print(f"[probe] {name} N={n}: {'OK (exact)' if ok else 'FAILED'} {'' if ok else detail[:400]}")
  return ok, (r.stdout + r.stderr)[-3000:]


def tokamax_config_report():
  info = {}
  try:
    from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu_kernel as k
    for nm in dir(k):
      if "col_size" in nm.lower():
        fn = getattr(k, nm)
        try:
          info[f"{nm}(128, 1)"] = fn(128, 1)
        except Exception as e:  # noqa: BLE001
          info[nm] = f"call failed: {type(e).__name__}: {e}"
        try:
          info[f"{nm}_source"] = inspect.getsource(fn)
        except Exception:  # noqa: BLE001
          pass
  except Exception as e:  # noqa: BLE001
    info["error"] = f"{type(e).__name__}: {e}"
  print("[tokamax config] " + json.dumps(info, default=str, indent=1))
  return info


def run_trace(n: int):
  import sparsecore_gather_device_trace as dt
  x, idx, xn, ie, xf = make_inputs(n)
  c1, _ = build_fns(n)
  runs = {}
  for name in ("local_narrow", "tokamax_narrow", "xla_narrow"):
    f = c1[name]
    table = xf if name == "tokamax_narrow" else xn
    g = (lambda t, i, m, f=f: f(None, None, t, i, t))  # noqa: E731
    g.__name__ = name
    runs[name] = (jax.jit(g), table)
  dt.capture_trace_runs(runs, ie, jnp.zeros((1,), jnp.int32), trace_dir="/tmp/tokamax_narrow_trace", num_repeats=5)
  print("NOTE: read module spans and SparseCore program names only; nested ep_* totals are not interpreted.")
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir="/tmp/tokamax_narrow_trace", top_n_ops=10)


def main():
  RESULTS_DIR.mkdir(exist_ok=True)
  # Probes run FIRST, while this process has not touched the TPU (a child needs the device;
  # hc.record_session_environment below initialises it).
  alive = {n: {nm: probe(nm, n)[0] for nm in NAMES} for n in NS}
  (RESULTS_DIR / "A_probes.json").write_text(json.dumps(alive, indent=2))
  hc.record_session_environment(RESULTS_DIR, "A")
  tokamax_config_report()
  raw, rows1, rows2 = [], [], []
  for n in NS:
    x, idx, xn, ie, xf = make_inputs(n)
    c1, c2 = build_fns(n)
    exp_narrow = jnp.take(xn, ie, axis=0, mode="clip")
    exp_wide = jnp.take(x, idx, axis=0, mode="clip")
    assert bool(jnp.array_equal(exp_narrow.reshape(n, WIDTH), exp_wide)), "reference reshape identity broken"
    print(f"\n##### N={n}  expanded={n * K14} #####")

    f1 = {}
    for name, fn in c1.items():
      if name in alive[n] and not alive[n][name]:
        print(f"[calibre 1] {name}: skipped (probe failed)")
        continue
      try:
        out = jax.jit(fn)(x, idx, xn, ie, xf)
        jax.block_until_ready(out)
        ok, how = check_c1(name, out, exp_narrow, exp_wide, n)
      except Exception as e:  # noqa: BLE001
        print(f"[calibre 1] {name}: ERROR\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        continue
      print(f"[calibre 1][correctness] {name}: {'exact' if ok else 'MISMATCH (' + how + ')'} dtype={out.dtype} shape={tuple(out.shape)}")
      if ok:
        f1[name] = fn
      del out
    if "xla_narrow" in f1:
      ordered = {"xla_narrow": f1.pop("xla_narrow"), **f1}
      s1 = hc.rotated_timing(ordered, (x, idx, xn, ie, xf), f"A calibre1 pure-narrow-read N={n}", raw,
                             point={"experiment": "A", "calibre": 1, "N": n})
      for nm, s in s1.items():
        rows1.append({"N": n, "candidate": nm, "pipelined_ms": f"{s['pipelined_median_ms']:.4f}",
                      "per_call_ms": f"{s['per_call_median_ms']:.4f}",
                      "speedup_vs_xla_narrow_pipelined": f"{s['speedup_vs_ref_pipelined']:.3f}",
                      "speedup_vs_xla_narrow_per_call": f"{s['speedup_vs_ref_per_call']:.3f}"})

    f2 = {}
    for name, fn in c2.items():
      base = name.replace("_full", "")
      if base in alive[n] and not alive[n][base]:
        print(f"[calibre 2] {name}: skipped (probe failed)")
        continue
      try:
        out = jax.jit(fn)(x, idx)
        jax.block_until_ready(out)
        ok = out.shape == exp_wide.shape and out.dtype == exp_wide.dtype and bool(jnp.array_equal(out, exp_wide))
      except Exception as e:  # noqa: BLE001
        print(f"[calibre 2] {name}: ERROR\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        continue
      print(f"[calibre 2][correctness] {name}: {'exact' if ok else 'MISMATCH'} dtype={out.dtype} shape={tuple(out.shape)}")
      if ok:
        f2[name] = fn
      del out
    if "xla_wide" in f2:
      s2 = hc.rotated_timing(f2, (x, idx), f"A calibre2 complete-replacement N={n}", raw,
                             point={"experiment": "A", "calibre": 2, "N": n})
      for nm, s in s2.items():
        rows2.append({"N": n, "candidate": nm, "pipelined_ms": f"{s['pipelined_median_ms']:.4f}",
                      "per_call_ms": f"{s['per_call_median_ms']:.4f}",
                      "speedup_vs_xla_wide_pipelined": f"{s['speedup_vs_ref_pipelined']:.3f}",
                      "speedup_vs_xla_wide_per_call": f"{s['speedup_vs_ref_per_call']:.3f}"})
  hc.write_csv(RESULTS_DIR / "A_timing_raw.csv", raw)
  hc.write_csv(RESULTS_DIR / "A_calibre1_pure_narrow_read.csv", rows1)
  hc.write_csv(RESULTS_DIR / "A_calibre2_complete_replacement.csv", rows2)
  if alive[NS[-1]].get("tokamax_narrow"):
    run_trace(NS[-1])


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "probe":
    run_probe(sys.argv[2], int(sys.argv[3]))
  else:
    main()
