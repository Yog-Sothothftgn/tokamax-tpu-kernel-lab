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

DTYPE. Tokamax's v2 ragged gather accepts only float32, bfloat16, int8 and int4
tables and raises ValueError for int32 (upstream a669f05), so it can only be
given the float32 reinterpretation of the data. To keep the comparison fair every
candidate is therefore ALSO run on that same float32 view ("same-dtype control",
suffix _f32), and the int32 candidates are kept as the original reference
point. Equality is always checked on raw 32-bit patterns, never on float values.
`start = 0`, `end = N*14` (the expanded length, not N).

Candidates (each exists for int32 and, as `_f32`, for the float32 view):
  xla_wide        jnp.take on the wide table (clip mode)           -> [N, 1792]
  local_narrow    guide-style kernel, window 128, grid split across both cores
                  (ob.make_sc_gather("sc_official_core_split"))     -> narrow
  xla_narrow      jnp.take on the narrow table (clip mode)           -> narrow
  tokamax_narrow  ragged_gather(..., implementation="mosaic_tpu_v2") -> narrow   (f32 only)
  In calibre 2 the int32 group also has `tokamax_narrow_full_bitcast` (Tokamax
  fed through an in-call int32 -> float32 bitcast, conversion counted).

Two measurement calibres, never mixed, each timed separately for the int32 and
the float32 group:
  1  PURE NARROW READ: narrow table and expanded indices are prepared before the
     timed region and passed as arguments; every narrow candidate returns the same
     [N*14, 128] output. xla_wide is listed only as context (same bytes, different
     output shape). The earlier `narrow_raw` number (0.8533 ms at N=65536) INCLUDED
     the index expansion and must not be used as the pure-read baseline.
  2  COMPLETE REPLACEMENT: starts from the raw wide table and raw indices and counts
     layout preparation + index expansion + gather + restoring the [N, 1792]
     output; compared against the plain XLA wide gather of the same dtype.

Protocol: shape / dtype / bitwise equality first; all arrays are jit arguments;
compile + warm-up outside the timed region; 10 rounds x 20 calls, rotated order;
pipelined (<= 4 in flight) and per-call conventions both saved with raw rows.
Each candidate is first probed in its OWN process (a SparseCore core halt poisons
the rest of a process). A short device trace confirms that Tokamax really ran a
SparseCore program (program names on the TEC tracks, not an HLO string count).
Failures are recorded per candidate (A_status.json) and never stop the other
candidates. If Tokamax shows no improvement, nothing larger is added this round.

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
PROBE_NAMES = ("local_narrow", "local_narrow_f32", "tokamax_narrow_f32")


def make_inputs(n: int) -> dict:
  x = (jnp.arange(ROWS * WIDTH, dtype=jnp.int32) + jnp.int32(0x3F800000)).reshape(ROWS, WIDTH)
  idx = jax.random.randint(jax.random.key(1), (n,), 0, ROWS, jnp.int32)
  xn = x.reshape(ROWS * K14, CH)
  return {
      "x": x, "x_f32": jax.lax.bitcast_convert_type(x, jnp.float32),
      "idx": idx, "ie": _expand(idx),
      "xn": xn, "xn_f32": jax.lax.bitcast_convert_type(xn, jnp.float32),
  }


def _expand(idx):
  return (idx[:, None] * K14 + jnp.arange(K14, dtype=jnp.int32)[None, :]).reshape(-1)


def _tokamax_gather(table_f32, idx_exp):
  from tokamax._src.ops.ragged_gather import api as rg_api  # lazily: import problems become candidate failures
  n = idx_exp.shape[0]
  start = jnp.zeros((1,), jnp.int32)
  end = jnp.full((1,), n, jnp.int32)  # the whole EXPANDED length
  return rg_api.ragged_gather(table_f32, idx_exp, start, end, implementation="mosaic_tpu_v2")


def _bits(a):
  return jax.lax.bitcast_convert_type(a, jnp.uint32) if a.dtype == jnp.float32 else a.astype(jnp.uint32)


def build(n: int):
  """Returns the four timing groups: each is (args_keys, {name: fn(*args)}), plus
  which expected output ('narrow' or 'wide') every name must reproduce."""
  local_i32 = ob.make_sc_gather("sc_official_core_split", n * K14, CH)
  local_f32 = ob.make_sc_gather("sc_official_core_split", n * K14, CH)  # dtype follows the table

  def take(t, i):
    return jnp.take(t, i, axis=0, mode="clip")

  # calibre 1: args are prepared arrays (table, wide table, idx, expanded idx)
  c1_i32 = {
      "xla_narrow": lambda xw, idx, tn, ie: take(tn, ie),
      "local_narrow": lambda xw, idx, tn, ie: local_i32(tn, ie),
      "xla_wide(context)": lambda xw, idx, tn, ie: take(xw, idx),
  }
  c1_f32 = {
      "xla_narrow_f32": lambda xw, idx, tn, ie: take(tn, ie),
      "local_narrow_f32": lambda xw, idx, tn, ie: local_f32(tn, ie),
      "tokamax_narrow_f32": lambda xw, idx, tn, ie: _tokamax_gather(tn, ie),
      "xla_wide_f32(context)": lambda xw, idx, tn, ie: take(xw, idx),
  }

  # calibre 2: args are the raw wide table and raw indices only
  def narrow_full(gather):
    def f(x, idx):
      return gather(x.reshape(ROWS * K14, CH), _expand(idx)).reshape(n, WIDTH)
    return f

  def tokamax_bitcast(x, idx):
    xf = jax.lax.bitcast_convert_type(x.reshape(ROWS * K14, CH), jnp.float32)
    return jax.lax.bitcast_convert_type(_tokamax_gather(xf, _expand(idx)), jnp.int32).reshape(n, WIDTH)

  c2_i32 = {
      "xla_wide": lambda x, idx: take(x, idx),
      "local_narrow_full": narrow_full(local_i32),
      "xla_narrow_full": narrow_full(take),
      "tokamax_narrow_full_bitcast": tokamax_bitcast,
  }
  c2_f32 = {
      "xla_wide_f32": lambda x, idx: take(x, idx),
      "local_narrow_full_f32": narrow_full(local_f32),
      "xla_narrow_full_f32": narrow_full(take),
      "tokamax_narrow_full_f32": narrow_full(_tokamax_gather),
  }
  return c1_i32, c1_f32, c2_i32, c2_f32


def _probe_fn(name: str, d: dict, n: int):
  c1_i32, c1_f32, _, _ = build(n)
  if name == "local_narrow":
    return jax.jit(c1_i32[name])(d["x"], d["idx"], d["xn"], d["ie"]), jnp.take(d["xn"], d["ie"], axis=0, mode="clip")
  fn = c1_f32[name]
  return jax.jit(fn)(d["x_f32"], d["idx"], d["xn_f32"], d["ie"]), jnp.take(d["xn_f32"], d["ie"], axis=0, mode="clip")


def run_probe(name: str, n: int):
  d = make_inputs(n)
  out, exp = _probe_fn(name, d, n)
  jax.block_until_ready(out)
  ok = out.shape == exp.shape and out.dtype == exp.dtype and bool(jnp.array_equal(_bits(out), _bits(exp)))
  print(f"PROBE_RESULT {'exact' if ok else 'mismatch'} dtype={out.dtype} shape={tuple(out.shape)}")


def probe(name: str, n: int):
  r = subprocess.run([sys.executable, __file__, "probe", name, str(n)], capture_output=True, text=True, timeout=900)
  ok = r.returncode == 0 and "PROBE_RESULT exact" in r.stdout
  tail = (r.stdout + r.stderr).strip().splitlines()
  detail = next((ln for ln in tail if "PROBE_RESULT" in ln or "Error" in ln), tail[-1] if tail else "")
  print(f"[probe] {name} N={n}: {'OK (exact)' if ok else 'FAILED'} {'' if ok else detail[:400]}")
  return {"ok": ok, "tail": (r.stdout + r.stderr)[-2500:]}


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
  d = make_inputs(n)
  _, c1_f32, _, _ = build(n)
  runs = {}
  for name in ("local_narrow_f32", "tokamax_narrow_f32", "xla_narrow_f32"):
    f = c1_f32[name]
    g = (lambda t, i, m, f=f: f(None, None, t, i))  # noqa: E731
    g.__name__ = name
    runs[name] = (jax.jit(g), d["xn_f32"])
  dt.capture_trace_runs(runs, d["ie"], jnp.zeros((1,), jnp.int32), trace_dir="/tmp/tokamax_narrow_trace", num_repeats=5)
  print("NOTE: read module spans and SparseCore program names only; nested ep_* totals are not interpreted.")
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir="/tmp/tokamax_narrow_trace", top_n_ops=10)


def _time_group(label, fns, args, expected, raw, rows, n, calibre, ref_name, status):
  """Correctness for each candidate (shape, dtype, raw bits), then rotated timing of the exact ones."""
  ok_fns = {}
  for name, fn in fns.items():
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
      # calibre 1: narrow candidates must reproduce the narrow output, the "(context)" row the wide one;
      # calibre 2: every candidate must reproduce the restored wide output.
      want = expected["wide"] if (calibre == 2 or name.endswith("(context)")) else expected["narrow"]
      good = out.shape == want.shape and out.dtype == want.dtype and bool(jnp.array_equal(_bits(out), _bits(want)))
      status[f"{label}/{name}"] = "exact" if good else f"MISMATCH shape={out.shape} dtype={out.dtype}"
      print(f"[{label}][correctness] {name}: {status[f'{label}/{name}']}")
      if good:
        ok_fns[name] = fn
      del out
    except Exception as e:  # noqa: BLE001
      status[f"{label}/{name}"] = f"ERROR {type(e).__name__}: {str(e)[:600]}"
      print(f"[{label}] {name}: ERROR\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
  if ref_name not in ok_fns or len(ok_fns) < 2:
    print(f"[{label}] no timing (reference {ref_name} or all other candidates failed)")
    return
  ordered = {ref_name: ok_fns.pop(ref_name), **ok_fns}
  summ = hc.rotated_timing(ordered, args, f"A {label} N={n}", raw,
                           point={"experiment": "A", "group": label, "calibre": calibre, "N": n})
  for nm, s in summ.items():
    rows.append({"group": label, "N": n, "candidate": nm, "pipelined_ms": f"{s['pipelined_median_ms']:.4f}",
                 "per_call_ms": f"{s['per_call_median_ms']:.4f}", f"speedup_vs_{ref_name}_pipelined": f"{s['speedup_vs_ref_pipelined']:.3f}",
                 f"speedup_vs_{ref_name}_per_call": f"{s['speedup_vs_ref_per_call']:.3f}"})


def main():
  RESULTS_DIR.mkdir(exist_ok=True)
  status = {}
  # Probes run FIRST, while this process has not touched the TPU (a child needs the device;
  # hc.record_session_environment below initialises it).
  probes = {n: {nm: probe(nm, n) for nm in PROBE_NAMES} for n in NS}
  status["probes"] = {str(n): {nm: v["ok"] for nm, v in d.items()} for n, d in probes.items()}
  (RESULTS_DIR / "A_probes.json").write_text(json.dumps(probes, indent=2))
  hc.record_session_environment(RESULTS_DIR, "A")
  status["tokamax_config"] = tokamax_config_report()
  raw, rows1, rows2 = [], [], []
  for n in NS:
    d = make_inputs(n)
    c1_i32, c1_f32, c2_i32, c2_f32 = build(n)
    exp = {
        "i32": {"narrow": jnp.take(d["xn"], d["ie"], axis=0, mode="clip"), "wide": jnp.take(d["x"], d["idx"], axis=0, mode="clip")},
        "f32": {"narrow": jnp.take(d["xn_f32"], d["ie"], axis=0, mode="clip"), "wide": jnp.take(d["x_f32"], d["idx"], axis=0, mode="clip")},
    }
    assert bool(jnp.array_equal(exp["i32"]["narrow"].reshape(n, WIDTH), exp["i32"]["wide"])), "reshape identity broken"
    assert bool(jnp.array_equal(_bits(exp["f32"]["wide"]), _bits(exp["i32"]["wide"]))), "f32 view must carry the same bits"
    print(f"\n##### N={n}  expanded={n * K14} #####")
    skip = lambda names: [nm for nm in names if status["probes"][str(n)].get(nm) is False]  # noqa: E731
    for nm in skip(PROBE_NAMES):
      print(f"[note] probe failed for {nm} at N={n}; its candidates are expected to fail below (recorded)")
    groups = [
        ("calibre1_int32", c1_i32, (d["x"], d["idx"], d["xn"], d["ie"]), exp["i32"], 1, "xla_narrow", rows1),
        ("calibre1_f32", c1_f32, (d["x_f32"], d["idx"], d["xn_f32"], d["ie"]), exp["f32"], 1, "xla_narrow_f32", rows1),
        ("calibre2_int32", c2_i32, (d["x"], d["idx"]), exp["i32"], 2, "xla_wide", rows2),
        ("calibre2_f32", c2_f32, (d["x_f32"], d["idx"]), exp["f32"], 2, "xla_wide_f32", rows2),
    ]
    for label, fns, args, expected, calibre, ref_name, rows in groups:
      try:
        _time_group(label, fns, args, expected, raw, rows, n, calibre, ref_name, status.setdefault(f"N{n}", {}))
      except Exception as e:  # noqa: BLE001
        status.setdefault(f"N{n}", {})[label] = f"GROUP ERROR {type(e).__name__}: {str(e)[:600]}"
        print(f"[{label}] group failed:\n" + traceback.format_exc())
  hc.write_csv(RESULTS_DIR / "A_timing_raw.csv", raw)
  hc.write_csv(RESULTS_DIR / "A_calibre1_pure_narrow_read.csv", rows1)
  hc.write_csv(RESULTS_DIR / "A_calibre2_complete_replacement.csv", rows2)
  if status["probes"][str(NS[-1])].get("tokamax_narrow_f32"):
    try:
      run_trace(NS[-1])
    except Exception as e:  # noqa: BLE001
      status["trace"] = f"ERROR {type(e).__name__}: {str(e)[:600]}"
      print("[trace] failed:\n" + traceback.format_exc())
  (RESULTS_DIR / "A_status.json").write_text(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
  if len(sys.argv) > 1 and sys.argv[1] == "probe":
    run_probe(sys.argv[2], int(sys.argv[3]))
  else:
    try:
      main()
    except Exception:  # noqa: BLE001
      RESULTS_DIR.mkdir(exist_ok=True)
      (RESULTS_DIR / "A_fatal_error.txt").write_text(traceback.format_exc())
      raise
