"""CORRECTION (added after reading filter_and_pad_to_shard_jittable): production
does not gather with "invalid slots -> row 0". That is how THIS arc's harness
reconstructs the indices from the public -1 sentinel. Production gathers with
the internal sorted_token_idx_all, whose invalid tail holds ascending real token
ids (each repeated up to top_k times). See sparsecore_gather_production_routing.py.
Everything below describes the harness's "row0" pattern, not production's.

One-variable end-to-end control: what happens to the full bf16 dispatch
gather (XLA / our latest SparseCore version / Tokamax mosaic_tpu_v2) when ONLY
the index values stored in the INVALID slots change.

Background (measured, see sparsecore_official_gather_benchmark.py `pattern`):
on the real dispatch indices about 55% of the slots are invalid and the
production code points every one of them at row 0. For our SparseCore pure
gather kernel (int32, 1792 words wide, 4864 slots) that costs ~3x
(0.2816 ms vs 0.0898 ms); replacing only the invalid slots by random
in-range rows removed the whole penalty. The earlier three-way comparison
(sparsecore_gather_tokamax_comparison.py) used the row-0 indices, so its
SparseCore numbers include that penalty; whether Tokamax is affected was
never checked. This script measures the complete candidates both ways in the
SAME rotated rounds.

The only difference between the two index arrays is the value stored in
invalid slots (row 0 vs a fixed random in-range row). The mask is the same,
and it is applied after the gather, so every candidate must still return the
bit-identical output; this is checked (value + raw bits) against the unified
reference computed from the ORIGINAL indices, which in turn is checked
against the production `sorted_tokens`.

Entries are named <candidate>_<pattern>: xla_ref / ours_unpack_outside /
tokamax_v2 x row0 / spread. Candidates and their code are imported unchanged
from sparsecore_gather_tokamax_comparison.py (no Tokamax modification, no
environment change). Results go to a separate directory so the earlier
three-way results are not overwritten.

To run (real v6e VM only, venv active, from 05_ragged_dot_on_tpu):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_gather_spread_invalid_slots.py \
      2>&1 | tee sparsecore_spread_invalid_slots.log
"""

import csv
import json
import pathlib
import sys
import time
import traceback

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparsecore_gather_tokamax_comparison as tc  # noqa: E402
from sparsecore_gather_prototype import real_dispatch_indices  # noqa: E402
from sparsecore_gather_group_b_comparison import NUM_TOKENS, LOCAL_NUM_EXPERTS  # noqa: E402
from sparsecore_gather_bf16_colhalf import _named  # noqa: E402
import sparsecore_gather_device_trace as dt  # noqa: E402

RESULTS_DIR = _HERE / "sparsecore_spread_invalid_results"
TRACE_DIR = "/tmp/sparsecore_spread_trace"
NUM_ROUNDS = 10
NUM_REPEATS = 20
TRACE_REPEATS = 10  # fewer than the earlier trace: 6 modules, avoid the profiler event cap


def spread_invalid_slots(idx, mask, num_rows):
  """Same indices, except invalid slots hold a fixed random in-range row."""
  rnd = jax.random.randint(jax.random.key(1), idx.shape, 0, num_rows, jnp.int32)
  return jnp.where(mask, idx, rnd)


def main(trace_only: bool = False) -> None:
  from jax.experimental.pallas import tpu as pltpu
  print(f"devices: {jax.devices()}")
  if pltpu.get_tpu_info().sparse_core is None:
    print("No SparseCore on this TPU -- cannot run this comparison.")
    raise SystemExit(1)
  RESULTS_DIR.mkdir(exist_ok=True)
  tc.RESULTS_DIR = RESULTS_DIR  # keep record_environment's output out of the earlier results dir
  tc.record_environment()

  x, idx_row0, mask, production = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=0)
  idx_row0 = jnp.where(idx_row0 < 0, 0, idx_row0).astype(jnp.int32)  # what production feeds the gather
  idx_spread = spread_invalid_slots(idx_row0, mask, NUM_TOKENS)
  n = int(idx_row0.shape[0])
  print(f"[setup] m_padded={n} num_valid={int(mask.sum())} "
        f"slots==0: row0-pattern={int((idx_row0 == 0).sum())} spread-pattern={int((idx_spread == 0).sum())}; "
        f"valid-slot indices identical: {bool(jnp.array_equal(jnp.where(mask, idx_row0, 0), jnp.where(mask, idx_spread, 0)))}")

  expected = jax.jit(tc.reference_gather)(x, idx_row0, mask)
  jax.block_until_ready(expected)
  ref_val = bool(jnp.array_equal(expected, production))
  ref_bits = tc._bits_equal(expected, production)
  print(f"[reference vs production sorted_tokens] value={ref_val} bits={ref_bits}")
  if not (ref_val and ref_bits):
    print("reference does not match production -- stopping.")
    return

  patterns = {"row0": idx_row0, "spread": idx_spread}
  entries = {}  # name -> (fn, args)
  report = {}
  for cname, fn in tc.CANDIDATES.items():
    for pname, idx in patterns.items():
      name = f"{cname}_{pname}"
      args = (x, idx, mask)
      try:
        out = jax.jit(fn)(*args)
        jax.block_until_ready(out)
      except Exception as e:  # noqa: BLE001
        print(f"\n[{name}] FAILED to compile/run:\n" + "".join(traceback.format_exception(type(e), e, e.__traceback__)))
        report[name] = "error"
        continue
      ok = bool(jnp.array_equal(out, expected)) and tc._bits_equal(out, expected)
      report[name] = "exact" if ok else "MISMATCH"
      print(f"[correctness] {name}: {report[name]}")
      if ok:
        entries[name] = (fn, args)
  (RESULTS_DIR / "correctness.json").write_text(json.dumps(report, indent=2))
  if "xla_ref_row0" not in entries or len(entries) < 2:
    print("Need xla_ref_row0 plus at least one other exact entry -- stopping before timing.")
    return

  names = list(entries)
  if not trace_only:
    run_timing(entries, names)
  run_trace(entries)


def run_timing(entries, names) -> None:
  jitted = {nm: jax.jit(fn) for nm, (fn, _a) in entries.items()}
  for nm, f in jitted.items():
    jax.block_until_ready(f(*entries[nm][1]))
    jax.block_until_ready(f(*entries[nm][1]))

  rows = []
  for rnd in range(NUM_ROUNDS):
    rot = rnd % len(names)
    order = names[rot:] + names[:rot]
    for pos, nm in enumerate(order):
      args = entries[nm][1]
      t0 = time.perf_counter()
      for _ in range(NUM_REPEATS):
        out = jitted[nm](*args)
      jax.block_until_ready(out)
      pipe = (time.perf_counter() - t0) / NUM_REPEATS * 1000
      t0 = time.perf_counter()
      for _ in range(NUM_REPEATS):
        jax.block_until_ready(jitted[nm](*args))
      blk = (time.perf_counter() - t0) / NUM_REPEATS * 1000
      rows.append({"round": rnd, "order_position": pos, "entry": nm, "pipelined_ms": pipe, "per_call_ms": blk})
    print(f"[round {rnd}] order={order}")
  with open(RESULTS_DIR / "timing_raw_per_round.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["round", "order_position", "entry", "pipelined_ms", "per_call_ms"])
    w.writeheader()
    w.writerows(rows)

  summ = {}
  for nm in names:
    p = [r["pipelined_ms"] for r in rows if r["entry"] == nm]
    b = [r["per_call_ms"] for r in rows if r["entry"] == nm]
    summ[nm] = dict(pipe=tc._median(p), pipe_min=min(p), pipe_max=max(p),
                    call=tc._median(b), call_min=min(b), call_max=max(b))
  base = summ["xla_ref_row0"]
  print(f"\n[timing: {NUM_ROUNDS} rounds x {NUM_REPEATS} calls, rotated; speedup = xla_ref_row0 latency / entry latency "
        "(xla_ref_row0 is the production baseline); neither convention is pure device time]")
  table = [["entry", "pipelined_median_ms", "pipelined_min", "pipelined_max", "pipelined_speedup_vs_xla_row0",
            "per_call_median_ms", "per_call_min", "per_call_max", "per_call_speedup_vs_xla_row0"]]
  for nm in names:
    s = summ[nm]
    ps, bs = base["pipe"] / s["pipe"], base["call"] / s["call"]
    table.append([nm, f"{s['pipe']:.4f}", f"{s['pipe_min']:.4f}", f"{s['pipe_max']:.4f}", f"{ps:.3f}",
                  f"{s['call']:.4f}", f"{s['call_min']:.4f}", f"{s['call_max']:.4f}", f"{bs:.3f}"])
    print(f"  {nm:30s} pipelined {s['pipe']:.4f}ms (min {s['pipe_min']:.4f}, max {s['pipe_max']:.4f}) x{ps:.3f} | "
          f"per-call {s['call']:.4f}ms (min {s['call_min']:.4f}, max {s['call_max']:.4f}) x{bs:.3f}")
  with open(RESULTS_DIR / "result_table.csv", "w", newline="") as f:
    csv.writer(f).writerows(table)
  print("\n[row0 -> spread, same candidate, pipelined / per-call latency ratio (row0 time / spread time)]")
  for cname in tc.CANDIDATES:
    a, b = summ.get(f"{cname}_row0"), summ.get(f"{cname}_spread")
    if a and b:
      print(f"  {cname:22s} x{a['pipe'] / b['pipe']:.3f} / x{a['call'] / b['call']:.3f}")


def run_trace(entries) -> None:
  # Device trace: module-level spans + confirmation that SparseCore programs ran.
  # The row0 and spread entries of one candidate are the SAME computation (only
  # the argument values differ). In the first run the profiler attributed all
  # calls of both entries to the first-compiled module name (20 instances under
  # the row0 name, 0 under the spread name), so the module spans there mixed
  # both index patterns. To get one module per entry, each entry's trace
  # wrapper clamps the indices with a DIFFERENT constant (indices are always
  # >= 0 so the clamp never changes a value); that changes the HLO.
  print("\n" + "=" * 78 + "\nDEVICE TRACE (module spans; nested ep_* totals are NOT interpreted)\n" + "=" * 78)

  def wrapped(nm, fn, const):
    def f(x, idx, mask):
      return fn(x, jnp.maximum(idx, jnp.int32(const)), mask)
    return _named(nm, f)

  tjit = {nm: jax.jit(wrapped(nm, fn, -(k + 1))) for k, (nm, (fn, _a)) in enumerate(entries.items())}
  for nm, f in tjit.items():
    jax.block_until_ready(f(*entries[nm][1]))
  with jax.profiler.trace(TRACE_DIR):
    for nm, f in tjit.items():
      with jax.profiler.TraceAnnotation(f"{nm}_repeats"):
        for _ in range(TRACE_REPEATS):
          out = f(*entries[nm][1])
        jax.block_until_ready(out)
  dt.analyze_trace(module_names=tuple(tjit.keys()), trace_dir=TRACE_DIR, top_n_ops=14)


if __name__ == "__main__":
  main(trace_only=(len(sys.argv) > 1 and sys.argv[1] == "trace"))
