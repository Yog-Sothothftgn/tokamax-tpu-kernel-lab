"""Verification on the REAL production routing at larger token batches.

Motivation (measured, sparsecore_gather_scale_and_split.py `sweep2`): with
uniform-random in-range indices and N=65536 slots, Tokamax mosaic_tpu_v2 beat
the XLA gather by ~1.46x once the table had >= ~28K rows and lost to it at
<= 8192 rows (XLA's gather time jumped from 1.19 ms to 3.23 ms between 8192 and
32768 rows while Tokamax stayed at 2.19 ms). Production's table is the token
batch, so a large token batch looks like the large-table regime. That test used
synthetic random indices; this one uses the project's real routing function
(`route_and_filter_to_local_shard_jittable`, via real_dispatch_indices) at
num_tokens = 2048 (control, = the earlier real case), 8192, 32768, 131072.

Entries (all must reproduce the production `sorted_tokens` bit for bit):
  xla_ref            the unified XLA reference used so far: x[max(idx,0)] + mask
  xla_take_clip      same, with jnp.take(..., mode="clip")       } XLA baseline
  xla_at_pib         same, with x.at[idx].get(mode="promise_in_bounds") } sensitivity:
                     is the XLA number an artefact of the gather mode?
  xla_ref_spread     xla_ref on the SPREAD indices (invalid slots -> random row), so the
  xla_at_pib_spread  XLA baseline gets the same index treatment as tokamax_spread
  tokamax_row0       Tokamax mosaic_tpu_v2 on the production indices (invalid
                     slots -> row 0, exactly what production feeds the gather)
  tokamax_spread     identical except invalid slots hold a fixed random in-range
                     row (known from the earlier control to matter for SparseCore)
Nothing about Tokamax or the environment is modified. Each token count runs in
its own process. `pipelined` keeps at most k calls in flight,
k = clamp(4e9 // output_bytes, 1, 4), recorded per row. Rotated order,
10 rounds x 20 calls, per-call timing too. Whether a SparseCore program ran is
checked weakly (async `call-start` ops in the compiled HLO).

To run (real v6e VM only, venv active, from 05_ragged_dot_on_tpu):
  python3 -u sparsecore_gather_production_routing.py sweep 2>&1 | tee sparsecore_prod_routing.log
"""

import collections
import csv
import pathlib
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparsecore_gather_tokamax_comparison as tc  # noqa: E402
from sparsecore_gather_prototype import real_dispatch_indices  # noqa: E402
from sparsecore_gather_spread_invalid_slots import spread_invalid_slots  # noqa: E402

RESULTS_DIR = _HERE / "sparsecore_prod_routing_results"
TOKEN_COUNTS = (2048, 8192, 32768, 131072)
# Run 1 (commit 8044f3c) showed xla_take_clip == xla_ref, tokamax_row0 loses everywhere, tokamax_spread wins at >=32768
# tokens -- but with NO spread-index XLA entry, so the win compared a spread-index Tokamax with a row-0-index XLA.
# Run 2 (this file) adds xla_ref_spread and the promise_in_bounds .at[].get variants (row0 and spread).


def xla_take_clip(x, idx, mask):
  g = jnp.take(x, jnp.maximum(idx, 0), axis=0, mode="clip")
  return jnp.where(mask[:, None], g, jnp.zeros((), x.dtype))


def xla_at_pib(x, idx, mask):
  # jnp.take rejects mode="promise_in_bounds" (measured: ValueError); .at[].get accepts it.
  g = x.at[jnp.maximum(idx, 0)].get(mode="promise_in_bounds")
  return jnp.where(mask[:, None], g, jnp.zeros((), x.dtype))


def _med(v):
  s = sorted(v)
  n = len(s)
  return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_point(num_tokens: int, rounds: int = 10, repeats: int = 20):
  x, idx, mask, production = real_dispatch_indices(num_tokens=num_tokens, local_num_experts=64, seed=0)
  n = int(idx.shape[0])
  out_bytes = n * x.shape[1] * 2
  k = int(max(1, min(4, 4_000_000_000 // out_bytes)))
  idx_spread = spread_invalid_slots(jnp.where(idx < 0, 0, idx), mask, num_tokens)
  print(f"\n##### tokens={num_tokens} table_rows={x.shape[0]} slots={n} valid={int(mask.sum())} "
        f"out={out_bytes / 1e9:.2f}GB inflight={k} slots_with_idx<0={int((idx < 0).sum())} #####")
  entries = {
      "xla_ref": (tc.reference_gather, (x, idx, mask)),
      "xla_ref_spread": (tc.reference_gather, (x, idx_spread, mask)),
      "xla_take_clip": (xla_take_clip, (x, idx, mask)),
      "xla_at_pib": (xla_at_pib, (x, idx, mask)),
      "xla_at_pib_spread": (xla_at_pib, (x, idx_spread, mask)),
      "tokamax_row0": (tc.tokamax_v2, (x, idx, mask)),
      "tokamax_spread": (tc.tokamax_v2, (x, idx_spread, mask)),
  }
  ok_entries, status, offload = {}, {}, {}
  for nm, (fn, args) in entries.items():
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
      exact = bool(jnp.array_equal(out, production)) and tc._bits_equal(out, production)
      status[nm] = "exact" if exact else "MISMATCH"
      if exact:
        ok_entries[nm] = (fn, args)
        try:
          offload[nm] = jax.jit(fn).lower(*args).compile().as_text().count("call-start")
        except Exception:  # noqa: BLE001
          offload[nm] = -1
      del out
    except Exception as e:  # noqa: BLE001
      status[nm] = f"ERROR {type(e).__name__}: {str(e).splitlines()[0][:200]}"
  print(f"[correctness vs production sorted_tokens] {status}\n[offload call-start ops] {offload}")
  if "xla_ref" not in ok_entries or len(ok_entries) < 2:
    print("need xla_ref + one more exact entry; skipping timing")
    return
  names = list(ok_entries)
  jitted = {nm: jax.jit(fn) for nm, (fn, _a) in ok_entries.items()}
  for nm, f in jitted.items():
    jax.block_until_ready(f(*ok_entries[nm][1]))

  def pipelined(nm):
    f, args = jitted[nm], ok_entries[nm][1]
    pend = collections.deque()
    t0 = time.perf_counter()
    for _ in range(repeats):
      pend.append(f(*args))
      if len(pend) > k:
        jax.block_until_ready(pend.popleft())
    while pend:
      jax.block_until_ready(pend.popleft())
    return (time.perf_counter() - t0) / repeats * 1000

  def blocking(nm):
    f, args = jitted[nm], ok_entries[nm][1]
    t0 = time.perf_counter()
    for _ in range(repeats):
      jax.block_until_ready(f(*args))
    return (time.perf_counter() - t0) / repeats * 1000

  raw = []
  for r in range(rounds):
    rot = r % len(names)
    for pos, nm in enumerate(names[rot:] + names[:rot]):
      raw.append({"tokens": num_tokens, "slots": n, "inflight": k, "round": r, "pos": pos, "entry": nm,
                  "pipelined_ms": f"{pipelined(nm):.5f}", "per_call_ms": f"{blocking(nm):.5f}"})
  RESULTS_DIR.mkdir(exist_ok=True)
  new = not (RESULTS_DIR / "prod2_timing_raw.csv").exists()
  with open(RESULTS_DIR / "prod2_timing_raw.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
    if new:
      w.writeheader()
    w.writerows(raw)

  def med(nm, key):
    return _med([float(r[key]) for r in raw if r["entry"] == nm])

  base_p, base_b = med("xla_ref", "pipelined_ms"), med("xla_ref", "per_call_ms")
  summary = []
  for nm in names:
    p, b = med(nm, "pipelined_ms"), med(nm, "per_call_ms")
    summary.append({"tokens": num_tokens, "slots": n, "entry": nm, "pipelined_ms": f"{p:.4f}", "per_call_ms": f"{b:.4f}",
                    "speedup_vs_xla_ref_pipelined": f"{base_p / p:.3f}", "speedup_vs_xla_ref_per_call": f"{base_b / b:.3f}",
                    "offload_ops": offload.get(nm, "")})
    print(f"  tokens={num_tokens} {nm:16s} pipelined {p:.4f}ms (x{base_p / p:.3f} vs xla_ref) | "
          f"per-call {b:.4f}ms (x{base_b / b:.3f}) | offload_ops={offload.get(nm)}")
  new = not (RESULTS_DIR / "prod2_summary.csv").exists()
  with open(RESULTS_DIR / "prod2_summary.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
    if new:
      w.writeheader()
    w.writerows(summary)


def run_sweep():
  RESULTS_DIR.mkdir(exist_ok=True)
  for t in TOKEN_COUNTS:
    r = subprocess.run([sys.executable, __file__, "point", str(t)])
    if r.returncode != 0:
      print(f"!!! tokens={t} exited with code {r.returncode}")
  f = RESULTS_DIR / "prod2_summary.csv"
  print("\nsummary CSV:", f)
  print(f.read_text() if f.exists() else "(none)")


if __name__ == "__main__":
  mode = sys.argv[1] if len(sys.argv) > 1 else "sweep"
  if mode == "sweep":
    run_sweep()
  elif mode == "point":
    run_point(int(sys.argv[2]))
  else:
    raise SystemExit("usage: sparsecore_gather_production_routing.py [sweep]")
