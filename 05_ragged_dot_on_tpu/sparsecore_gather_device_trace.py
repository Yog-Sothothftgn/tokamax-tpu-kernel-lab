"""Device trace profiling (2026-10-02), per explicit user request: attribute
where the SparseCore gather's time actually goes -- is it something OUR
implementation does inefficiently (fixable), or is the SparseCore
mechanism itself a poor fit for this access pattern (few, wide, scattered
rows) regardless of implementation?

First real trace (earlier version of this file) found, from the XLA side:
`jit_xla_gather` takes only ~51us of device time per call (gather_fusion
~28us + broadcast_select_fusion ~22us) for a ~34MB output -- already
within ~2x of this chip's HBM roofline (~30us minimum for ~49MB of
necessary traffic at ~1.6TB/s), so NO implementation can beat it by more
than ~1.7x on this op. The SparseCore path measured ~4.6ms per call
(two SC kernel launches of ~2.12ms each + ~0.36ms TensorCore-side
reshape/select work), with the 32 TECs (2 SparseCores x 16 subcores)
only ~37-49% busy inside each kernel.

Known limitation of that first analysis, fixed here: device events were
attributed to regions via HOST-side annotation time windows, but host and
device clocks are offset -- a few device events landed in the wrong
window (e.g. one SparseCore call's events showed up under the XLA
region). This version attributes by DEVICE-side module spans instead:
each implementation gets a distinct jit function name, so its XLA module
(`jit_<name>(...)`, on the "XLA Modules" track) has an unambiguous
device-clock time span, and TensorCore ops / SparseCore kernels / TEC
events are counted only if they fall inside such a span. Track roles
(XLA Modules / XLA Ops / Sparse Core Modules / SparseCore Offload Type /
TEC n) are identified from the trace's own thread-name metadata, not from
hardcoded pid/tid numbers.

Three implementations are traced, so the cost can be split instead of
inferred:
  1. `xla_bf16_gather`      -- plain XLA bf16 gather (the baseline).
  2. `sc_int32_wholerow`    -- SparseCore W=8 whole-row gather, int32,
                               NO unpack compute (pure indirect-DMA cost).
  3. `sc_bf16_two_chunk`    -- the real-dtype candidate (two 1792-wide
                               chunks, bitwise unpack, live repack).
Same DMA byte count between (2) and (3) by construction (68MB gathered,
68MB written either way) -- so the gap between them is NOT more data
moved, it is the unpack compute, the second launch, and the TC-side
repack/reassembly.

For one kernel launch of each SparseCore implementation, one TEC's
event timeline is also printed (startup offset, each ep_run_kernel's
start/duration, gaps between them) to show whether the idle time is a
fixed startup/teardown cost or spread between steps.

To run (real v6e VM only):
  python sparsecore_gather_device_trace.py
"""

import glob
import gzip
import json
import pathlib
import sys
from collections import defaultdict

import jax
import jax.numpy as jnp

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from sparsecore_gather_prototype import LATENT_SIZE, real_dispatch_indices  # noqa: E402
from sparsecore_gather_group_b_comparison import (  # noqa: E402
    NUM_TOKENS,
    LOCAL_NUM_EXPERTS,
    xla_gather,
    sparsecore_gather_whole_row_w8,
    sparsecore_gather_two_chunk_w8_bf16,
)

TRACE_DIR = "/tmp/sparsecore_gather_trace"
NUM_REPEATS = 20
MODULE_NAMES = ("xla_bf16_gather", "sc_int32_wholerow", "sc_bf16_two_chunk")


def xla_bf16_gather(x, idx, mask):
  return xla_gather(x, idx, mask)


def sc_int32_wholerow(x, idx, mask):
  return sparsecore_gather_whole_row_w8(x, idx, mask, window_size=8)


def sc_bf16_two_chunk(x, idx, mask):
  return sparsecore_gather_two_chunk_w8_bf16(x, idx, mask, window_size=8, repack_every_call=True)


def capture_trace(seed: int = 0) -> None:
  print(f"devices: {jax.devices()}")
  print(f"jax version: {jax.__version__}")

  x_bf16, padded_token_idx, valid_mask, _production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=seed
  )
  num_indices = int(padded_token_idx.shape[0])
  print(
      f"[setup] num_tokens={NUM_TOKENS} local_num_experts={LOCAL_NUM_EXPERTS} "
      f"num_indices(m_padded)={num_indices} value_dim={LATENT_SIZE} num_valid={int(jnp.sum(valid_mask))}"
  )
  # Non-degenerate int32 stand-in (same fix as everywhere else in this
  # investigation -- never astype(int32) on small-scale bf16 data).
  x_int32 = jnp.arange(NUM_TOKENS * LATENT_SIZE, dtype=jnp.int32).reshape(NUM_TOKENS, LATENT_SIZE)

  runs = {
      "xla_bf16_gather": (jax.jit(xla_bf16_gather), x_bf16),
      "sc_int32_wholerow": (jax.jit(sc_int32_wholerow), x_int32),
      "sc_bf16_two_chunk": (jax.jit(sc_bf16_two_chunk), x_bf16),
  }

  # Warm up (compile) everything BEFORE tracing, with explicit arguments
  # (not closure-captured constants -- same fix as run_two_chunk_bf16_timing)
  # so every traced call genuinely re-runs the repack.
  for name, (fn, x) in runs.items():
    jax.block_until_ready(fn(x, padded_token_idx, valid_mask))

  print(f"Writing device trace to {TRACE_DIR} ...")
  with jax.profiler.trace(TRACE_DIR):
    for name, (fn, x) in runs.items():
      with jax.profiler.TraceAnnotation(f"{name}_repeats"):
        for _ in range(NUM_REPEATS):
          out = fn(x, padded_token_idx, valid_mask)
        jax.block_until_ready(out)
  print(f"Trace written to {TRACE_DIR}.")


def _find_trace_json() -> str:
  matches = glob.glob(f"{TRACE_DIR}/plugins/profile/*/*.trace.json.gz")
  if not matches:
    raise FileNotFoundError(f"no trace.json.gz found under {TRACE_DIR} -- run capture_trace() first")
  return sorted(matches)[-1]


def _mean(vals):
  vals = list(vals)
  return sum(vals) / len(vals) if vals else float("nan")


def analyze_trace() -> None:
  path = _find_trace_json()
  print(f"Analyzing {path}")
  with gzip.open(path, "rt") as f:
    data = json.load(f)
  events = data["traceEvents"] if isinstance(data, dict) else data
  print(f"total events: {len(events)}")

  thread_names: dict[tuple, str] = {}
  for e in events:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
      thread_names[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "?")

  # Index duration events by (pid, tid) once, sorted by start time.
  by_track: dict[tuple, list] = defaultdict(list)
  for e in events:
    if e.get("ph") == "X":
      by_track[(e.get("pid"), e.get("tid"))].append(e)
  for evs in by_track.values():
    evs.sort(key=lambda e: e["ts"])

  def tracks_named(pred):
    return [k for k, n in thread_names.items() if pred(n)]

  tc_module_tracks = tracks_named(lambda n: n == "XLA Modules")
  tc_op_tracks = tracks_named(lambda n: n == "XLA Ops")
  offload_tracks = tracks_named(lambda n: n == "SparseCore Offload Type")
  tec_tracks = tracks_named(lambda n: n.startswith("TEC"))
  print(
      f"track roles found by thread-name: XLA Modules={tc_module_tracks} XLA Ops={tc_op_tracks} "
      f"Offload={offload_tracks} num_TEC_tracks={len(tec_tracks)}"
  )

  def inside(evs, lo, hi):
    return [e for e in evs if lo <= e["ts"] <= hi]

  for mod in MODULE_NAMES:
    instances = []
    for tk in tc_module_tracks:
      for e in by_track[tk]:
        if e.get("name", "").startswith(f"jit_{mod}("):
          instances.append(e)
    instances.sort(key=lambda e: e["ts"])
    print("\n" + "=" * 78)
    print(f"module `jit_{mod}`: {len(instances)} instances found")
    print("=" * 78)
    if not instances:
      continue

    mod_durs = [e["dur"] for e in instances]
    print(f"TensorCore-side module duration per call: mean={_mean(mod_durs):.1f}us "
          f"min={min(mod_durs):.1f} max={max(mod_durs):.1f}")

    # TensorCore ops inside each instance, summed per op name, averaged per call.
    op_totals: dict[str, float] = defaultdict(float)
    for inst in instances:
      lo, hi = inst["ts"], inst["ts"] + inst["dur"]
      for tk in tc_op_tracks:
        for e in inside(by_track[tk], lo, hi):
          op_totals[e["name"]] += e["dur"]
    print("TensorCore XLA ops inside the module (mean us per call, top 8):")
    for name, total in sorted(op_totals.items(), key=lambda kv: -kv[1])[:8]:
      print(f"  {name!r:50s} {total / len(instances):9.1f}us")

    # SparseCore offload kernels overlapping each instance.
    kernel_durs = []
    kernels_per_call = []
    busy_fracs, ev_counts, ev_durs, start_gaps, tail_gaps = [], [], [], [], []
    first_timeline = None
    for inst in instances:
      lo, hi = inst["ts"] - 50, inst["ts"] + inst["dur"] + 50
      n_kernels_this_call = 0
      for ok in offload_tracks:
        pid = ok[0]
        for k in inside(by_track[ok], lo, hi):
          n_kernels_this_call += 1
          a, b = k["ts"], k["ts"] + k["dur"]
          kernel_durs.append(k["dur"])
          for tk in tec_tracks:
            if tk[0] != pid:
              continue
            tevs = inside(by_track[tk], a, b)
            if not tevs:
              continue
            busy = sum(t["dur"] for t in tevs)
            busy_fracs.append(busy / k["dur"])
            ev_counts.append(len(tevs))
            ev_durs.extend(t["dur"] for t in tevs)
            start_gaps.append(tevs[0]["ts"] - a)
            tail_gaps.append(b - (tevs[-1]["ts"] + tevs[-1]["dur"]))
            if first_timeline is None:
              first_timeline = (thread_names.get(tk, "?"), pid, a, k["dur"], tevs)
      kernels_per_call.append(n_kernels_this_call)

    if not kernel_durs:
      print("(no SparseCore offload kernels overlapped this module -- pure TensorCore path)")
      continue

    print(f"SparseCore offload kernels: {_mean(kernels_per_call):.1f} per call (counted across both SparseCores), "
          f"mean kernel duration={_mean(kernel_durs):.1f}us")
    print(f"TEC activity inside each kernel: busy fraction (sum of TEC event durations / kernel wall)={_mean(busy_fracs):.1%} "
          f"(min {min(busy_fracs):.1%}, max {max(busy_fracs):.1%}); "
          f"events per TEC per kernel={_mean(ev_counts):.1f}, mean event duration={_mean(ev_durs):.1f}us")
    print(f"  startup gap (kernel start -> first TEC event)={_mean(start_gaps):.1f}us, "
          f"tail gap (last TEC event -> kernel end)={_mean(tail_gaps):.1f}us")

    if first_timeline is not None:
      tname, pid, a, kdur, tevs = first_timeline
      print(f"One TEC's timeline for the first kernel ({tname} on pid {pid}, kernel wall={kdur:.1f}us), "
            f"offsets relative to kernel start:")
      prev_end = 0.0
      shown = tevs[:10] + ([None] if len(tevs) > 14 else []) + tevs[-4:] if len(tevs) > 14 else tevs
      for t in shown:
        if t is None:
          print("   ...")
          continue
        rel = t["ts"] - a
        print(f"   +{rel:8.1f}us  dur={t['dur']:7.1f}us  gap_before={rel - prev_end:7.1f}us  name={t['name']!r}")
        prev_end = rel + t["dur"]


if __name__ == "__main__":
  from jax.experimental.pallas import tpu as pltpu
  sc_info = pltpu.get_tpu_info().sparse_core
  if sc_info is None:
    print("No SparseCore on this TPU -- cannot run this profiling here.")
    raise SystemExit(1)
  print(f"sparse_core info: {sc_info}")

  capture_trace()
  print("\n" + "=" * 78 + "\nAnalyzing the trace just captured\n" + "=" * 78)
  analyze_trace()
