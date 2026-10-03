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


def capture_trace_runs(runs, padded_token_idx, valid_mask, trace_dir=TRACE_DIR, num_repeats=NUM_REPEATS) -> None:
  """Generic capture: `runs` maps a distinct name -> (jitted fn, x). Each
  fn is called as `fn(x, padded_token_idx, valid_mask)` with explicit
  arguments (not closure-captured constants -- same fix as
  run_two_chunk_bf16_timing) so every traced call genuinely re-runs any
  packing. Names must be distinct AND be the jit function's `__name__`
  (the XLA module is named `jit_<name>(...)`), since device events are
  attributed by module span, not by host annotation windows."""
  for name, (fn, x) in runs.items():
    jax.block_until_ready(fn(x, padded_token_idx, valid_mask))
  print(f"Writing device trace to {trace_dir} ...")
  with jax.profiler.trace(trace_dir):
    for name, (fn, x) in runs.items():
      with jax.profiler.TraceAnnotation(f"{name}_repeats"):
        for _ in range(num_repeats):
          out = fn(x, padded_token_idx, valid_mask)
        jax.block_until_ready(out)
  print(f"Trace written to {trace_dir}.")


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
  capture_trace_runs(runs, padded_token_idx, valid_mask)


def _find_trace_json(trace_dir: str = TRACE_DIR) -> str:
  matches = glob.glob(f"{trace_dir}/plugins/profile/*/*.trace.json.gz")
  if not matches:
    raise FileNotFoundError(f"no trace.json.gz found under {trace_dir} -- run capture first")
  return sorted(matches)[-1]


def _mean(vals):
  vals = list(vals)
  return sum(vals) / len(vals) if vals else float("nan")


def analyze_trace(module_names=MODULE_NAMES, trace_dir: str = TRACE_DIR, top_n_ops: int = 8) -> None:
  path = _find_trace_json(trace_dir)
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

  for mod in module_names:
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
    print(f"TensorCore XLA ops inside the module (mean us per call, top {top_n_ops}):")
    for name, total in sorted(op_totals.items(), key=lambda kv: -kv[1])[:top_n_ops]:
      print(f"  {name!r:50s} {total / len(instances):9.1f}us")

    # SparseCore offload kernels belonging to each instance. A kernel
    # counts only if it lies FULLY inside the module's device-clock span
    # (+-20us tolerance) -- an earlier version of this analysis used a
    # looser +-50us window that also counted adjacent back-to-back calls'
    # kernels (showing 3.9/5.9 kernels per call instead of 2 per
    # SparseCore-launch).
    #
    # Only `ep_*` events are counted as TEC activity. An earlier version
    # summed EVERY event on a TEC track, including the single enclosing
    # span event that covers the whole kernel (e.g. 'sc_int32_wholerow.6',
    # 853us) -- double-counting that gave impossible busy fractions
    # (178%). Busy time here = sum of `ep_run_kernel` durations only
    # (the per-grid-step body: indirect gather + any unpack compute);
    # `ep_wait_in`/`ep_wait_out` are reported separately.
    kernel_durs = []
    kernels_per_call = []
    run_events_per_sc_kernel = []   # ep_run_kernel total across ALL 16 TECs of ONE SparseCore, ONE kernel
    run_events_per_call = []        # ep_run_kernel total across BOTH SparseCores and every kernel of ONE call
    sc_program_names = set()        # names of the enclosing per-kernel span events seen on TEC tracks
    steps_per_tec = []
    busy_fracs, run_durs = [], []
    wait_in_per_tec, wait_out_per_tec = [], []
    start_offsets, finish_offsets_min, finish_offsets_med, finish_offsets_max = [], [], [], []
    first_timeline = None
    for inst in instances:
      lo, hi = inst["ts"] - 20, inst["ts"] + inst["dur"] + 20
      n_kernels_this_call = 0
      call_run_total = 0
      for ok in offload_tracks:
        pid = ok[0]
        for k in by_track[ok]:
          a, b = k["ts"], k["ts"] + k["dur"]
          if not (a >= lo and b <= hi):
            continue
          n_kernels_this_call += 1
          kernel_durs.append(k["dur"])
          sc_run_total = 0
          finishes = []
          for tk in tec_tracks:
            if tk[0] != pid:
              continue
            all_tevs = inside(by_track[tk], a, b + 1)
            sc_program_names.update(t["name"] for t in all_tevs if not t["name"].startswith("ep_"))
            tevs = [t for t in all_tevs if t["name"].startswith("ep_")]
            if not tevs:
              continue
            runs = [t for t in tevs if t["name"] == "ep_run_kernel"]
            sc_run_total += len(runs)
            steps_per_tec.append(len(runs))
            busy_fracs.append(sum(t["dur"] for t in runs) / k["dur"])
            run_durs.extend(t["dur"] for t in runs)
            wait_in_per_tec.append(sum(t["dur"] for t in tevs if t["name"] == "ep_wait_in"))
            wait_out_per_tec.append(sum(t["dur"] for t in tevs if t["name"] == "ep_wait_out"))
            start_offsets.append(min(t["ts"] for t in tevs) - a)
            finishes.append(max(t["ts"] + t["dur"] for t in tevs) - a)
            if first_timeline is None:
              first_timeline = (thread_names.get(tk, "?"), pid, a, k["dur"], tevs)
          if finishes:
            finishes.sort()
            finish_offsets_min.append(finishes[0])
            finish_offsets_med.append(finishes[len(finishes) // 2])
            finish_offsets_max.append(finishes[-1])
            run_events_per_sc_kernel.append(sc_run_total)
            call_run_total += sc_run_total
      kernels_per_call.append(n_kernels_this_call)
      run_events_per_call.append(call_run_total)

    if not kernel_durs:
      print("(no SparseCore offload kernels inside this module -- pure TensorCore path)")
      continue

    print(f"SparseCore offload kernels: {_mean(kernels_per_call):.1f} per call (expect 2 = one launch per SparseCore "
          f"per sc kernel), mean kernel duration={_mean(kernel_durs):.1f}us")
    print(f"grid-step bodies (`ep_run_kernel`): per TEC per kernel={_mean(steps_per_tec):.1f}, "
          f"TOTAL across one SparseCore's TECs per kernel={_mean(run_events_per_sc_kernel):.1f}")
    print(f"ep_run_kernel TOTAL per call across BOTH SparseCores and all kernels={_mean(run_events_per_call):.1f} "
          f"(= grid steps actually executed; equals the grid size x number of kernels if the work is split, "
          f"2x that if both SparseCores each run everything)")
    print(f"SparseCore program names seen on TEC tracks inside this module: {sorted(sc_program_names)} "
          "(confirms WHICH kernel ran on the SparseCores; NOTE for nested emit_pipeline kernels, ep_run_kernel "
          "counts include BOTH pipeline levels, so they are not a plain grid-step count)")
    print("  -> compare that total with the kernel's grid size (num_indices // window_size = 592): "
          "~592 means this SparseCore alone runs the whole grid (so both SparseCores together do it TWICE); "
          "~296 would mean the grid is split across both SparseCores.")
    print(f"TEC busy fraction (sum of ep_run_kernel / kernel wall)={_mean(busy_fracs):.1%} "
          f"(min {min(busy_fracs):.1%}, max {max(busy_fracs):.1%}); mean ep_run_kernel duration={_mean(run_durs):.1f}us")
    print(f"per-TEC totals inside a kernel: ep_wait_in={_mean(wait_in_per_tec):.1f}us, ep_wait_out={_mean(wait_out_per_tec):.1f}us")
    print(f"TEC start offset after kernel start: mean={_mean(start_offsets):.1f}us; "
          f"TEC finish offset (min/median/max across TECs of one SparseCore): "
          f"{_mean(finish_offsets_min):.1f} / {_mean(finish_offsets_med):.1f} / {_mean(finish_offsets_max):.1f}us "
          f"(kernel wall {_mean(kernel_durs):.1f}us) -- a large spread = load imbalance or waiting on the slowest TEC")

    if first_timeline is not None:
      tname, pid, a, kdur, tevs = first_timeline
      print(f"One TEC's timeline for the first kernel ({tname} on pid {pid}, kernel wall={kdur:.1f}us), "
            f"offsets relative to kernel start (gaps computed against the TRUE previous event):")
      rows, prev_end = [], 0.0
      for t in tevs:
        rel = t["ts"] - a
        rows.append((rel, t["dur"], rel - prev_end, t["name"]))
        prev_end = max(prev_end, rel + t["dur"])
      shown = rows[:12] + [None] + rows[-6:] if len(rows) > 20 else rows
      for r in shown:
        if r is None:
          print("   ...")
          continue
        print(f"   +{r[0]:8.1f}us  dur={r[1]:7.1f}us  gap_before={r[2]:7.1f}us  name={r[3]!r}")
      big = sorted(rows, key=lambda r: -r[2])[:3]
      print("  three largest gaps before an event on this TEC: "
            + ", ".join(f"{g:.1f}us before {n!r} at +{rel:.1f}us" for rel, _d, g, n in big))


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
