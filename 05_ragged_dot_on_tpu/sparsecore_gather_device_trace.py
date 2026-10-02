"""Device trace profiling (2026-10-02), per explicit user request: before
concluding further (or trying to) optimize SparseCore for this workload,
or treating "SparseCore is just inherently slower here" as settled,
capture a REAL device trace of both the two-chunk bf16 SparseCore gather
(`sparsecore_gather_two_chunk_w8_bf16`, the confirmed-correct, best-found
real-dtype candidate) and the plain XLA bf16 gather, so time can be
ATTRIBUTED to specific stages -- DMA/transfer, compute, kernel-launch
dispatch, host-side gaps -- instead of inferred from wall-clock numbers
alone. Answers: is the real bottleneck something OUR implementation does
inefficiently (fixable), or is it the SparseCore mechanism itself being
slow for THIS access pattern (few, wide, scattered rows) regardless of
implementation (a real hardware/workload mismatch, matching the original
suspicion from this whole investigation's very first finding -- the
guide's own benchmark was narrow-row/millions-of-indices, the opposite
shape from ours)?

Reuses this project's established `jax.profiler.trace` workflow (same
Chrome Trace Format JSON output, parseable via plain `gzip`+`json`
stdlib, matching the pattern already used for the TensorCore kernel work
earlier in this project -- see memory for `explore_trace.py`/
`explore_trace_2.py`'s grouping-by-(pid,tid,name) discipline, including
the earlier finding that "Pallas Primitives"/"TC Overlay" tracks can be
EMPTY for a given kernel style -- don't assume SparseCore's own
profiling granularity ahead of time, dump everything found and let the
real trace decide).

Two steps, both in this one file for convenience (capture needs the real
TPU; analysis is pure stdlib and could run anywhere, but there's no
reason to split them here):
  1. `capture_trace()`: warms up (compiles) both implementations, then
     traces `num_repeats` calls of EACH inside its own
     `TraceAnnotation`-wrapped region, so the two can be told apart in
     the resulting trace.
  2. `analyze_trace()`: loads the written trace.json.gz, prints
     process/thread metadata, and a full (pid, tid, name) breakdown of
     every duration event whose timestamp falls inside either annotated
     region -- not just a guess at which track matters.

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
    sparsecore_gather_two_chunk_w8_bf16,
)

TRACE_DIR = "/tmp/sparsecore_gather_trace"
NUM_REPEATS = 20


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

  xla_fn = jax.jit(xla_gather)
  sc_fn = jax.jit(
      lambda x, idx, mask: sparsecore_gather_two_chunk_w8_bf16(
          x, idx, mask, window_size=8, repack_every_call=True
      )
  )

  # Warm up (exclude compile time from the trace) -- explicit args, same
  # fix as run_two_chunk_bf16_timing's correction, so the repack step is
  # genuinely re-run (and thus genuinely traced) on every call below, not
  # silently constant-folded away.
  jax.block_until_ready(xla_fn(x_bf16, padded_token_idx, valid_mask))
  jax.block_until_ready(sc_fn(x_bf16, padded_token_idx, valid_mask))

  print(f"Writing device trace to {TRACE_DIR} ...")
  with jax.profiler.trace(TRACE_DIR):
    with jax.profiler.TraceAnnotation("xla_bf16_repeats"):
      for _ in range(NUM_REPEATS):
        out = xla_fn(x_bf16, padded_token_idx, valid_mask)
      jax.block_until_ready(out)
    with jax.profiler.TraceAnnotation("sparsecore_two_chunk_repeats"):
      for _ in range(NUM_REPEATS):
        out = sc_fn(x_bf16, padded_token_idx, valid_mask)
      jax.block_until_ready(out)
  print(
      f"Trace written to {TRACE_DIR}. Look for the 'xla_bf16_repeats' and "
      "'sparsecore_two_chunk_repeats' annotated regions below, or run "
      "analyze_trace() (called automatically if this file is run as a "
      "script) for a full breakdown."
  )


def _find_trace_json() -> str:
  matches = glob.glob(f"{TRACE_DIR}/plugins/profile/*/*.trace.json.gz")
  if not matches:
    raise FileNotFoundError(f"no trace.json.gz found under {TRACE_DIR} -- run capture_trace() first")
  return sorted(matches)[-1]  # most recent, if more than one


def analyze_trace() -> None:
  path = _find_trace_json()
  print(f"Analyzing {path}")
  with gzip.open(path, "rt") as f:
    data = json.load(f)
  events = data["traceEvents"] if isinstance(data, dict) else data
  print(f"total events: {len(events)}")

  # Process/thread names, so pid/tid numbers can be read as human labels.
  thread_names: dict[tuple, str] = {}
  for e in events:
    if e.get("ph") == "M" and e.get("name") == "thread_name":
      thread_names[(e.get("pid"), e.get("tid"))] = e.get("args", {}).get("name", "?")

  # Find the two annotated regions' time windows (ph=='b'/'e' or a single
  # ph=='X' span, depending on how TraceAnnotation is recorded) by name.
  windows: dict[str, tuple[float, float]] = {}
  for e in events:
    name = e.get("name", "")
    if name in ("xla_bf16_repeats", "sparsecore_two_chunk_repeats") and e.get("ph") == "X":
      ts, dur = e.get("ts", 0), e.get("dur", 0)
      windows[name] = (ts, ts + dur)
  print(f"annotated regions (ts_start, ts_end): {windows}")
  if not windows:
    print(
        "WARNING: could not find the annotation events by exact name match -- "
        "dumping ALL ph=='X' events whose name contains 'repeats' as a fallback."
    )
    for e in events:
      if e.get("ph") == "X" and "repeats" in e.get("name", ""):
        print(f"  {e.get('name')!r} pid={e.get('pid')} tid={e.get('tid')} ts={e.get('ts')} dur={e.get('dur')}")

  # Full (pid, tid, name) breakdown of every duration event inside EACH
  # annotated window, on the device process (pid==3, per this project's
  # established convention) -- not guessing which track matters, dumping
  # all of them.
  for region_name, (lo, hi) in windows.items():
    print(f"\n=== device events inside '{region_name}' (ts in [{lo}, {hi}]) ===")
    grouped = defaultdict(lambda: [0, 0.0])
    for e in events:
      if e.get("ph") != "X":
        continue
      ts = e.get("ts", 0)
      if not (lo <= ts <= hi):
        continue
      key = (e.get("pid"), e.get("tid"), e.get("name"))
      grouped[key][0] += 1
      grouped[key][1] += e.get("dur", 0)
    for (pid, tid, name), (count, total_dur) in sorted(grouped.items(), key=lambda kv: -kv[1][1])[:40]:
      tname = thread_names.get((pid, tid), "?")
      print(
          f"  pid={pid} tid={tid} ({tname:20s}) name={name!r:50s} "
          f"count={count:5d} total_us={total_dur:10.2f} mean_us={total_dur / max(count, 1):8.3f}"
      )
    print(f"  (region wall-clock span: {(hi - lo):.1f}us, "
          f"per-call average: {(hi - lo) / NUM_REPEATS:.2f}us)")


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
