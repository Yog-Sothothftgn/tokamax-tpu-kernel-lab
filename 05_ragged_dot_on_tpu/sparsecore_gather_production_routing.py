"""Verification on the REAL production routing at larger token batches, using
the gather index vector that production ACTUALLY feeds its gather.

CORRECTION OF AN EARLIER ASSUMPTION (found by reading
kimi_k3_latent_moe_reference.py::filter_and_pad_to_shard_jittable after run 2):
the public `padded_token_idx` output marks invalid slots with -1, and every
earlier script in this arc ("row0" entries, the `real_invalid_to_row0` pattern,
tokamax_row0, ours_row0, xla_ref) reconstructed the gather indices by mapping
-1 -> 0. Production does NOT gather with that vector: it gathers with the
internal `sorted_token_idx_all = token_of_slot[order]`, whose invalid tail
(slots sorted after the valid ones, truncated to m_padded) holds the token ids
of the first non-local slots in ascending order -- real, ascending token ids,
each repeated up to top_k times in a row, not "all row 0". The "row0" numbers
are therefore measurements of OUR reconstruction, still valid as pattern-
sensitivity findings (all-same-row slots are slow for SparseCore and for XLA at
scale), but NOT a measurement of what production feeds its gather.

This script rebuilds the true vector with exactly the production code path
(same router, same sort_key / padding / stable argsort) and checks that, where
valid, it equals the public padded_token_idx, and that gathering with it
reproduces production `sorted_tokens` bit for bit.

Entries (all must reproduce production `sorted_tokens` bit for bit):
  xla_ref_true / tokamax_true / ours_true   gather indices = production's actual vector
  xla_ref_row0 / tokamax_row0               our earlier reconstruction (-1 -> 0)
  xla_ref_spread / tokamax_spread           invalid slots -> fixed random in-range rows
(ours_* only at <= 8192 tokens: its live packing of the whole table scales with
table rows.) Nothing about Tokamax or the environment is modified. Each token
count runs in its own process; `pipelined` keeps at most k calls in flight,
k = clamp(4e9 // output_bytes, 1, 4), recorded per row; rotated order, 10 rounds
x 20 calls, per-call timing too; SparseCore execution is checked weakly (async
`call-start` ops in the compiled HLO).

History: run 1 (commit 8044f3c) and run 2 (c8cfce2) used only row0/spread; they
showed xla_take_clip and x.at[].get(promise_in_bounds) == xla_ref (gather mode
is irrelevant), tokamax_spread faster than xla_ref(row0) at >= 32768 tokens, but
xla_ref_spread faster still. This is run 3.

To run (real v6e VM only, venv active, from 05_ragged_dot_on_tpu):
  python3 -u sparsecore_gather_production_routing.py sweep 2>&1 | tee sparsecore_prod_routing3.log
"""

import collections
import csv
import math
import pathlib
import subprocess
import sys
import time

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

import sparsecore_gather_tokamax_comparison as tc  # noqa: E402
from sparsecore_gather_spread_invalid_slots import spread_invalid_slots  # noqa: E402
from kimi_k3_latent_moe_reference import (  # noqa: E402
    _MOSAIC_TILE_SIZE,
    _round_up_to_tile,
    _router_gate,
    filter_and_pad_to_shard_jittable,
    kimi_k3_config,
)

RESULTS_DIR = _HERE / "sparsecore_prod_routing_results"
TOKEN_COUNTS = (2048, 8192, 32768, 131072)
OURS_MAX_TOKENS = 8192


def real_dispatch_with_true_indices(num_tokens: int, local_expert_start: int = 137, local_num_experts: int = 64,
                                    capacity_factor: float = 2.0, seed: int = 0):
  """Same inputs as sparsecore_gather_prototype.real_dispatch_indices (same key
  splits), plus the TRUE gather index vector production uses."""
  config = kimi_k3_config()
  keys = jax.random.split(jax.random.key(seed), 4)
  hidden = jax.random.normal(keys[0], (num_tokens, config.hidden_size), dtype=jnp.bfloat16)
  router_weight = jax.random.normal(keys[1], (config.hidden_size, config.num_experts)) * 0.02
  bias = jax.random.normal(keys[2], (config.num_experts,)) * 0.02
  x = (jax.random.normal(keys[3], (num_tokens, config.latent_size)) * 0.02).astype(jnp.bfloat16)
  topk_idx, topk_weight = _router_gate(hidden, router_weight, bias, config)
  sorted_tokens, _, valid_mask, _, padded_idx, _ = filter_and_pad_to_shard_jittable(
      topk_idx, topk_weight, x, config, local_expert_start, local_num_experts, capacity_factor=capacity_factor)

  # Replica of filter_and_pad_to_shard_jittable's index construction (lines for
  # token_of_slot / sort_key / padding / argsort / [:m_padded]).
  total_slots = num_tokens * config.top_k
  flat = topk_idx.reshape(-1)
  token_of_slot = jnp.arange(total_slots) // config.top_k
  local_end = local_expert_start + local_num_experts
  in_shard = (flat >= local_expert_start) & (flat < local_end)
  sort_key = jnp.where(in_shard, flat - local_expert_start, local_num_experts)
  expected_total = num_tokens * config.top_k * local_num_experts / config.num_experts
  m_padded = _round_up_to_tile(math.ceil(expected_total * capacity_factor), _MOSAIC_TILE_SIZE)
  pad_amount = max(0, m_padded - total_slots)
  if pad_amount > 0:
    sort_key = jnp.concatenate([sort_key, jnp.full((pad_amount,), local_num_experts, dtype=sort_key.dtype)])
    token_of_slot = jnp.concatenate([token_of_slot, jnp.zeros((pad_amount,), dtype=token_of_slot.dtype)])
  order = jnp.argsort(sort_key)[:m_padded]
  true_idx = token_of_slot[order].astype(jnp.int32)
  assert bool(jnp.array_equal(jnp.where(valid_mask, true_idx, -1), padded_idx)), "true index vector disagrees with public one"
  return x, padded_idx, valid_mask, sorted_tokens, true_idx


def _med(v):
  s = sorted(v)
  n = len(s)
  return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _tail_stats(true_idx, mask):
  inv = true_idx[~mask]
  if inv.shape[0] == 0:
    return "no invalid slots"
  distinct = int(jnp.unique(inv).shape[0])
  same_as_prev = int((inv[1:] == inv[:-1]).sum())
  return (f"invalid slots={int(inv.shape[0])} distinct rows={distinct} (ids {int(inv.min())}..{int(inv.max())}) "
          f"consecutive repeats={same_as_prev} ({same_as_prev / max(1, int(inv.shape[0]) - 1):.0%})")


def run_point(num_tokens: int, rounds: int = 10, repeats: int = 20):
  x, idx_public, mask, production, true_idx = real_dispatch_with_true_indices(num_tokens)
  n = int(idx_public.shape[0])
  out_bytes = n * x.shape[1] * 2
  k = int(max(1, min(4, 4_000_000_000 // out_bytes)))
  idx_row0 = jnp.where(idx_public < 0, 0, idx_public)
  idx_spread = spread_invalid_slots(idx_row0, mask, num_tokens)
  print(f"\n##### tokens={num_tokens} table_rows={x.shape[0]} slots={n} valid={int(mask.sum())} "
        f"out={out_bytes / 1e9:.2f}GB inflight={k} #####")
  print(f"[true production index vector] {_tail_stats(true_idx, mask)}")
  entries = {
      "xla_ref_true": (tc.reference_gather, (x, true_idx, mask)),
      "xla_ref_row0": (tc.reference_gather, (x, idx_public, mask)),
      "xla_ref_spread": (tc.reference_gather, (x, idx_spread, mask)),
      "tokamax_true": (tc.tokamax_v2, (x, true_idx, mask)),
      "tokamax_row0": (tc.tokamax_v2, (x, idx_public, mask)),
      "tokamax_spread": (tc.tokamax_v2, (x, idx_spread, mask)),
  }
  if num_tokens <= OURS_MAX_TOKENS:
    entries["ours_true"] = (tc.ours_unpack_outside, (x, true_idx, mask))
    entries["ours_row0"] = (tc.ours_unpack_outside, (x, idx_public, mask))
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
  if "xla_ref_true" not in ok_entries or len(ok_entries) < 2:
    print("need xla_ref_true + one more exact entry; skipping timing")
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
  new = not (RESULTS_DIR / "prod3_timing_raw.csv").exists()
  with open(RESULTS_DIR / "prod3_timing_raw.csv", "a", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(raw[0].keys()))
    if new:
      w.writeheader()
    w.writerows(raw)

  def med(nm, key):
    return _med([float(r[key]) for r in raw if r["entry"] == nm])

  base_p, base_b = med("xla_ref_true", "pipelined_ms"), med("xla_ref_true", "per_call_ms")
  summary = []
  for nm in names:
    p, b = med(nm, "pipelined_ms"), med(nm, "per_call_ms")
    summary.append({"tokens": num_tokens, "slots": n, "entry": nm, "pipelined_ms": f"{p:.4f}", "per_call_ms": f"{b:.4f}",
                    "speedup_vs_xla_ref_true_pipelined": f"{base_p / p:.3f}",
                    "speedup_vs_xla_ref_true_per_call": f"{base_b / b:.3f}", "offload_ops": offload.get(nm, "")})
    print(f"  tokens={num_tokens} {nm:16s} pipelined {p:.4f}ms (x{base_p / p:.3f} vs xla_ref_true) | "
          f"per-call {b:.4f}ms (x{base_b / b:.3f}) | offload_ops={offload.get(nm)}")
  new = not (RESULTS_DIR / "prod3_summary.csv").exists()
  with open(RESULTS_DIR / "prod3_summary.csv", "a", newline="") as f:
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
  f = RESULTS_DIR / "prod3_summary.csv"
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
