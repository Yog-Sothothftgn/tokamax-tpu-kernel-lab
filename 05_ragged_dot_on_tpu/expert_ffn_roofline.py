"""Stage C (expert FFN / tokamax.ragged_dot) roofline analysis.

Continues WP4/WP5's attribution work now that dispatch (Stage B) has been
JIT-ified and collapsed to ~0.08-0.35ms (see wp4_summary.md) -- Stage C is
now clearly the dominant cost (~2.5-4.7ms across the tested scales). This
answers a boundary/limitation flagged in wp4_summary.md itself: "Stage C
dominating total time doesn't by itself prove it's MXU-compute-bound --
could be weight-HBM-read/padding/tiling limited instead."

CPU-only, no tokamax/TPU needed: computes two theoretical lower bounds
(compute-bound and memory-bandwidth-bound) from known tensor shapes and
PUBLIC TPU v6e hardware specs, then compares them against the ALREADY-
MEASURED real hardware numbers recorded in moe_latency_baseline.md.

**Hardware spec caveat** (same discipline as memory_budget_estimate.py's
HBM-capacity caveat): the numbers below are from Google Cloud's public TPU
v6e documentation (https://cloud.google.com/tpu/docs/v6e, confirmed via
direct fetch 2026-09-11), NOT independently verified via a real device
query on this project's own hardware. Achievable real-world performance is
typically well below the peak spec for either bound.
"""

import argparse
import csv as _csv
import pathlib

# Public TPU v6e (Trillium) per-chip specs, confirmed via
# https://cloud.google.com/tpu/docs/v6e (fetched 2026-09-11) -- see module
# docstring caveat.
_TPU_V6E_PEAK_BF16_TFLOPS = 918.0
_TPU_V6E_HBM_BANDWIDTH_GBPS = 1638.0
_TPU_V6E_HBM_CAPACITY_GB = 32.0  # matches memory_budget_estimate.py's existing assumption

# Real Kimi K3 dims (kimi_k3_config() in kimi_k3_latent_moe_reference.py).
_LATENT_SIZE = 3584
_INTERMEDIATE_SIZE = 3072
_BF16_BYTES = 2

# (m_padded, xla_ms, mosaic_v2_ms) from moe_latency_baseline.md's
# prefill-like table (2026-09-11, local_num_experts=64) -- NOT re-measured
# here, this script only compares against already-confirmed real numbers.
_MEASURED_PREFILL_MS = (
    (384, 2.866, 2.465),
    (1280, 3.380, 2.851),
    (4736, 3.729, 3.092),
    (9472, 4.693, 3.411),
)


def stage_c_roofline(
    m_padded: int,
    local_num_experts: int = 64,
    latent_size: int = _LATENT_SIZE,
    intermediate_size: int = _INTERMEDIATE_SIZE,
) -> dict:
  """Theoretical compute-bound and memory-bandwidth-bound floors for one
  Stage C call (the 3 tokamax.ragged_dot calls: gate, up, down).

  Compute: 3 matmuls, each (m_padded, K) @ (G, K, N) -- FLOPs counted over
  ALL m_padded rows (not just valid ones), since ragged_dot still performs
  the matmul for padding rows against the trailing padding-bucket expert's
  weight row; padding is not free compute-wise, only correctness-wise
  (its contribution gets discarded/zeroed later, but the FLOPs still run).

  Memory: total weight bytes for local_num_experts+1 experts (gate+up+down,
  bf16) that must be read from HBM -- independent of m_padded/num_tokens,
  since weight size only depends on the shard's expert count, not how many
  tokens are dispatched to it. This is why the memory-bound floor is
  CONSTANT across num_tokens in the table below, while the compute-bound
  floor grows with num_tokens.
  """
  bytes_per_expert = 3 * latent_size * intermediate_size * _BF16_BYTES
  total_weight_bytes = (local_num_experts + 1) * bytes_per_expert

  flops_per_row = 3 * 2 * latent_size * intermediate_size  # 2x for multiply-add
  total_flops = m_padded * flops_per_row

  compute_bound_ms = (total_flops / (_TPU_V6E_PEAK_BF16_TFLOPS * 1e12)) * 1000
  memory_bound_ms = (total_weight_bytes / (_TPU_V6E_HBM_BANDWIDTH_GBPS * 1e9)) * 1000
  roofline_ms = max(compute_bound_ms, memory_bound_ms)

  return {
      "m_padded": m_padded,
      "total_flops": total_flops,
      "total_weight_bytes": total_weight_bytes,
      "compute_bound_ms": compute_bound_ms,
      "memory_bound_ms": memory_bound_ms,
      "roofline_ms": roofline_ms,
      "bound_type": "memory" if memory_bound_ms >= compute_bound_ms else "compute",
  }


def compare_against_measured(output_dir: pathlib.Path | None = None) -> list[dict]:
  """Compares the roofline floors against moe_latency_baseline.md's already-
  confirmed real hardware numbers. `roofline_efficiency = roofline_ms /
  real_ms` -- values <=100% mean the real measurement is at or below the
  theoretical floor (expected if the model is conservative or the real run
  overlaps some cost); values >100% would mean the real measurement beat a
  supposed hard floor, which should prompt questioning the model's
  assumptions, not just accepting a number over 100%.
  """
  rows = []
  print(
      f"[roofline] Stage C memory-bound floor is CONSTANT at "
      f"{stage_c_roofline(1)['memory_bound_ms']:.4f}ms regardless of num_tokens "
      "(weight-read cost doesn't depend on how many tokens are dispatched)."
  )
  for m_padded, real_xla, real_v2 in _MEASURED_PREFILL_MS:
    r = stage_c_roofline(m_padded)
    xla_eff = r["roofline_ms"] / real_xla
    v2_eff = r["roofline_ms"] / real_v2
    row = {
        "m_padded": m_padded,
        "compute_bound_ms": r["compute_bound_ms"],
        "memory_bound_ms": r["memory_bound_ms"],
        "roofline_ms": r["roofline_ms"],
        "bound_type": r["bound_type"],
        "real_xla_ms": real_xla,
        "real_mosaic_v2_ms": real_v2,
        "xla_roofline_efficiency": xla_eff,
        "mosaic_v2_roofline_efficiency": v2_eff,
    }
    rows.append(row)
    print(
        f"[roofline] m_padded={m_padded:6d} bound={r['bound_type']:>7s} "
        f"compute_floor={r['compute_bound_ms']:.4f}ms memory_floor={r['memory_bound_ms']:.4f}ms "
        f"real_xla={real_xla:.3f}ms (eff={xla_eff:.1%}) "
        f"real_mosaic_v2={real_v2:.3f}ms (eff={v2_eff:.1%})"
    )
    if xla_eff > 1.0 or v2_eff > 1.0:
      print(
          f"  WARNING: efficiency >100% at m_padded={m_padded} -- the real measurement beat this "
          "script's theoretical 'cold HBM read every call' floor. Most likely explanation: "
          "the benchmark harness's repeated-call timing loop does NOT pay a full cold-weight-read "
          "cost on every iteration (e.g. weight tensors staying resident across repeats), so this "
          "is a real caveat about what these latency numbers represent, not a modeling error to "
          "paper over. A genuinely cold-weight production serving pattern could be SLOWER than "
          "this benchmark suggests."
      )

  print(
      "\n[roofline] CONCLUSION: memory_bound_ms >> compute_bound_ms at every measured scale -- "
      "Stage C is memory-bandwidth-bound (HBM weight reads), not MXU-compute-bound. This matches "
      "the empirical pattern (Stage C's measured latency grows only modestly with num_tokens, "
      "since the dominant cost -- reading ~4.3GB of expert weights -- doesn't depend on token "
      "count at all)."
  )

  if output_dir is not None:
    fieldnames = [
        "m_padded", "compute_bound_ms", "memory_bound_ms", "roofline_ms", "bound_type",
        "real_xla_ms", "real_mosaic_v2_ms", "xla_roofline_efficiency", "mosaic_v2_roofline_efficiency",
    ]
    path = pathlib.Path(output_dir) / "expert_ffn_roofline.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
      writer = _csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      for row in rows:
        writer.writerow(row)
    print(f"  (structured data written to {path})")

  return rows


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--output-dir", type=pathlib.Path, default=None)
  args = parser.parse_args()
  compare_against_measured(output_dir=args.output_dir)
