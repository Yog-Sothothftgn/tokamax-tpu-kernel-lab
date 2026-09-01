"""Full-config memory budget estimate (no TPU needed) for the real sharded
LatentMoE pipeline (`route_and_filter_to_local_shard`-based), at real Kimi
K3 per-expert dims, across a sweep of `(batch_size, seq_len)` values --
so OOM-prone combinations can be flagged, and the batch/seq sweep's
practical upper bound decided, BEFORE spending scarce TPU time finding out
the hard way.

Computes exact tensor-byte sizes for every stage:
  - router weights (router_weight, e_score_correction_bias)
  - latent projection (down_proj, up_proj)
  - this shard's expert gate/up/down weights (local_num_experts + 1 rows --
    the trailing padding-bucket expert, same convention as
    generate_local_shard_workload/route_and_filter_to_local_shard)
  - the dispatched-token buffer (sorted_tokens, size m_padded)
  - the padding portion of that buffer (informational -- already counted
    inside the buffer above, not added again)
  - the output/combine buffers (shard_out, routed_out, up_projection_output,
    shared-expert activations, final_output)
  - the float32 accumulator the router gate uses regardless of compute
    dtype (see `_router_gate`'s docstring) -- a real, separate allocation
    alongside the bf16 buffers

**This is a LOWER-BOUND estimate, not a guaranteed OOM predictor.** It sums
the byte size of every named tensor this pipeline's computation graph
actually holds, but deliberately does NOT add compiler-managed temporaries
(upcast copies during the float32 router matmul, Pallas's double-buffered
VMEM tiles during prefetch, XLA's own scratch/workspace buffers) --
omitting those keeps this estimate a genuine lower bound, so:
  - "estimated total > available HBM" means this configuration WILL
    definitely OOM (a true lower bound can't be wrong in that direction).
  - "estimated total < available HBM" means it MIGHT be fine -- not
    guaranteed, since real usage is always higher than this estimate.
Only an actual run on hardware confirms which combinations really work.

**Rough cross-check against a real measurement**: this project's
`run_latency_sweep` (a DIFFERENT, dense/unsharded 64-expert architecture,
not the sharded one this file estimates -- so not an exact comparison, but
the same order of magnitude) measured `peak_mem=5293MB` at `num_tokens=2048`
on real v6e hardware. This file's weight-memory formula, applied to that
same dense config (64 real experts, hidden=7168/latent=3584/intermediate=3072,
2 shared experts), gives ~3.3GB of weight memory alone -- consistent with
(smaller than, as expected since it excludes activations/compiler overhead)
the ~5.2GB measured total. This is a sanity check that the byte arithmetic
below isn't wildly wrong, not a validation of exact accuracy for the
sharded architecture specifically.

**v6e per-chip HBM capacity used below (32GB) is from public TPU v6e
(Trillium) specs, NOT independently confirmed by a real device query in
this project.** Querying `jax.devices()[0]`'s memory stats on the next TPU
session (see this file's `if __name__` block for a ready-to-run snippet)
would give the real number -- treat 32GB as an approximate ceiling until
then, not a verified one.

No tokamax dependency -- runs anywhere `kimi_k3_latent_moe_reference.py` does.

Usage:
  python memory_budget_estimate.py
"""

import csv as _csv
import dataclasses
import math
import pathlib

from kimi_k3_latent_moe_reference import (
    LatentMoEConfig,
    _MOSAIC_TILE_SIZE,
    _round_up_to_tile,
    kimi_k3_config,
)

_BF16_BYTES = 2
_FP32_BYTES = 4

# Public TPU v6e (Trillium) spec, NOT independently confirmed by a real
# device query in this project -- see module docstring.
_ASSUMED_V6E_HBM_GB = 32

# Same shape sweep as run_latency_sweep/run_realistic_shard_latency_sweep,
# extended further (8192+) specifically to find where the estimate crosses
# the assumed HBM ceiling -- that crossing point is the recommended sweep
# upper bound for the next real hardware run.
_SWEEP_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 128), (1, 512), (1, 2048), (2, 1024), (1, 4096), (4, 1024),
    (1, 8192), (1, 16384), (1, 32768), (1, 65536), (1, 131072), (1, 262144),
)


@dataclasses.dataclass(frozen=True)
class MemoryBudget:
  batch_size: int
  seq_len: int
  num_tokens: int
  expected_local_assignments: float
  m_padded: int
  weight_memory_bytes: int
  activation_memory_bytes: int
  padding_memory_bytes: int
  fp32_accumulator_bytes: int
  estimated_total_bytes: int

  @property
  def estimated_total_gb(self) -> float:
    return self.estimated_total_bytes / (1024**3)


def _tensor_bytes(shape: tuple[int, ...], dtype_bytes: int) -> int:
  n = 1
  for d in shape:
    n *= d
  return n * dtype_bytes


def compute_weight_memory(
    config: LatentMoEConfig, local_num_experts: int, dtype_bytes: int = _BF16_BYTES,
) -> dict[str, int]:
  """Weight memory is CONSTANT across num_tokens/batch/seq -- computed once,
  independent of the shape sweep below."""
  breakdown = {
      "router_weight": _tensor_bytes((config.hidden_size, config.num_experts), dtype_bytes),
      "e_score_correction_bias": _tensor_bytes((config.num_experts,), dtype_bytes),
      "down_proj": _tensor_bytes((config.hidden_size, config.latent_size), dtype_bytes),
      "up_proj": _tensor_bytes((config.latent_size, config.hidden_size), dtype_bytes),
      "norm_scale": _tensor_bytes((config.latent_size,), dtype_bytes),
  }
  # +1 trailing padding-bucket expert row, matching this project's
  # established convention (generate_local_shard_workload / route_and_filter_to_local_shard).
  n_shard_experts = local_num_experts + 1
  breakdown["expert_gate"] = _tensor_bytes(
      (n_shard_experts, config.latent_size, config.intermediate_size), dtype_bytes
  )
  breakdown["expert_up"] = _tensor_bytes(
      (n_shard_experts, config.latent_size, config.intermediate_size), dtype_bytes
  )
  breakdown["expert_down"] = _tensor_bytes(
      (n_shard_experts, config.intermediate_size, config.latent_size), dtype_bytes
  )
  shared_intermediate = config.intermediate_size * config.num_shared_experts
  breakdown["shared_gate"] = _tensor_bytes((config.hidden_size, shared_intermediate), dtype_bytes)
  breakdown["shared_up"] = _tensor_bytes((config.hidden_size, shared_intermediate), dtype_bytes)
  breakdown["shared_down"] = _tensor_bytes((shared_intermediate, config.hidden_size), dtype_bytes)
  breakdown["total"] = sum(breakdown.values())
  return breakdown


def estimate_for_shape(
    config: LatentMoEConfig,
    local_num_experts: int,
    batch_size: int,
    seq_len: int,
    capacity_factor: float = 2.0,
    tile_size: int = _MOSAIC_TILE_SIZE,
    dtype_bytes: int = _BF16_BYTES,
) -> MemoryBudget:
  num_tokens = batch_size * seq_len

  weight_memory = compute_weight_memory(config, local_num_experts, dtype_bytes)["total"]

  # Same m_padded formula filter_and_pad_to_shard actually uses.
  expected_local = num_tokens * config.top_k * local_num_experts / config.num_experts
  m_padded = _round_up_to_tile(math.ceil(expected_local * capacity_factor), tile_size)
  padding_rows = max(0, m_padded - int(expected_local))

  # Activation/output buffers, all sized for THIS num_tokens/m_padded.
  # `sorted_tokens` and `shard_out` both carry m_padded rows (real +
  # padding) -- their padding-row bytes are already included here, not
  # added again by padding_memory below.
  activation = {
      "hidden_states": _tensor_bytes((num_tokens, config.hidden_size), dtype_bytes),
      "x_down_projected": _tensor_bytes((num_tokens, config.latent_size), dtype_bytes),
      "sorted_tokens": _tensor_bytes((m_padded, config.latent_size), dtype_bytes),
      "gate_activation": _tensor_bytes((m_padded, config.intermediate_size), dtype_bytes),
      "up_activation": _tensor_bytes((m_padded, config.intermediate_size), dtype_bytes),
      "situ_activation": _tensor_bytes((m_padded, config.intermediate_size), dtype_bytes),
      "shard_out": _tensor_bytes((m_padded, config.latent_size), dtype_bytes),
      "routed_out": _tensor_bytes((num_tokens, config.latent_size), dtype_bytes),
      "normed": _tensor_bytes((num_tokens, config.latent_size), dtype_bytes),
      "up_projection_output": _tensor_bytes((num_tokens, config.hidden_size), dtype_bytes),
  }
  shared_intermediate = config.intermediate_size * config.num_shared_experts
  activation["shared_gate_activation"] = _tensor_bytes((num_tokens, shared_intermediate), dtype_bytes)
  activation["shared_up_activation"] = _tensor_bytes((num_tokens, shared_intermediate), dtype_bytes)
  activation["shared_situ_activation"] = _tensor_bytes((num_tokens, shared_intermediate), dtype_bytes)
  activation["shared_expert_output"] = _tensor_bytes((num_tokens, config.hidden_size), dtype_bytes)
  activation["final_output"] = _tensor_bytes((num_tokens, config.hidden_size), dtype_bytes)
  activation_memory = sum(activation.values())

  # Informational only -- the padding rows' bytes within sorted_tokens/
  # shard_out, NOT an additive term (already inside activation_memory above).
  padding_memory = _tensor_bytes((padding_rows, config.latent_size), dtype_bytes) * 2

  # The router gate computes logits/sigmoid-scores/combine-weight in
  # float32 regardless of compute dtype (see _router_gate's docstring) --
  # a real, separate allocation alongside the bf16 buffers above.
  fp32_accumulator = (
      _tensor_bytes((num_tokens, config.num_experts), _FP32_BYTES)  # logits
      + _tensor_bytes((num_tokens, config.num_experts), _FP32_BYTES)  # scores
      + _tensor_bytes((num_tokens, config.num_experts), _FP32_BYTES)  # scores_for_choice
      + _tensor_bytes((num_tokens, config.top_k), _FP32_BYTES)  # topk_weight (pre-cast)
  )

  estimated_total = weight_memory + activation_memory + fp32_accumulator

  return MemoryBudget(
      batch_size=batch_size, seq_len=seq_len, num_tokens=num_tokens,
      expected_local_assignments=expected_local, m_padded=m_padded,
      weight_memory_bytes=weight_memory, activation_memory_bytes=activation_memory,
      padding_memory_bytes=padding_memory, fp32_accumulator_bytes=fp32_accumulator,
      estimated_total_bytes=estimated_total,
  )


def _fmt_mb(nbytes: int) -> str:
  return f"{nbytes / (1024**2):.1f}MB"


def print_budget_table(
    config: LatentMoEConfig | None = None,
    local_num_experts: int = 64,
    shapes: tuple[tuple[int, int], ...] = _SWEEP_SHAPES,
    capacity_factor: float = 2.0,
    output_dir: pathlib.Path | None = None,
) -> list[MemoryBudget]:
  config = config or kimi_k3_config()
  weight_bd = compute_weight_memory(config, local_num_experts)

  print(f"[memory-budget] config: hidden={config.hidden_size} latent={config.latent_size} "
        f"intermediate={config.intermediate_size} global_experts={config.num_experts} "
        f"top_k={config.top_k} local_num_experts={local_num_experts} capacity_factor={capacity_factor} "
        "dtype=bf16")
  print(f"[memory-budget] weight memory breakdown (constant across shapes):")
  for name, nbytes in weight_bd.items():
    if name != "total":
      print(f"    {name:<28} {_fmt_mb(nbytes)}")
  print(f"    {'TOTAL WEIGHT MEMORY':<28} {_fmt_mb(weight_bd['total'])}")
  print(f"[memory-budget] assumed v6e HBM per chip: {_ASSUMED_V6E_HBM_GB}GB "
        "(public spec, not independently confirmed on real hardware -- see module docstring)\n")

  header = (
      f"{'batch':>6} {'seq_len':>8} {'tokens':>8} {'local_assign':>13} {'m_padded':>9} "
      f"{'weight_mem':>11} {'activ_mem':>11} {'padding_mem':>12} {'est_total':>11} {'fits_32gb':>10}"
  )
  print(header)
  budgets = []
  for batch_size, seq_len in shapes:
    b = estimate_for_shape(config, local_num_experts, batch_size, seq_len, capacity_factor)
    budgets.append(b)
    fits = b.estimated_total_gb < _ASSUMED_V6E_HBM_GB
    print(
        f"{b.batch_size:>6} {b.seq_len:>8} {b.num_tokens:>8} "
        f"{b.expected_local_assignments:>13.1f} {b.m_padded:>9} "
        f"{_fmt_mb(b.weight_memory_bytes):>11} {_fmt_mb(b.activation_memory_bytes):>11} "
        f"{_fmt_mb(b.padding_memory_bytes):>12} {b.estimated_total_gb:>10.2f}GB "
        f"{'yes' if fits else 'NO -- OOM'}"
    )

  first_oom = next((b for b in budgets if b.estimated_total_gb >= _ASSUMED_V6E_HBM_GB), None)
  if first_oom is not None:
    print(
        f"\n[memory-budget] recommended sweep upper bound: below num_tokens={first_oom.num_tokens} "
        f"(batch={first_oom.batch_size}, seq_len={first_oom.seq_len}) -- estimated total "
        f"({first_oom.estimated_total_gb:.2f}GB) already reaches the assumed {_ASSUMED_V6E_HBM_GB}GB ceiling, "
        "and real usage will only be higher than this lower-bound estimate."
    )
  else:
    print(
        f"\n[memory-budget] no shape in this sweep is estimated to exceed the assumed "
        f"{_ASSUMED_V6E_HBM_GB}GB ceiling -- consider extending _SWEEP_SHAPES further if a tighter "
        "upper bound is needed."
    )

  if output_dir is not None:
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "memory_budget.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
      writer = _csv.DictWriter(f, fieldnames=[
          "batch_size", "seq_len", "num_tokens", "expected_local_assignments", "m_padded",
          "weight_memory_bytes", "activation_memory_bytes", "padding_memory_bytes",
          "fp32_accumulator_bytes", "estimated_total_bytes", "estimated_total_gb", "fits_assumed_hbm",
      ])
      writer.writeheader()
      for b in budgets:
        writer.writerow({
            "batch_size": b.batch_size, "seq_len": b.seq_len, "num_tokens": b.num_tokens,
            "expected_local_assignments": b.expected_local_assignments, "m_padded": b.m_padded,
            "weight_memory_bytes": b.weight_memory_bytes, "activation_memory_bytes": b.activation_memory_bytes,
            "padding_memory_bytes": b.padding_memory_bytes, "fp32_accumulator_bytes": b.fp32_accumulator_bytes,
            "estimated_total_bytes": b.estimated_total_bytes, "estimated_total_gb": b.estimated_total_gb,
            "fits_assumed_hbm": b.estimated_total_gb < _ASSUMED_V6E_HBM_GB,
        })
    print(f"\n[memory-budget] structured data written to {csv_path}")
  return budgets


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-dir", type=pathlib.Path, default=None,
      help="if given, also write memory_budget.csv there",
  )
  args = parser.parse_args()

  print_budget_table(output_dir=args.output_dir)

  print(
      "\n[memory-budget] to get v6e's REAL per-chip HBM capacity (replacing the "
      f"{_ASSUMED_V6E_HBM_GB}GB public-spec assumption above) on the next TPU session, run:\n"
      "    import jax\n"
      "    print(jax.devices()[0].memory_stats())\n"
      "and look for the 'bytes_limit' (or equivalent) field in the returned dict."
  )
