"""WP-KV2 (KIMI_K3_LATENT_MOE_VALIDATION_PLAN_2026-08-26.md): generate a
golden bundle from the REAL official Kimi K3 PyTorch implementation
(official_kimi_k3/modeling_kimi_linear.py, pinned commit -- see
validate_official_config.py / MANIFEST.md for WP-KV1's provenance record).

Uses a SMALL config (matching this project's existing toy_config() in
kimi_k3_latent_moe_reference.py: hidden=64, latent=32, intermediate=48,
experts=8, top_k=2, shared=1) but with the REAL semantic settings confirmed
in WP-KV1 (moe_renormalize, sigmoid router, SiTU-GLU activation with the
real beta/linear_beta, float32 gate/RMSNorm precision, num_expert_group=1 so
the grouped-topk branch is genuinely dead code for Kimi K3 -- confirmed by
reading the real KimiMoEGate.forward, not assumed). Random weight init, NOT
the real (and MXFP4-quantized) checkpoint -- that's WP-KV6's job.

Two environment workarounds needed to even import the official module,
neither of which changes any MoE math (confirmed by reading the source):
  1. `fla-core` (Flash Linear Attention) is an unconditional top-level
     import in modeling_kimi_linear.py, but its actual kernels
     (FusedRMSNormGated/ShortConvolution/chunk_kda/fused_recurrent_kda) are
     used ONLY inside KimiDeltaAttention (linear attention), never by the
     MoE classes this script exercises (KimiMoEGate, KimiSparseMoeBlock,
     KimiBlockSparseMLP, KimiMLP, KimiRMSNorm, SituAndMul) -- and fla-core
     targets NVIDIA/Triton GPUs specifically, which neither this machine
     nor the v6e TPU VM has. Worked around via `fla_stub/` (a package of
     stub classes/functions that raise NotImplementedError if actually
     called, so an accidental attention-path exercise fails loudly rather
     than silently).
  2. The installed `transformers` (newer than what this file was written
     against) moved `OutputRecorder` from `transformers.utils.generic` to
     `transformers.utils.output_capturing`; `check_model_inputs` is
     unaffected. One-line compat shim below, not a stub (the real symbol,
     just re-exposed at its old import path).

Instrumentation: nn.Module forward hooks only see a submodule's own
input/output, not local variables inside a method body -- and several
requested intermediates (router_logits, router_scores_before_bias,
sorted_token_indices, situ_output, ...) are local variables inside
KimiMoEGate.forward / KimiSparseMoeBlock.moe_infer, not separate submodule
calls. So `_run_instrumented` below is a line-by-line copy of the REAL
official forward/moe_infer control flow (confirmed against the actual
source, not reconstructed from memory), calling the SAME real submodule
instances (block.gate, block.experts[i], block.routed_expert_*,
block.shared_experts) at each step and stashing intermediates as it goes --
not a reimplementation of the math, just added visibility into an
unmodified call sequence. Cross-checked against calling `block(hidden_states)`
directly (see main()) to catch any transcription mistake.

Usage:
  python generate_pytorch_golden.py
"""

import hashlib
import json
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE / "fla_stub"))
sys.path.insert(0, str(_HERE))

from validate_official_config import DEFAULT_COMMIT  # noqa: E402

# Workaround 2: OutputRecorder moved between transformers versions -- see
# module docstring. Must run before importing modeling_kimi_linear.
import transformers.utils.generic as _generic  # noqa: E402
from transformers.utils.output_capturing import OutputRecorder as _OutputRecorder  # noqa: E402

_generic.OutputRecorder = _OutputRecorder

from official_kimi_k3.configuration_kimi_k3 import KimiLinearConfig  # noqa: E402
from official_kimi_k3.modeling_kimi_linear import KimiSparseMoeBlock  # noqa: E402

BUNDLE_DIR = _HERE / "golden_bundle_small"

SMALL_CONFIG_KWARGS = dict(
    hidden_size=64,
    routed_expert_hidden_size=32,
    moe_intermediate_size=48,
    intermediate_size=48,  # KimiMLP's default intermediate_size fallback; unused here since shared_experts passes its own
    num_experts=8,
    num_experts_per_token=2,
    num_shared_experts=1,
    moe_renormalize=True,
    moe_router_activation_func="sigmoid",
    routed_scaling_factor=1.0,
    rms_norm_eps=1e-5,
    latent_moe_use_norm=True,
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    hidden_act="situ",
    num_expert_group=1,
    topk_group=1,
)

# Dims must be identical to kimi_k3_latent_moe_ragged_dot.py's own
# mosaic_correctness_config() (hidden=256/latent=128/intermediate=128) --
# Mosaic's ragged_dot kernel enforces a hard >=128 floor on every matmul
# dim (confirmed on hardware: `NotImplementedError: RaggedDot inputs must
# be >= 128` at the SMALL_CONFIG_KWARGS scale above), so WP-KV4 needs a
# SEPARATE golden bundle at this larger scale specifically to validate
# Mosaic v1/v2 -- the small bundle can only ever validate xla.
MOSAIC_CONFIG_KWARGS = dict(
    hidden_size=256,
    routed_expert_hidden_size=128,
    moe_intermediate_size=128,
    intermediate_size=128,
    num_experts=8,
    num_experts_per_token=2,
    num_shared_experts=1,
    moe_renormalize=True,
    moe_router_activation_func="sigmoid",
    routed_scaling_factor=1.0,
    rms_norm_eps=1e-5,
    latent_moe_use_norm=True,
    activation_situ_beta=4.0,
    activation_situ_linear_beta=25.0,
    hidden_act="situ",
    num_expert_group=1,
    topk_group=1,
)


def _to_np(t: torch.Tensor) -> np.ndarray:
  return t.detach().cpu().to(torch.float32).numpy()


def _sha256(path: pathlib.Path) -> str:
  return hashlib.sha256(path.read_bytes()).hexdigest()


def _run_instrumented(block: KimiSparseMoeBlock, hidden_states: torch.Tensor):
  """Line-by-line copy of KimiMoEGate.forward + KimiSparseMoeBlock.forward/
  moe_infer, calling the same real submodule instances, stashing every
  intermediate the validation plan's WP-KV2 section asks for. See module
  docstring for why hooks alone can't do this."""
  intermediates = {}
  identity = hidden_states
  orig_shape = hidden_states.shape

  # --- KimiMoEGate.forward ---
  gate = block.gate
  bsz, seq_len, h = hidden_states.shape
  gate_input = hidden_states.view(-1, h)
  logits = F.linear(gate_input.to(torch.float32), gate.weight.to(torch.float32), None)
  intermediates["router_logits"] = logits

  if gate.moe_router_activation_func == "sigmoid":
    scores = logits.sigmoid()
  elif gate.moe_router_activation_func == "softmax":
    scores = logits.softmax(dim=1)
  else:
    raise NotImplementedError(gate.moe_router_activation_func)
  intermediates["router_scores_before_bias"] = scores

  scores = scores.view(bsz * seq_len, -1)
  scores_for_choice = scores + gate.e_score_correction_bias.unsqueeze(0)
  intermediates["router_scores_for_choice"] = scores_for_choice

  # num_expert_group=1 for Kimi K3's real config -> the grouped-topk branch
  # in the official KimiMoEGate.forward is dead code (confirmed via
  # WP-KV1's field check: num_expert_group=1, topk_group=1, and the branch
  # condition is `num_expert_group > 1 and num_expert_group > topk_group`).
  # Not replicated here since it never executes for this model.
  tmp_scores = scores_for_choice

  _, topk_idx = torch.topk(tmp_scores, k=gate.top_k, dim=-1, sorted=False)
  topk_weight = scores.gather(1, topk_idx)
  if gate.top_k > 1 and gate.moe_renormalize:
    denominator = topk_weight.sum(dim=-1, keepdim=True) + 1e-20
    topk_weight = topk_weight / denominator
  topk_weight = topk_weight * gate.routed_scaling_factor
  intermediates["topk_indices"] = topk_idx
  intermediates["topk_weights"] = topk_weight

  # --- KimiSparseMoeBlock.forward (pre-expert part) ---
  hs = hidden_states.view(-1, hidden_states.shape[-1])
  if block.use_latent_moe:
    hs = block.routed_expert_down_proj(hs)
  intermediates["latent_projection"] = hs

  # --- moe_infer ---
  x = hs
  cnts = topk_idx.new_zeros((topk_idx.shape[0], len(block.experts)))
  cnts.scatter_(1, topk_idx, 1)
  tokens_per_expert = cnts.sum(dim=0)
  idxs = topk_idx.view(-1).argsort()
  sorted_tokens = x[idxs // topk_idx.shape[1]]
  intermediates["sorted_token_indices"] = idxs
  intermediates["group_sizes"] = tokens_per_expert

  tokens_per_expert_np = tokens_per_expert.cpu().numpy()

  outputs, gate_outs, up_outs, situ_outs, down_outs = [], [], [], [], []
  start_idx = 0
  for i, num_tokens in enumerate(tokens_per_expert_np):
    end_idx = start_idx + int(num_tokens)
    if num_tokens == 0:
      continue
    expert = block.experts[i]
    tokens_for_this_expert = sorted_tokens[start_idx:end_idx]
    # KimiBlockSparseMLP.forward, expanded for per-step intermediate capture.
    gate_out = expert.w1(tokens_for_this_expert)
    up_out = expert.w3(tokens_for_this_expert)
    gate_up = torch.cat([gate_out, up_out], dim=-1)
    situ_out = expert.act_fn(gate_up)
    expert_out = expert.w2(situ_out)
    gate_outs.append(gate_out)
    up_outs.append(up_out)
    situ_outs.append(situ_out)
    down_outs.append(expert_out)
    outputs.append(expert_out)
    start_idx = end_idx

  intermediates["expert_gate_output"] = torch.cat(gate_outs, dim=0)
  intermediates["expert_up_output"] = torch.cat(up_outs, dim=0)
  intermediates["situ_output"] = torch.cat(situ_outs, dim=0)
  intermediates["expert_down_output"] = torch.cat(down_outs, dim=0)

  outs = torch.cat(outputs, dim=0)
  new_x = torch.empty_like(outs)
  new_x[idxs] = outs
  # .clone() is load-bearing, not defensive style: when new_x's dtype
  # already equals topk_weight.dtype (true for this FP32 bundle, since both
  # are float32), `.type(topk_weight.dtype)` below is a no-op that returns
  # a tensor sharing new_x's storage (PyTorch's `.type()` returns `self` if
  # the dtype already matches) -- the following `.mul_()` is in-place, so
  # without cloning here, the stored "routed_output_before_combine"
  # reference gets silently overwritten in place by the *_combined_* value
  # once `final_out` is computed below. Caught this via WP-KV3: JAX's
  # routed_output_before_combine matched a from-scratch manual FFN
  # computation exactly, while this array's stored value equaled
  # FFN_output * topk_weight -- i.e. exactly the post-multiply value.
  intermediates["routed_output_before_combine"] = new_x.clone()

  final_out = (
      new_x.view(*topk_idx.shape, -1)
      .type(topk_weight.dtype)
      .mul_(topk_weight.unsqueeze(dim=-1))
      .sum(dim=1)
      .type(new_x.dtype)
  )
  intermediates["routed_output_after_combine"] = final_out

  # --- back in KimiSparseMoeBlock.forward ---
  y = final_out
  if block.use_latent_moe:
    if block.latent_moe_use_norm:
      y = block.routed_expert_norm(y)
    intermediates["normalized_output"] = y
    y = block.routed_expert_up_proj(y)
    intermediates["up_projection_output"] = y

  y = y.view(*orig_shape)

  shared_out = block.shared_experts(identity)
  intermediates["shared_expert_output"] = shared_out

  final_output = y + shared_out
  intermediates["final_output"] = final_output

  return final_output, intermediates


def main(
    seed: int = 0,
    dtype: torch.dtype = torch.float32,
    bundle_dir: pathlib.Path | None = None,
    config_kwargs: dict | None = None,
) -> None:
  """dtype=bfloat16 exercises the router/RMSNorm/combine "compute in
  float32, cast back to compute dtype" logic for real -- with the default
  float32, that upcast is a no-op (already float32), so it can't by itself
  catch a bug in the cast-back-down path. Uses a SEPARATE bundle_dir per
  dtype so the float32 bundle (already validated against this project's
  JAX reference via WP-KV3) is never overwritten by a bf16 run.

  `config_kwargs` defaults to SMALL_CONFIG_KWARGS (dims below Mosaic's
  128-tiling floor, xla-only); pass MOSAIC_CONFIG_KWARGS for a bundle whose
  dims satisfy that floor, needed to validate mosaic/mosaic_tpu_v2 (WP-KV4).
  """
  bundle_dir = bundle_dir or BUNDLE_DIR
  config_kwargs = config_kwargs or SMALL_CONFIG_KWARGS
  torch.manual_seed(seed)

  config = KimiLinearConfig(**config_kwargs)
  block = KimiSparseMoeBlock(config)
  block.eval()
  block = block.to(dtype)

  num_tokens = 20
  hidden_states = torch.randn(1, num_tokens, config.hidden_size).to(dtype)

  with torch.no_grad():
    instrumented_output, intermediates = _run_instrumented(block, hidden_states)
    direct_output = block(hidden_states)

  max_err = (instrumented_output.float() - direct_output.float()).abs().max().item()
  print(f"[generate-golden] instrumented vs. direct block(hidden_states) max_err={max_err:.2e}")
  assert max_err == 0.0, (
      "instrumented replay diverged from calling the official block directly -- "
      "there's a transcription bug in _run_instrumented, fix before trusting this bundle"
  )

  bundle_dir.mkdir(parents=True, exist_ok=True)

  np.savez(bundle_dir / "inputs.npz", hidden_states=_to_np(hidden_states))

  weights = {name: _to_np(param) for name, param in block.state_dict().items()}
  np.savez(bundle_dir / "weights.npz", **weights)

  outputs = {name: _to_np(t) if t.dtype != torch.int64 else t.detach().cpu().numpy() for name, t in intermediates.items()}
  outputs["final_output"] = _to_np(direct_output)
  np.savez(bundle_dir / "golden_outputs.npz", **outputs)

  (bundle_dir / "config.json").write_text(json.dumps(config_kwargs, indent=2), encoding="utf-8")

  # Recomputed fresh from the actual imported files (not hardcoded) so this
  # can't silently drift from validate_official_config.py's own MANIFEST.md
  # -- if WP-KV1 is ever re-run against a different commit, this picks up
  # the new hashes automatically instead of needing a manual update here.
  official_dir = _HERE / "official_kimi_k3"
  source_provenance = {
      "pinned_commit": DEFAULT_COMMIT,
      "repo": "moonshotai/Kimi-K3",
      "file_sha256": {
          "config.json": _sha256(official_dir / "config.json"),
          "modeling_kimi_linear.py": _sha256(official_dir / "modeling_kimi_linear.py"),
      },
  }

  metadata = {
      "seed": seed,
      "dtype": str(dtype),
      "torch_version": torch.__version__,
      "transformers_version": __import__("transformers").__version__,
      "num_tokens": num_tokens,
      "small_config": config_kwargs,
      "instrumented_vs_direct_max_err": max_err,
      "intermediate_keys": sorted(intermediates.keys()),
      "source_provenance": source_provenance,
      "note": "Random weight init -- NOT the real (MXFP4-quantized) Kimi K3 checkpoint. "
      "See WP-KV6 for real-checkpoint validation.",
  }
  (bundle_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

  print(f"[generate-golden] bundle written to {bundle_dir}")
  print(f"[generate-golden] intermediates captured: {sorted(intermediates.keys())}")


if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
  parser.add_argument(
      "--config",
      choices=["small", "mosaic"],
      default="small",
      help="'small' (64/32/48, xla-only -- below Mosaic's 128-tiling floor) or "
      "'mosaic' (256/128/128, satisfies the floor -- needed to validate mosaic/mosaic_tpu_v2)",
  )
  args = parser.parse_args()

  torch_dtype = torch.float32 if args.dtype == "fp32" else torch.bfloat16
  cfg_kwargs = SMALL_CONFIG_KWARGS if args.config == "small" else MOSAIC_CONFIG_KWARGS
  dir_name = f"golden_bundle_{args.config}_{args.dtype}" if args.config == "mosaic" else (
      "golden_bundle_small" if args.dtype == "fp32" else "golden_bundle_small_bf16"
  )
  main(dtype=torch_dtype, bundle_dir=_HERE / dir_name, config_kwargs=cfg_kwargs)
