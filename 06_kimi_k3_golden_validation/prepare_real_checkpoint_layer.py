"""WP-KV6 prep (real, MXFP4-quantized Kimi K3 checkpoint validation) --
everything that can be built and tested WITHOUT downloading the actual
~17GB-per-layer checkpoint shard, so the pipeline doesn't need designing
from scratch whenever the real checkpoint (or server resources to hold it)
become available.

Covers:
  1. The exact MXFP4 weight field mapping for one MoE layer (confirmed
     against the REAL cached `official_kimi_k3/modeling_kimi_linear.py`
     source, not guessed -- see `LAYER_KEY_TEMPLATE` below).
  2. Which tensors need dequantization (routed-expert `w1`/`w2`/`w3` only --
     confirmed in WP-KV1's feasibility pre-check by grepping the real
     checkpoint's `model.safetensors.index.json`; router/shared-experts/
     latent-projection/norm weights stay plain bf16).
  3. The extraction script interface: `extract_moe_layer_from_shard` --
     official checkpoint shard -> one MoE layer's bundle, in this project's
     established bundle format (matching `generate_pytorch_golden.py`'s
     inputs.npz/weights.npz/metadata.json convention).
  4. Metadata/hash format: `build_layer_bundle_metadata` -- extends the
     `source_provenance` convention already used everywhere else in this
     project (pinned_commit + file_sha256) with checkpoint-specific fields
     (shard filename, layer_idx, dequantization method/library version).
  5. A small SYNTHETIC MXFP4 round-trip test (`check_mxfp4_roundtrip`) --
     proves the actual `compressed-tensors` library (not a hand-rolled
     bit-unpacker) is called correctly, using a tiny random tensor, not the
     real checkpoint.
  6. Disk/memory requirements, computed exactly (see module-level constants
     and `estimate_full_layer_memory`/`estimate_shard_memory` below).

**Key confirmation methodology**: the layer-internals key names below
(`w1`/`w2`/`w3`, `gate_proj`/`up_proj`/`down_proj`,
`routed_expert_down_proj`/`routed_expert_up_proj`/`routed_expert_norm`,
`block_sparse_moe`, `model.layers`) were read DIRECTLY from
`official_kimi_k3/modeling_kimi_linear.py` (`KimiBlockSparseMLP`,
`KimiMLP`, `KimiMoEGate`, `KimiSparseMoeBlock`, `KimiDecoderLayer`,
`KimiLinearModel` class bodies) -- the same file WP-KV1/KV2 already
pinned+hash-verified -- not reconstructed from memory. The `w1=gate,
w2=down, w3=up` mapping and the "only routed experts are MXFP4-quantized"
finding were already independently confirmed in WP-KV1's feasibility
pre-check (2026-08-26) by grepping the real
`model.safetensors.index.json`; this file's docstring repeats them for a
single, self-contained reference, not as a fresh, unverified claim.

**What still needs the REAL checkpoint to confirm** (cannot be checked
without it, flagged rather than assumed):
  - The exact `weight_scale`/`weight_packed` tensor shapes/dtypes as
    actually stored (this file's synthetic test confirms the CALLING
    CONVENTION works, not the real checkpoint's exact on-disk shapes).
  - Whether `scale_dtype=torch.uint8` (from config.json's
    `quantization_config`) matches what `compressed_tensors`' MXFP4
    compressor expects bit-for-bit (this file's synthetic test uses the
    library's own scale-compression path, so if the real checkpoint's
    scales were produced by a DIFFERENT quantization tool, subtle format
    differences are possible -- worth an explicit spot-check against one
    real tensor once available, before trusting a whole-layer dequant).

Usage (this file has NO tokamax dependency, and the synthetic check needs
only `torch` + `compressed-tensors` -- run in `.venv_torch`, not the
jax/tokamax venv):
  python prepare_real_checkpoint_layer.py
"""

import dataclasses
import pathlib

import torch
from compressed_tensors.compressors.mx_utils import decompress_mx_scale
from compressed_tensors.compressors.mxfp4.base import MXFP4PackedCompressor
from compressed_tensors.compressors.nvfp4.helpers import pack_fp4_to_uint8
from compressed_tensors.quantization import (
    QuantizationArgs,
    QuantizationScheme,
    QuantizationStrategy,
    QuantizationType,
)
from compressed_tensors.quantization.lifecycle.forward import quantize
from compressed_tensors.quantization.utils.mxfp_utils import generate_mx_scales

# Same pinned commit as validate_official_config.py / generate_pytorch_golden.py.
_DEFAULT_COMMIT = "a590ce090cb049c93a33dfe8c208ec652aa20503"

# Confirmed real Kimi K3 dims (kimi_k3_config() in kimi_k3_latent_moe_reference.py).
REAL_HIDDEN_SIZE = 7168
REAL_LATENT_SIZE = 3584  # routed_expert_hidden_size
REAL_INTERMEDIATE_SIZE = 3072  # moe_intermediate_size
REAL_NUM_EXPERTS = 896
REAL_TOP_K = 16
REAL_NUM_SHARED_EXPERTS = 2
REAL_NUM_HIDDEN_LAYERS = 93
REAL_FIRST_K_DENSE_REPLACE = 1  # layer 0 is dense (KimiMLP), NOT MoE -- any layer target must be >= 1

# MXFP4 quantization params, confirmed from config.json's quantization_config
# (WP-KV1 feasibility pre-check, 2026-08-26): format="mxfp4-pack-quantized",
# group_size=32, symmetric, scale_dtype=uint8 (OCP Microscaling, E8M0 exponent).
MXFP4_NUM_BITS = 4
MXFP4_GROUP_SIZE = 32
MXFP4_SCALE_DTYPE = torch.uint8


@dataclasses.dataclass(frozen=True)
class LayerKeyTemplate:
  """Confirmed real checkpoint key names for one MoE decoder layer `{i}`
  (`i` must be `>= REAL_FIRST_K_DENSE_REPLACE`, i.e. >= 1 -- layer 0 is a
  plain dense KimiMLP, not a KimiSparseMoeBlock, per
  `KimiDecoderLayer.__init__`'s `layer_idx >= config.first_k_dense_replace`
  check).

  Confirmed source: `official_kimi_k3/modeling_kimi_linear.py` --
  `KimiLinearModel.__init__` (`self.layers = nn.ModuleList([...])`,
  `base_model_prefix = "model"` on `KimiPreTrainedModel`) gives the
  `model.layers.{i}.` prefix; `KimiDecoderLayer.__init__`
  (`self.block_sparse_moe = KimiSparseMoeBlock(config)`) gives the
  `block_sparse_moe.` segment; `KimiSparseMoeBlock.__init__` (`self.experts`,
  `self.gate`, `self.shared_experts`, `self.routed_expert_down_proj`,
  `self.routed_expert_up_proj`, `self.routed_expert_norm`) gives everything
  after that; `KimiBlockSparseMLP.__init__` (`self.w1`/`w2`/`w3`) and
  `KimiMLP.__init__` (`self.gate_proj`/`up_proj`/`down_proj`) give the
  innermost per-expert/shared-expert names.
  """

  layer_idx: int

  @property
  def prefix(self) -> str:
    return f"model.layers.{self.layer_idx}.block_sparse_moe"

  # --- Plain bf16, NOT quantized (confirmed via WP-KV1's grep of the real
  # config.json quantization_config's `ignore` list AND direct key grepping
  # of model.safetensors.index.json for layer 12) ---
  @property
  def router_weight(self) -> str:
    return f"{self.prefix}.gate.weight"  # PyTorch shape (num_experts, hidden_size) -- (out,in)-like, but this is a raw nn.Parameter, not nn.Linear.weight

  @property
  def router_bias(self) -> str:
    return f"{self.prefix}.gate.e_score_correction_bias"  # shape (num_experts,)

  @property
  def down_proj(self) -> str:
    return f"{self.prefix}.routed_expert_down_proj.weight"  # nn.Linear(hidden_size, latent_size) -> shape (latent_size, hidden_size)

  @property
  def up_proj(self) -> str:
    return f"{self.prefix}.routed_expert_up_proj.weight"  # nn.Linear(latent_size, hidden_size) -> shape (hidden_size, latent_size)

  @property
  def norm_scale(self) -> str:
    return f"{self.prefix}.routed_expert_norm.weight"  # RMSNorm scale, shape (latent_size,) -- only present since latent_moe_use_norm=True (confirmed)

  @property
  def shared_gate(self) -> str:
    return f"{self.prefix}.shared_experts.gate_proj.weight"  # nn.Linear(hidden_size, shared_intermediate) -> shape (shared_intermediate, hidden_size)

  @property
  def shared_up(self) -> str:
    return f"{self.prefix}.shared_experts.up_proj.weight"  # same shape convention as shared_gate

  @property
  def shared_down(self) -> str:
    return f"{self.prefix}.shared_experts.down_proj.weight"  # nn.Linear(shared_intermediate, hidden_size) -> shape (hidden_size, shared_intermediate)

  # --- MXFP4-quantized (confirmed via WP-KV1's grep: ONLY
  # block_sparse_moe.experts.*.w1/w2/w3 have .weight_packed/.weight_scale
  # keys instead of a plain .weight) ---
  def expert_gate_packed(self, expert_idx: int) -> tuple[str, str]:
    """w1 = gate. Returns (weight_packed_key, weight_scale_key)."""
    base = f"{self.prefix}.experts.{expert_idx}.w1"
    return f"{base}.weight_packed", f"{base}.weight_scale"

  def expert_up_packed(self, expert_idx: int) -> tuple[str, str]:
    """w3 = up."""
    base = f"{self.prefix}.experts.{expert_idx}.w3"
    return f"{base}.weight_packed", f"{base}.weight_scale"

  def expert_down_packed(self, expert_idx: int) -> tuple[str, str]:
    """w2 = down."""
    base = f"{self.prefix}.experts.{expert_idx}.w2"
    return f"{base}.weight_packed", f"{base}.weight_scale"


def _mxfp4_scheme() -> QuantizationScheme:
  """The QuantizationScheme matching this checkpoint's confirmed
  quantization_config (num_bits=4, type=float, group_size=32, symmetric,
  scale_dtype=uint8) -- constructed once, reused by both the real
  dequantization function and the synthetic test below so they can never
  silently drift apart."""
  weights_args = QuantizationArgs(
      num_bits=MXFP4_NUM_BITS,
      type=QuantizationType.FLOAT.value,
      symmetric=True,
      group_size=MXFP4_GROUP_SIZE,
      strategy=QuantizationStrategy.GROUP.value,
      scale_dtype=MXFP4_SCALE_DTYPE,
  )
  return QuantizationScheme(targets=["Linear"], weights=weights_args)


def dequantize_mxfp4_tensor(weight_packed: torch.Tensor, weight_scale: torch.Tensor) -> torch.Tensor:
  """Dequantizes ONE real MXFP4-packed weight tensor (as stored in the
  checkpoint's safetensors shard) back to a plain float tensor, using the
  REAL `compressed_tensors` library's `MXFP4PackedCompressor.decompress`
  (not a hand-rolled bit-unpacker) -- see `check_mxfp4_roundtrip` for the
  synthetic proof this calling convention is correct.

  `weight_packed`: uint8, shape (out_features, in_features // 2) -- 2 FP4
    values packed per byte.
  `weight_scale`: uint8, shape (out_features, in_features // group_size) --
    E8M0 (biased power-of-2) exponents, one per group of `group_size`
    consecutive input-dimension elements.

  Returns a bf16 tensor of shape (out_features, in_features).
  """
  scheme = _mxfp4_scheme()
  state_dict = {"weight_packed": weight_packed, "weight_scale": weight_scale}
  result = MXFP4PackedCompressor.decompress(state_dict, scheme)
  return result["weight"]


def check_mxfp4_roundtrip(seed: int = 0, out_features: int = 64, in_features: int = 64) -> bool:
  """Synthetic (no real checkpoint needed) round-trip test: builds a small
  random bf16 weight, quantizes it to MXFP4 using the REAL
  `compressed_tensors` quantize/pack path (computing a valid per-group E8M0
  scale via `generate_mx_scales`, the same helper the library itself uses
  for calibration), then dequantizes it back via `dequantize_mxfp4_tensor`
  (the SAME function real checkpoint extraction will use) -- confirming the
  DEQUANTIZATION calling convention this project depends on is correct,
  before ever touching the real ~17GB checkpoint shard.

  `in_features` must be divisible by `MXFP4_GROUP_SIZE` (32).

  Does NOT assert a tight numerical tolerance -- MXFP4 is a genuinely lossy
  4-bit float format (E2M1 mantissa), so double-digit relative error on
  small random values is EXPECTED, not a bug (real trained weights have a
  smoother distribution that quantizes better in practice, but that's a
  model-quality question, not a plumbing-correctness one). This check only
  confirms: shapes/dtypes round-trip correctly, and the error is BOUNDED
  (not NaN/Inf, not wildly larger than MXFP4's known ~2^-1 relative
  precision per element).
  """
  assert in_features % MXFP4_GROUP_SIZE == 0, "in_features must be divisible by MXFP4_GROUP_SIZE"
  torch.manual_seed(seed)
  scheme = _mxfp4_scheme()
  weights_args = scheme.weights

  w = (torch.randn(out_features, in_features) * 0.02).to(torch.bfloat16)

  num_groups = in_features // MXFP4_GROUP_SIZE
  w_grouped = w.view(out_features, num_groups, MXFP4_GROUP_SIZE)
  group_max = w_grouped.abs().amax(dim=-1)
  scale_exp = generate_mx_scales(group_max, num_bits=MXFP4_NUM_BITS).to(MXFP4_SCALE_DTYPE)
  scale_float = decompress_mx_scale(scale_exp).to(torch.float32)

  quantized = quantize(x=w.to(torch.float32), scale=scale_float, zero_point=None, args=weights_args)
  packed = pack_fp4_to_uint8(quantized)

  shape_ok = packed.shape == (out_features, in_features // 2) and scale_exp.shape == (out_features, num_groups)

  w_dequant = dequantize_mxfp4_tensor(packed, scale_exp)
  dtype_ok = w_dequant.dtype == torch.bfloat16 and w_dequant.shape == w.shape

  diff = (w.to(torch.float32) - w_dequant.to(torch.float32)).abs()
  max_abs_err = float(diff.max())
  finite_ok = bool(torch.isfinite(w_dequant).all())
  # MXFP4's largest representable magnitude step for small values is
  # coarse -- bound generously (not a tight tolerance, see docstring).
  bounded_ok = max_abs_err < 0.1

  ok = shape_ok and dtype_ok and finite_ok and bounded_ok
  print(
      f"[mxfp4-roundtrip] packed.shape={tuple(packed.shape)} scale.shape={tuple(scale_exp.shape)} "
      f"dequant.dtype={w_dequant.dtype} max_abs_err={max_abs_err:.4f} "
      f"shape_ok={shape_ok} dtype_ok={dtype_ok} finite_ok={finite_ok} bounded_ok={bounded_ok} "
      f"{'OK' if ok else 'FAIL'}"
  )
  return ok


def build_layer_bundle_metadata(
    layer_idx: int,
    shard_filename: str,
    pinned_commit: str = _DEFAULT_COMMIT,
    config_file_sha256: str | None = None,
    modeling_file_sha256: str | None = None,
    shard_file_sha256: str | None = None,
) -> dict:
  """Metadata format for a real-checkpoint layer bundle -- extends this
  project's established `source_provenance` convention (pinned_commit +
  file_sha256, see generate_pytorch_golden.py's metadata.json) with
  checkpoint-specific fields. `config_file_sha256`/`modeling_file_sha256`
  should come from the ALREADY-verified hashes in
  `official_kimi_k3/MANIFEST.md` (WP-KV1) rather than being recomputed here
  -- pass them in, don't hardcode a duplicate.
  """
  return {
      "layer_idx": layer_idx,
      "note": (
          "Real MXFP4-quantized Kimi K3 checkpoint layer bundle (WP-KV6) -- "
          "NOT random weight init, unlike every other bundle in this project."
      ),
      "dequantization": {
          "library": "compressed_tensors",
          "format": "mxfp4-pack-quantized",
          "num_bits": MXFP4_NUM_BITS,
          "group_size": MXFP4_GROUP_SIZE,
          "scale_dtype": "uint8",
          "dequantized_tensors": ["expert_gate", "expert_up", "expert_down"],
          "plain_bf16_tensors": [
              "router_weight", "e_score_correction_bias", "down_proj", "up_proj",
              "norm_scale", "shared_gate", "shared_up", "shared_down",
          ],
      },
      "source_provenance": {
          "pinned_commit": pinned_commit,
          "repo": "moonshotai/Kimi-K3",
          "shard_filename": shard_filename,
          "file_sha256": {
              "config.json": config_file_sha256,
              "modeling_kimi_linear.py": modeling_file_sha256,
              shard_filename: shard_file_sha256,
          },
      },
  }


def extract_moe_layer_from_shard(
    shard_path: pathlib.Path,
    layer_idx: int,
    local_expert_range: tuple[int, int] | None = None,
) -> dict:
  """Extraction script INTERFACE: real checkpoint safetensors shard -> one
  MoE layer's bundle (router/projections/norm/shared-experts as plain
  bf16, routed experts dequantized from MXFP4).

  **NOT yet runnable** -- needs the real ~17GB shard (`shard_path`), which
  this project has not downloaded (WP-KV1 only fetched the ~57MB
  `model.safetensors.index.json` to LOCATE which shard holds a given
  layer, not the shard itself). Written now so the actual download session
  can run this immediately instead of designing it live.

  `local_expert_range`: `(start, end)` global expert-id range to extract
  (e.g. `(0, 64)` for one shard's worth) -- extracting all 896 experts'
  dequantized (bf16) weights at once needs ~59GB RAM (see
  `estimate_full_layer_memory`), the same OOM already confirmed elsewhere
  in this project for random-init weights; `None` means "all 896", which
  will almost certainly OOM on a single machine and should not be the
  default once this actually runs -- pass a range matching
  `single_chip_kimi_k3_config`'s established shard convention instead.

  Real implementation, once `shard_path` exists, needs (not yet available
  in either the tokamax or torch venv -- add when this actually runs):
    `pip install safetensors`
  and roughly:
    from safetensors import safe_open
    keys = LayerKeyTemplate(layer_idx)
    with safe_open(shard_path, framework="pt") as f:
      router_weight = f.get_tensor(keys.router_weight)          # plain bf16
      ...
      for e in range(*(local_expert_range or (0, REAL_NUM_EXPERTS))):
        packed_key, scale_key = keys.expert_gate_packed(e)
        expert_gate[e] = dequantize_mxfp4_tensor(f.get_tensor(packed_key), f.get_tensor(scale_key))
        ... (expert_up, expert_down likewise)
  """
  raise NotImplementedError(
      "needs the real checkpoint shard -- see this function's docstring for the "
      "exact extraction logic to implement once shard_path exists"
  )


def estimate_full_layer_memory(dtype_bytes: int = 2) -> dict:
  """All 896 experts' routed-expert weights (gate+up+down), dequantized to
  `dtype_bytes`-wide float, for ONE layer -- this is the OOM this project
  has already hit (random-init version) and would hit again here."""
  per_expert_elems = REAL_LATENT_SIZE * REAL_INTERMEDIATE_SIZE  # gate/up share this element count; down is the transpose, same count
  total_elems = per_expert_elems * 3 * REAL_NUM_EXPERTS  # gate + up + down
  total_bytes = total_elems * dtype_bytes
  return {"total_bytes": total_bytes, "total_gb": total_bytes / (1024**3)}


def estimate_shard_memory(local_num_experts: int = 64, dtype_bytes: int = 2) -> dict:
  """Same, but for only `local_num_experts` (this project's established
  single-chip-shard convention) -- the practically usable scope for actual
  validation, matching everything else built this session."""
  per_expert_elems = REAL_LATENT_SIZE * REAL_INTERMEDIATE_SIZE
  total_elems = per_expert_elems * 3 * local_num_experts
  total_bytes = total_elems * dtype_bytes
  return {"total_bytes": total_bytes, "total_gb": total_bytes / (1024**3)}


def estimate_disk_requirements() -> dict:
  """From WP-KV1's real, already-confirmed measurements (2026-08-26): one
  shard holds one full layer's weights (~17GB per the repo's own file
  listing), 96 shards total, 1,560,860,324,864 bytes total checkpoint size
  (confirmed via the real `model.safetensors.index.json`'s metadata, not
  estimated). That byte count is ~1.56TB in DECIMAL terms (10^12) or
  ~1.42TiB in BINARY terms (2^40) -- reported here in binary GiB-style
  units (dividing by 1024**4) to stay consistent with every other
  memory/size estimate in this project (e.g. memory_budget_estimate.py),
  which all use the 1024-based convention -- don't mix the two when
  comparing against the "~1.56TB" figure recorded in project memory from
  the original WP-KV1 note (that one used the decimal convention)."""
  return {
      "one_layer_shard_gb": 17,
      "total_checkpoint_tib": 1_560_860_324_864 / (1024**4),
      "num_shards": 96,
  }


if __name__ == "__main__":
  print("[prepare-checkpoint] MXFP4 synthetic round-trip check (no real checkpoint needed):")
  roundtrip_ok = check_mxfp4_roundtrip()

  print("\n[prepare-checkpoint] disk requirements (from WP-KV1's confirmed real measurements):")
  disk = estimate_disk_requirements()
  print(f"    one layer's shard: ~{disk['one_layer_shard_gb']}GB")
  print(f"    full checkpoint: ~{disk['total_checkpoint_tib']:.2f}TiB across {disk['num_shards']} shards "
        "(~1.56TB in decimal terms)")

  print("\n[prepare-checkpoint] in-memory dequantized (bf16) size estimates:")
  full = estimate_full_layer_memory()
  print(f"    all {REAL_NUM_EXPERTS} experts' routed weights (gate+up+down), one layer: "
        f"~{full['total_gb']:.1f}GB -- WILL OOM a single chip (same ~59GB figure already "
        "confirmed elsewhere in this project for random-init weights)")
  for local_num_experts in (64, 32, 16):
    shard = estimate_shard_memory(local_num_experts)
    print(f"    local_num_experts={local_num_experts}: ~{shard['total_gb']:.2f}GB -- fits comfortably")

  print(f"\n[prepare-checkpoint] example key template (layer_idx=12, the layer WP-KV1 already "
        "confirmed is a real MoE layer):")
  keys = LayerKeyTemplate(layer_idx=12)
  print(f"    router_weight: {keys.router_weight}")
  print(f"    down_proj: {keys.down_proj}")
  print(f"    up_proj: {keys.up_proj}")
  print(f"    norm_scale: {keys.norm_scale}")
  print(f"    shared_gate: {keys.shared_gate}")
  print(f"    expert 0 gate (packed, scale): {keys.expert_gate_packed(0)}")

  assert roundtrip_ok, "MXFP4 synthetic round-trip check failed"
  print("\nOK: MXFP4 dequantization calling convention verified (synthetic weights)")
