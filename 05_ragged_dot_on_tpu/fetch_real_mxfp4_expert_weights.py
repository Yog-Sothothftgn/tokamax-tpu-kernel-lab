"""WP-KV6 step 3 (2026-09-16): fetches ONE real MoE layer's worth of real,
MXFP4-quantized expert weights from the official `moonshotai/Kimi-K3` HF
repo -- without downloading the full ~2.8T-param checkpoint, or even the
full ~15GB shard file these tensors live in.

How: safetensors files store a JSON header (length-prefixed) mapping each
tensor name to an exact byte range within the file. This project's own
research (2026-09-15/16, see project memory) confirmed via a real fetch that
ALL of `language_model.model.layers.1.block_sparse_moe.experts.*`'s tensors
(w1/w2/w3, each `weight_packed` + `weight_scale`) live in ONE shard file
(`model-00002-of-000096.safetensors`), and that each INDIVIDUAL expert's 6
tensors are stored back-to-back with zero gap (confirmed: exactly
17,547,264 bytes per expert, no slack) -- even though experts are NOT laid
out in numeric order across the whole ~15.7GB span. This means one HTTP
Range request per expert (not per tensor, not the whole file) is both
correct and efficient: ~16.7MB/expert, ~1.07GB for `local_num_experts=64`.

Saves each expert's 6 tensors as local `.npy` files under
`wp_kv6_real_weights/layer1_expert<N>/` (gitignored -- real checkpoint
bytes, not ours to redistribute, same reasoning as
`06_kimi_k3_golden_validation/official_kimi_k3/`).

Usage:
  python fetch_real_mxfp4_expert_weights.py
  python fetch_real_mxfp4_expert_weights.py --num-experts 8  # smaller test run
"""

import argparse
import json
import pathlib
import struct
import sys
import time
import urllib.request

REPO = "moonshotai/Kimi-K3"
COMMIT = "a590ce090cb049c93a33dfe8c208ec652aa20503"
SHARD_FILE = "model-00002-of-000096.safetensors"
LAYER = 1
OUT_DIR = pathlib.Path(__file__).parent / "wp_kv6_real_weights"

_RESOLVE_URL = f"https://huggingface.co/{REPO}/resolve/{COMMIT}/{SHARD_FILE}"


def _http_range_get(url: str, start: int, end_inclusive: int) -> bytes:
  req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end_inclusive}"})
  with urllib.request.urlopen(req) as resp:
    return resp.read()


def _fetch_header() -> tuple[dict, int]:
  """Returns `(header, data_section_start)` -- a safetensors file's header
  stores each tensor's `data_offsets` RELATIVE TO THE START OF THE DATA
  SECTION (`8 + header_len`), NOT as absolute file byte positions. Missing
  this offset is a real, silent-looking bug this project hit the hard way:
  reading raw bytes at the header's `data_offsets` directly (without adding
  `data_section_start`) fetches real bytes from the WRONG location in the
  file -- plausible-looking but wrong data, not an error -- confirmed by a
  real weight_scale tensor coming back with values spread almost uniformly
  across the full uint8 range (a strong tell for "these are the wrong
  bytes", since a real per-block scale should cluster narrowly) until this
  offset was added.
  """
  header_len_bytes = _http_range_get(_RESOLVE_URL, 0, 7)
  header_len = struct.unpack("<Q", header_len_bytes)[0]
  header_bytes = _http_range_get(_RESOLVE_URL, 8, 8 + header_len - 1)
  return json.loads(header_bytes), 8 + header_len


_DTYPE_MAP = {"U8": "uint8"}


def main(num_experts: int) -> None:
  import numpy as np  # noqa: PLC0415 -- only needed here, not a project-wide dependency

  OUT_DIR.mkdir(parents=True, exist_ok=True)
  print(f"[fetch-mxfp4-weights] fetching header from {SHARD_FILE} ...")
  header, data_section_start = _fetch_header()

  prefix = f"language_model.model.layers.{LAYER}.block_sparse_moe.experts."
  experts: dict[int, dict[str, dict]] = {}
  for name, meta in header.items():
    if not name.startswith(prefix):
      continue
    rest = name[len(prefix):]
    idx_str, tensor_name = rest.split(".", 1)
    experts.setdefault(int(idx_str), {})[tensor_name] = meta

  missing = [i for i in range(num_experts) if i not in experts]
  if missing:
    raise KeyError(f"experts {missing} not found in {SHARD_FILE}'s header")

  for expert_idx in range(num_experts):
    tensors = experts[expert_idx]
    lo = min(meta["data_offsets"][0] for meta in tensors.values())
    hi = max(meta["data_offsets"][1] for meta in tensors.values())
    span = hi - lo
    expected_span = sum(meta["data_offsets"][1] - meta["data_offsets"][0] for meta in tensors.values())
    assert span == expected_span, (
        f"expert {expert_idx}'s 6 tensors are not contiguous (span={span}, "
        f"sum_of_tensor_sizes={expected_span}) -- the no-gap assumption this script relies on "
        "does not hold for this expert; do not trust a single Range request for it"
    )

    expert_dir = OUT_DIR / f"layer{LAYER}_expert{expert_idx}"
    if expert_dir.exists() and all((expert_dir / f"{n}.npy").exists() for n in tensors):
      print(f"[fetch-mxfp4-weights] expert {expert_idx}: already fetched, skipping")
      continue

    t0 = time.perf_counter()
    raw = _http_range_get(_RESOLVE_URL, data_section_start + lo, data_section_start + hi - 1)
    elapsed = time.perf_counter() - t0
    expert_dir.mkdir(parents=True, exist_ok=True)
    for tensor_name, meta in tensors.items():
      start, end = meta["data_offsets"]
      local_start, local_end = start - lo, end - lo
      dtype = _DTYPE_MAP[meta["dtype"]]
      arr = np.frombuffer(raw[local_start:local_end], dtype=dtype).reshape(meta["shape"])
      np.save(expert_dir / f"{tensor_name}.npy", arr)
    print(
        f"[fetch-mxfp4-weights] expert {expert_idx}: fetched {span / 1e6:.2f}MB in "
        f"{elapsed:.2f}s ({span / 1e6 / max(elapsed, 1e-6):.1f} MB/s)"
    )

  total_mb = num_experts * 17547264 / 1e6
  print(f"\n[fetch-mxfp4-weights] done: {num_experts} experts, ~{total_mb:.1f}MB total, saved under {OUT_DIR}")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--num-experts", type=int, default=64)
  args = parser.parse_args()
  main(args.num_experts)
  sys.exit(0)
