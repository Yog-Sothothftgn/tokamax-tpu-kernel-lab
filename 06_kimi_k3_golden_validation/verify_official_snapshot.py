"""Independent re-verification step, requested explicitly by the user
before trusting WP-KV3's comparison: re-download config.json,
configuration_kimi_k3.py, and modeling_kimi_linear.py directly from the
PINNED commit's `/resolve/<commit>/...` URLs (not `/raw/`, which can 302 to
a Git LFS pointer for large files -- see validate_official_config.py's own
history this session with the safetensors index), and diff the downloaded
bytes against whatever is currently sitting in `official_kimi_k3/` --
byte-for-byte, not just hash-vs-hash.

This exists as a SEPARATE step from validate_official_config.py (WP-KV1)
for two reasons:
  1. `configuration_kimi_k3.py` was fetched during WP-KV2 via an ad-hoc curl
     command, never hashed or locked anywhere -- this closes that gap by
     covering all three files WP-KV2/KV3 actually import from.
  2. It re-fetches FRESH bytes from the network right now and compares them
     against the local files on disk, rather than trusting that the local
     files still match what validate_official_config.py recorded earlier --
     catching local edits/corruption/drift, not just re-stating a past hash.

Usage:
  python verify_official_snapshot.py
  python verify_official_snapshot.py --commit <other-sha>
"""

import argparse
import hashlib
import pathlib
import sys
import urllib.request

from validate_official_config import DEFAULT_COMMIT, REPO

_HERE = pathlib.Path(__file__).parent
OFFICIAL_DIR = _HERE / "official_kimi_k3"

FILES = ("config.json", "configuration_kimi_k3.py", "modeling_kimi_linear.py")


def _sha256(data: bytes) -> str:
  return hashlib.sha256(data).hexdigest()


def _fetch(commit: str, filename: str) -> bytes:
  # /resolve/<commit>/<path> follows LFS pointers and returns the real file
  # bytes for both small text files and large ones -- unlike /raw/, which
  # can return a Git LFS pointer stub instead of the actual content for
  # LFS-tracked files (confirmed the hard way earlier this session with the
  # safetensors index.json).
  url = f"https://huggingface.co/{REPO}/resolve/{commit}/{filename}"
  with urllib.request.urlopen(url) as resp:
    return resp.read()


def main(commit: str) -> bool:
  print(f"[verify-official-snapshot] pinned commit: {commit}")
  all_ok = True

  for filename in FILES:
    local_path = OFFICIAL_DIR / filename
    if not local_path.exists():
      print(f"  {filename}: FAIL -- no local copy at {local_path}")
      all_ok = False
      continue

    local_bytes = local_path.read_bytes()
    local_hash = _sha256(local_bytes)

    print(f"[verify-official-snapshot] re-downloading {filename} ...")
    remote_bytes = _fetch(commit, filename)
    remote_hash = _sha256(remote_bytes)

    bytes_match = remote_bytes == local_bytes
    print(f"  {filename}:")
    print(f"    remote sha256={remote_hash}")
    print(f"    local  sha256={local_hash}")
    print(f"    byte-for-byte match: {'OK' if bytes_match else 'FAIL'}")
    all_ok = all_ok and bytes_match

  print(f"\n[verify-official-snapshot] {'ALL FILES VERIFIED' if all_ok else 'MISMATCH DETECTED'}")
  return all_ok


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--commit", default=DEFAULT_COMMIT, help="HF commit SHA to verify against")
  args = parser.parse_args()

  ok = main(args.commit)
  sys.exit(0 if ok else 1)
