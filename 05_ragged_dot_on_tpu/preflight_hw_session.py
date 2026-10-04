"""Pre-flight checks for the hardware session (A, B, C). Touches the TPU only to
list devices; computes nothing. Prints PASS/WARN/FAIL lines and exits non-zero on FAIL.

  python3 preflight_hw_session.py
"""

import pathlib
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve().parent
EXPECT = {"jax": "0.11.0", "tokamax_commit": "a669f05d61d7c20f1397451abdd6afc8fb7937e3"}
fails = 0


def report(level, msg):
  global fails
  fails += level == "FAIL"
  print(f"[{level}] {msg}")


import jax  # noqa: E402

report("PASS" if jax.__version__ == EXPECT["jax"] else "FAIL", f"jax {jax.__version__} (expected {EXPECT['jax']})")
devs = jax.devices()
report("PASS" if devs[0].platform == "tpu" else "FAIL", f"devices {devs}")
try:
  from jax.experimental.pallas import tpu as pltpu
  sc = pltpu.get_tpu_info().sparse_core
  report("PASS" if sc is not None else "FAIL", f"sparse_core {sc}")
except Exception as e:  # noqa: BLE001
  report("FAIL", f"tpu info: {type(e).__name__}: {e}")
try:
  import tokamax
  root = pathlib.Path(tokamax.__file__).resolve().parent.parent
  commit = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
  report("PASS" if commit == EXPECT["tokamax_commit"] else "WARN", f"tokamax commit {commit[:12]} (earlier runs used {EXPECT['tokamax_commit'][:12]})")
  dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
  report("PASS" if not dirty else "WARN", "tokamax working tree clean" if not dirty else f"tokamax has local changes: {dirty[:200]}")
  from tokamax._src.ops.ragged_gather import api as rg_api  # noqa: F401
  report("PASS", "tokamax ragged_gather api imports")
except Exception as e:  # noqa: BLE001
  report("FAIL", f"tokamax: {type(e).__name__}: {e}")
for e in (0, 1, 2):
  d = HERE / "wp_kv6_real_weights" / f"layer1_expert{e}"
  need = [d / f"{n}.weight_{k}.npy" for n in ("w1", "w3") for k in ("packed", "scale")]
  missing = [p.name for p in need if not p.exists()]
  report("PASS" if not missing else "FAIL", f"expert {e} weights" + (f" missing {missing}" if missing else " present"))
st = subprocess.run(["git", "-C", str(HERE.parent), "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
report("PASS" if not st else "WARN", "kernel-lab clean" if not st else "kernel-lab has uncommitted/untracked files (saved as a patch by each experiment)")
sys.exit(1 if fails else 0)
