"""Three-way gather comparison (2026-10-03), per the user-approved plan:

  1. `xla_ref`             -- XLA bf16 gather: safe index, gather, mask.
  2. `ours_unpack_outside` -- our latest: column-half packing built live
                              every call, cross-core split, SparseCore
                              gathers packed int32 rows, unpack/concat/
                              mask outside the kernel
                              (`sparsecore_gather_colhalf_bf16_unpack_outside`).
  3. `tokamax_v2`          -- Tokamax's existing SparseCore ragged gather,
                              `implementation="mosaic_tpu_v2"` (explicit,
                              never auto-selected), wrapped with a safe
                              index, `start=0, end=len(indices)`, and OUR
                              valid_mask applied outside.

Gather ONLY: no router, sort, combine or forward integration. The three
candidates share one interface, `candidate(x_bf16, padded_token_idx,
valid_mask)`, and one reference
    expected = where(valid_mask[:, None], x[maximum(idx, 0)], 0).

Questions: is Tokamax's ready-made SparseCore implementation, at OUR real
shape, faster than the hand-written version, and if so, in which segment?
Nothing is assumed in advance, and no conclusion about SparseCore as a
whole is drawn from one round.

Interface facts this script relies on (read from upstream Tokamax source;
the INSTALLED copy is hash-checked against the version that was read and
any mismatch is reported loudly):
  - `api.ragged_gather(x, indices, start, end, *, implementation=...)`;
    `start`/`end` are shape-(1,) int arrays. With a single implementation
    named, NotImplementedError is re-raised (no fallback). The kernel
    itself returns `x[indices]` if the chip has no SparseCore, so SparseCore
    execution is CONFIRMED from the device trace (kernel name
    `sc_ragged_gather_v2`), not assumed.
  - The v2 kernel processes whole blocks covering [start, end) and does
    not write other blocks; the reference ignores start/end. Hence
    start=0, end=len(indices) (NOT num_valid: valid slots are not a
    contiguous prefix of the output in general).
  - Upstream tests skip unless device_kind == "TPU7x"; there is no
    upstream coverage on v6e, so correctness here is verified by this
    script (exact value AND raw-bit equality; upstream tolerates 1e-2).

Rules enforced: no environment upgrade, no Tokamax modification, no
fallback to XLA when a candidate fails (full traceback saved, candidate
excluded from the ranking, the others continue). A candidate that fails
correctness at ANY stage is never ranked.

Outputs (under `sparsecore_tokamax_results/`): env.json, copies of the
installed Tokamax ragged_gather files, correctness.json, raw per-round
timing CSV, a summary table CSV; the device trace path is printed.

Timing scope (everything a candidate needs per call is INSIDE the timed
function; routing/sorting are excluded because all three use the same
pre-generated indices):
  xla_ref: safe-index handling, gather, mask.
  ours:    live packing, SparseCore gather, unpack, crop, mask.
  tokamax: its own dynamic layout handling, padding, SparseCore gather,
           crop, then the mask. (Tokamax receives raw bf16 and does its
           own packing; the pre-packed variant of ours is NOT compared
           against it.)
Inputs/indices/mask are explicit jit arguments (no closure constants).
10 rounds x 20 calls, candidate order rotates each round, pipelined and
per-call both reported; neither is pure device kernel time. speedup =
XLA latency / candidate latency (>1 means faster than XLA).

To run (real v6e VM, from the kernel-lab checkout, venv active):
  JAX_TRACEBACK_FILTERING=off python3 -u sparsecore_gather_tokamax_comparison.py \\
    2>&1 | tee sparsecore_tokamax_comparison.log
"""

import csv
import hashlib
import importlib
import importlib.metadata
import json
import pathlib
import platform
import shutil
import subprocess
import sys
import time
import traceback

import jax
import jax.numpy as jnp

_HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from sparsecore_gather_prototype import LATENT_SIZE, real_dispatch_indices  # noqa: E402
from sparsecore_gather_group_b_comparison import NUM_TOKENS, LOCAL_NUM_EXPERTS  # noqa: E402
from sparsecore_gather_bf16_colhalf import (  # noqa: E402
    _named,
    sparsecore_gather_colhalf_bf16_unpack_outside,
)
import sparsecore_gather_device_trace as dt  # noqa: E402

RESULTS_DIR = _HERE / "sparsecore_tokamax_results"
TRACE_DIR_TOKAMAX = "/tmp/sparsecore_tokamax_trace"
NUM_ROUNDS = 10
NUM_REPEATS = 20
CAND_NAMES = ("xla_ref", "ours_unpack_outside", "tokamax_v2")

# sha256 of the upstream Tokamax files as READ by the author of this script
# (origin/main 3a0a3a34a5ccb8a6ac71f343d9cc44649d73db68, 2026-10-02).
UPSTREAM_READ_SHA256 = {
    "api.py": "bf8afc25dbd04b6d9fc0810111d66e19d346227efddb8d0eb0e04bb305a8cb37",
    "base.py": "75085e3277a9b082f531aedc40abea464c83ea8d23ecbb44e6842274a4a27a82",
    "pallas_mosaic_v2_tpu.py": "53fd11adf8e42b267ebf7c5215a42131c56277ee2494fc7efcf75fc333a9ed84",
    "pallas_mosaic_v2_tpu_kernel.py": "eca653fba422be9fd09a6ed81b8cacd87df5ff23d9e079748962352aded5f199",
    "pallas_mosaic_v2_tpu_test.py": "efd374881f8a69897221995024f61a093604b39e5425872bfb3781d2d48d1e1e",
    "test_base.py": "11e18768ffe8ae22f79beef718acbd71ed5dbd7f982c0bd937d0c10e1d3d87f5",
}


# ----------------------------------------------------------------------------
# Candidates
# ----------------------------------------------------------------------------

def reference_gather(x, idx, mask):
  """The unified reference (also the XLA candidate)."""
  safe_idx = jnp.maximum(idx, 0)
  return jnp.where(mask[:, None], x[safe_idx], jnp.zeros((), x.dtype))


def ours_unpack_outside(x, idx, mask):
  return sparsecore_gather_colhalf_bf16_unpack_outside(x, idx, mask, window_size=8, core_split=True)


def tokamax_v2(x, idx, mask):
  # Imported lazily so an import problem is reported as a candidate failure
  # (with the exact location), not as a crash of the whole script.
  from tokamax._src.ops.ragged_gather import api as rg_api

  safe_idx = jnp.maximum(idx, 0)
  start = jnp.zeros((1,), jnp.int32)
  end = jnp.full((1,), idx.shape[0], jnp.int32)  # the WHOLE output, not num_valid
  gathered = rg_api.ragged_gather(x, safe_idx, start, end, implementation="mosaic_tpu_v2")
  return jnp.where(mask[:, None], gathered, jnp.zeros((), gathered.dtype))


CANDIDATES = {
    "xla_ref": reference_gather,
    "ours_unpack_outside": ours_unpack_outside,
    "tokamax_v2": tokamax_v2,
}


# ----------------------------------------------------------------------------
# Environment record
# ----------------------------------------------------------------------------

def _git(repo, *args):
  try:
    out = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, timeout=30)
    return out.stdout.strip() if out.returncode == 0 else f"<git failed: {out.stderr.strip()}>"
  except Exception as e:  # noqa: BLE001
    return f"<git unavailable: {e}>"


def _pkg_version(*names):
  for n in names:
    try:
      return f"{n}=={importlib.metadata.version(n)}"
    except importlib.metadata.PackageNotFoundError:
      continue
  return f"<none of {names} installed as a distribution>"


def record_environment() -> dict:
  from jax.experimental.pallas import tpu as pltpu

  RESULTS_DIR.mkdir(exist_ok=True)
  env: dict = {"time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
  env["python"] = sys.version
  env["platform"] = platform.platform()
  env["jax"] = jax.__version__
  import jaxlib
  env["jaxlib"] = jaxlib.__version__
  env["libtpu"] = _pkg_version("libtpu", "libtpu-nightly")
  env["jax_distribution"] = _pkg_version("jax")
  dev = jax.devices()[0]
  env["device"] = {"repr": repr(dev), "device_kind": getattr(dev, "device_kind", "?"), "platform": dev.platform}
  try:
    tpu_info = pltpu.get_tpu_info()
    env["tpu_info"] = repr(tpu_info)
    env["tpu_generation"] = getattr(tpu_info, "generation", None)
    env["sparse_core"] = repr(tpu_info.sparse_core)
  except Exception as e:  # noqa: BLE001
    env["tpu_info"] = f"<error: {e}>"

  kl_root = _git(_HERE, "rev-parse", "--show-toplevel")
  env["kernel_lab"] = {
      "root": kl_root,
      "commit": _git(_HERE, "rev-parse", "HEAD"),
      "dirty_files": _git(_HERE, "status", "--porcelain"),
  }

  files = {}
  dest = RESULTS_DIR / "tokamax_ragged_gather_installed_copy"
  try:
    import tokamax
    tk_file = pathlib.Path(tokamax.__file__).resolve()
    tk_root = tk_file.parent.parent
    env["tokamax"] = {
        "import_path": str(tk_file),
        "repo_root_guess": str(tk_root),
        "commit": _git(tk_root, "rev-parse", "HEAD"),
        "dirty_files": _git(tk_root, "status", "--porcelain"),
        "distribution": _pkg_version("tokamax"),
    }

    # The Tokamax ragged_gather modules that are ACTUALLY imported, their
    # hashes vs the upstream version that was read, and verbatim copies.
    from tokamax._src.ops.ragged_gather import api as rg_api
    from tokamax._src.ops.ragged_gather import pallas_mosaic_v2_tpu, pallas_mosaic_v2_tpu_kernel
    mods = {
        "api.py": rg_api,
        "pallas_mosaic_v2_tpu.py": pallas_mosaic_v2_tpu,
        "pallas_mosaic_v2_tpu_kernel.py": pallas_mosaic_v2_tpu_kernel,
    }
    rg_dir = pathlib.Path(rg_api.__file__).resolve().parent
    dest = RESULTS_DIR / "tokamax_ragged_gather_installed_copy"
    dest.mkdir(exist_ok=True)
    files = {}
    for fname, expected_sha in UPSTREAM_READ_SHA256.items():
      f = rg_dir / fname
      if not f.exists():
        files[fname] = {"path": str(f), "exists": False}
        continue
      data = f.read_bytes()
      sha = hashlib.sha256(data).hexdigest()
      shutil.copy2(f, dest / fname)
      files[fname] = {
          "path": str(f), "exists": True, "lines": data.count(b"\n"), "sha256": sha,
          "matches_upstream_version_that_was_read": sha == expected_sha,
      }
    env["tokamax_ragged_gather_files"] = files
    env["imported_module_paths"] = {k: str(pathlib.Path(m.__file__).resolve()) for k, m in mods.items()}
    env["ragged_gather_implementations_registered"] = sorted(rg_api.IMPLEMENTATIONS.keys())
    env["ragged_gather_default_implementations"] = list(rg_api._DEFAULT_IMPLEMENTATIONS)
    try:
      env["v2_supported_on_this_device"] = bool(pallas_mosaic_v2_tpu.PallasV2TpuRaggedGather().supported_on(dev))
    except Exception as e:  # noqa: BLE001
      env["v2_supported_on_this_device"] = f"<error: {e}>"
    try:
      env["v2_col_size_for_real_hidden"] = pallas_mosaic_v2_tpu_kernel.calculate_col_size(LATENT_SIZE, 2)
    except Exception as e:  # noqa: BLE001
      env["v2_col_size_for_real_hidden"] = f"<error: {e}>"

  except Exception:  # noqa: BLE001 -- record the exact failing location instead of crashing
    env["tokamax_environment_error"] = traceback.format_exc()
    print("!!! could not inspect the installed Tokamax -- exact location recorded in env.json:\n"
          + env["tokamax_environment_error"])
  (RESULTS_DIR / "env.json").write_text(json.dumps(env, indent=2, default=str), encoding="utf-8")
  print("=" * 78 + "\nENVIRONMENT\n" + "=" * 78)
  print(json.dumps(env, indent=2, default=str))
  mism = [k for k, v in files.items() if v.get("exists") and not v["matches_upstream_version_that_was_read"]]
  missing = [k for k, v in files.items() if not v.get("exists")]
  if mism or missing:
    print(
        f"\n!!! INSTALLED Tokamax differs from the upstream version the interface notes were written against: "
        f"hash mismatch={mism}, missing={missing}. The start/end semantics and the wrapper below were written "
        "for the other version -- review the copied files in "
        f"{dest} before trusting any Tokamax result.\n"
    )
  elif "tokamax_environment_error" in env:
    print("\nTokamax could not be inspected (see the error above) -- NOTHING was verified about the installed files.")
  else:
    print("\nInstalled Tokamax ragged_gather files match the upstream version that was read (all hashes equal).")
  return env


# ----------------------------------------------------------------------------
# Correctness
# ----------------------------------------------------------------------------

def _bits_equal(a, b):
  return bool(jnp.array_equal(a.view(jnp.uint16), b.view(jnp.uint16)))


def make_small_case():
  """512 x 256 bf16 table (aligned for every candidate: half width 128),
  256 output indices incl. duplicates, a reversed run, first/last rows and
  invalid (negative, masked) positions. Values are exact in bf16 and
  include positive, negative, zero and -0.0."""
  rows, cols, n_out = 512, 256, 256
  vals = ((jnp.arange(rows * cols) % 509) - 254).astype(jnp.float32) / 16.0  # k/16, |k|<=254: exact in bf16
  x = vals.reshape(rows, cols).astype(jnp.bfloat16)
  x = x.at[3, 5].set(jnp.asarray(-0.0, jnp.bfloat16))
  key = jax.random.key(7)
  idx = jax.random.randint(key, (n_out,), 0, rows, jnp.int32)
  idx = idx.at[0].set(0).at[1].set(rows - 1)                 # first and last rows
  idx = idx.at[2:5].set(7)                                    # duplicates
  idx = idx.at[10:26].set(jnp.arange(25, 9, -1, dtype=jnp.int32))  # strictly descending run
  idx = idx.at[100:120].set(-1).at[200:204].set(-5)          # invalid slots (negative)
  mask = (idx >= 0)
  return x, idx, mask


def run_correctness(cands: dict):
  """Returns (surviving candidates, report dict). A candidate must pass the
  small case (value + raw bits), then the real case (value + raw bits)."""
  report = {}
  alive = dict(cands)

  def attempt(stage, name, fn, args, expected):
    try:
      out = jax.jit(fn)(*args)
      jax.block_until_ready(out)
    except Exception as e:  # noqa: BLE001 -- full error is the deliverable
      tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
      print(f"\n[{stage}][{name}] FAILED to compile/run -- full error below:\n{tb}")
      report.setdefault(name, {})[stage] = {"status": "error", "traceback": tb}
      alive.pop(name, None)
      return
    ok_val = bool(jnp.array_equal(out, expected))
    ok_bits = _bits_equal(out, expected)
    print(f"[{stage}][{name}] value match={ok_val}, raw-bit match={ok_bits}, shape={out.shape}, dtype={out.dtype}")
    report.setdefault(name, {})[stage] = {"status": "ok" if (ok_val and ok_bits) else "MISMATCH",
                                          "value_match": ok_val, "bit_match": ok_bits}
    if not (ok_val and ok_bits):
      alive.pop(name, None)

  print("\n" + "=" * 78 + "\nCORRECTNESS 1/2: small aligned case (512x256 bf16, 256 indices)\n" + "=" * 78)
  xs, idxs, masks = make_small_case()
  expected_small = jax.jit(reference_gather)(xs, idxs, masks)
  jax.block_until_ready(expected_small)
  print(f"small case: {int(jnp.sum(masks))} valid of {idxs.shape[0]} slots; "
        f"has -0.0 in table: {bool(jnp.any(xs.view(jnp.uint16) == 0x8000))}")
  for name, fn in cands.items():
    attempt("small", name, fn, (xs, idxs, masks), expected_small)

  print("\n" + "=" * 78 + "\nCORRECTNESS 2/2: real scale (production routing indices)\n" + "=" * 78)
  x_bf16, padded_token_idx, valid_mask, production_sorted_tokens = real_dispatch_indices(
      num_tokens=NUM_TOKENS, local_num_experts=LOCAL_NUM_EXPERTS, seed=0
  )
  print(f"num_tokens={NUM_TOKENS} latent_size={LATENT_SIZE} local_num_experts={LOCAL_NUM_EXPERTS} seed=0 "
        f"-> m_padded={int(padded_token_idx.shape[0])} num_valid={int(jnp.sum(valid_mask))}")
  expected_real = jax.jit(reference_gather)(x_bf16, padded_token_idx, valid_mask)
  jax.block_until_ready(expected_real)
  ref_vs_prod_val = bool(jnp.array_equal(expected_real, production_sorted_tokens))
  ref_vs_prod_bits = _bits_equal(expected_real, production_sorted_tokens)
  print(f"[reference vs production sorted_tokens] value match={ref_vs_prod_val}, raw-bit match={ref_vs_prod_bits}")
  report["_reference_vs_production_sorted_tokens"] = {"value_match": ref_vs_prod_val, "bit_match": ref_vs_prod_bits}
  if not (ref_vs_prod_val and ref_vs_prod_bits):
    print("!!! the unified reference does NOT match the production function's sorted_tokens -- stopping: "
          "every comparison below would be against a wrong ground truth.")
    (RESULTS_DIR / "correctness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return {}, report, None
  for name, fn in list(alive.items()):
    attempt("real", name, fn, (x_bf16, padded_token_idx, valid_mask), expected_real)

  (RESULTS_DIR / "correctness.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
  return alive, report, (x_bf16, padded_token_idx, valid_mask)


# ----------------------------------------------------------------------------
# Timing
# ----------------------------------------------------------------------------

def _time_pipelined(f_jit, args, n):
  t0 = time.perf_counter()
  for _ in range(n):
    out = f_jit(*args)
  jax.block_until_ready(out)
  return (time.perf_counter() - t0) / n * 1000


def _time_blocking(f_jit, args, n):
  t0 = time.perf_counter()
  for _ in range(n):
    out = f_jit(*args)
    jax.block_until_ready(out)
  return (time.perf_counter() - t0) / n * 1000


def _median(v):
  s = sorted(v)
  n = len(s)
  return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def run_timing(alive: dict, args: tuple) -> None:
  names = list(alive.keys())
  jitted = {n: jax.jit(fn) for n, fn in alive.items()}
  for n, f in jitted.items():            # compile + warm up OUTSIDE the timed region
    jax.block_until_ready(f(*args))
    jax.block_until_ready(f(*args))

  rows = []
  for rnd in range(NUM_ROUNDS):
    rot = rnd % len(names)
    order = names[rot:] + names[:rot]
    for pos, n in enumerate(order):
      pipe = _time_pipelined(jitted[n], args, NUM_REPEATS)
      blk = _time_blocking(jitted[n], args, NUM_REPEATS)
      rows.append({"round": rnd, "order_position": pos, "candidate": n,
                   "pipelined_ms": pipe, "per_call_ms": blk})
    print(f"[round {rnd}] order={order}")

  RESULTS_DIR.mkdir(exist_ok=True)
  with open(RESULTS_DIR / "timing_raw_per_round.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["round", "order_position", "candidate", "pipelined_ms", "per_call_ms"])
    w.writeheader()
    w.writerows(rows)

  summary = {}
  for n in names:
    p = [r["pipelined_ms"] for r in rows if r["candidate"] == n]
    b = [r["per_call_ms"] for r in rows if r["candidate"] == n]
    summary[n] = {"pipelined_median": _median(p), "pipelined_min": min(p), "pipelined_max": max(p),
                  "per_call_median": _median(b), "per_call_min": min(b), "per_call_max": max(b)}
  xla = summary.get("xla_ref")
  print(f"\n[timing: {NUM_ROUNDS} rounds x {NUM_REPEATS} calls; speedup = XLA latency / candidate latency; "
        "neither convention is pure device kernel time]")
  header = ["candidate", "pipelined_median_ms", "pipelined_min", "pipelined_max", "pipelined_speedup",
            "per_call_median_ms", "per_call_min", "per_call_max", "per_call_speedup"]
  table = []
  for n in names:
    s = summary[n]
    ps = xla["pipelined_median"] / s["pipelined_median"] if xla else float("nan")
    bs = xla["per_call_median"] / s["per_call_median"] if xla else float("nan")
    table.append([n, f"{s['pipelined_median']:.4f}", f"{s['pipelined_min']:.4f}", f"{s['pipelined_max']:.4f}", f"{ps:.3f}",
                  f"{s['per_call_median']:.4f}", f"{s['per_call_min']:.4f}", f"{s['per_call_max']:.4f}", f"{bs:.3f}"])
    print(f"  {n:22s} pipelined median={s['pipelined_median']:.4f}ms (min {s['pipelined_min']:.4f}, max {s['pipelined_max']:.4f}) "
          f"speedup={ps:.3f}x | per-call median={s['per_call_median']:.4f}ms (min {s['per_call_min']:.4f}, "
          f"max {s['per_call_max']:.4f}) speedup={bs:.3f}x")
  with open(RESULTS_DIR / "three_way_result_table.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)
    w.writerows(table)
  print(f"\nraw per-round timings -> {RESULTS_DIR / 'timing_raw_per_round.csv'}; "
        f"table -> {RESULTS_DIR / 'three_way_result_table.csv'}")


# ----------------------------------------------------------------------------
# Trace
# ----------------------------------------------------------------------------

def run_trace(alive: dict, args: tuple) -> None:
  runs = {n: (jax.jit(_named(n, fn)), args[0]) for n, fn in alive.items()}
  print("\n" + "=" * 78 + "\nDEVICE TRACE (distinct module names)\n" + "=" * 78)
  dt.capture_trace_runs(runs, args[1], args[2], trace_dir=TRACE_DIR_TOKAMAX)
  print(f"trace directory: {TRACE_DIR_TOKAMAX}")
  dt.analyze_trace(module_names=tuple(runs.keys()), trace_dir=TRACE_DIR_TOKAMAX, top_n_ops=14)
  print(
      "\nREADING NOTES (from the plan): do not add overlapping tracks together; call-done is time the TensorCore "
      "waited for the SparseCore, not compute; copy-done is not the full data-movement time; anything the trace "
      "cannot show is a hypothesis to verify, not a fact. For the Tokamax module, confirm a SparseCore program "
      "name containing 'sc_ragged_gather_v2' appears above -- if no SparseCore kernel ran, the call silently "
      "fell back to XLA and its timing must not be reported as a SparseCore result."
  )


def main() -> None:
  from jax.experimental.pallas import tpu as pltpu
  print(f"devices: {jax.devices()}")
  if pltpu.get_tpu_info().sparse_core is None:
    print("No SparseCore on this TPU -- cannot run this comparison.")
    raise SystemExit(1)
  record_environment()
  alive, _report, real_args = run_correctness(CANDIDATES)
  dropped = [n for n in CANDIDATES if n not in alive]
  print(f"\ncandidates that passed ALL correctness stages: {list(alive)}; excluded from ranking: {dropped}")
  if "xla_ref" not in alive or len(alive) < 2 or real_args is None:
    print("Need the XLA reference plus at least one other correct candidate to compare -- stopping before timing.")
    return
  print("\n" + "=" * 78 + "\nTIMING\n" + "=" * 78)
  run_timing(alive, real_args)
  run_trace(alive, real_args)


if __name__ == "__main__":
  main()
