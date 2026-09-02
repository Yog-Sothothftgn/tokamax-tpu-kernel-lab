"""One-shot v6e experiment suite entry point.

The point of this file: TPU v6e capacity in this project's zone has been
genuinely scarce (real stockouts lasting hours to days, confirmed not a
Google-side incident -- see project notes). Every prior session spent part
of its scarce VM time hunting down which script/flag to run next. This
script runs the FULL battery in one shot, in a fixed order, so a VM session
collects as much data as possible even if some individual step fails --
one step's OOM or compile error never blocks the rest.

Runs, in order:
  1. Environment/device check (jax/tokamax/jaxlib versions, jax.devices(),
     TPU device kind, GCE zone/machine-type if running on a real GCE VM,
     this repo's own git commit, and the tokamax repo's git commit if
     `~/tokamax` exists).
  2. Local (CPU-only, no TPU/tokamax needed) routing edge-case unit tests --
     empty shards, extreme skew, capacity overflow, padding contamination,
     top_k=16, global/local boundary off-by-ones, first/last shard,
     batch/seq invariance, dtype promotion, multi-config sharded-sum-vs-
     reference. Runs first and fast specifically so an ordinary logic
     regression is caught here, before any of the expensive TPU-dependent
     steps below spend real VM time on it.
  2b. Memory budget estimate (also CPU-only, no TPU needed): a theoretical
      lower-bound byte-count table (weight/activation/padding memory per
      batch_size/seq_len shape) flagging which shapes are likely to OOM
      before any of them are actually attempted below.
  3. Official Kimi K3 source snapshot verification (hash-checked against
     the pinned commit).
  4. Sharded ragged_dot correctness (the real 16-of-896-then-filtered-to-
     shard routing, using tokamax.ragged_dot instead of the naive loop).
  5. Latency sweep under the REAL routing distribution (not the dense/
     uniform simplification the earlier benchmark used).
  6. Direct, same-device, element-wise comparison of xla/mosaic/
     mosaic_tpu_v2's bf16 outputs against EACH OTHER, across all 18 staged
     intermediates -- not just each vs. the golden PyTorch reference
     separately (closes a long-flagged gap: prior "shared bf16 precision
     effect" conclusions were only ever an inference from matching summary
     statistics across separate runs). Saves each implementation's full
     raw output bundle to <name>_outputs.npz plus a correctness.json under
     this step's own subdirectory of --output-dir.
  7. **`reduced_scale_golden_regression_battery`** -- one fine-grained step
     PER (bundle_set, variant, implementation) combination that's actually
     expected to run (12 combinations; combinations already known to hit
     Mosaic v1's 128-row tiling floor, e.g. mosaic v1 against the "small"
     or original "mosaic" bundles, are NOT attempted -- they'd always
     report an expected skip, and running them in the SAME subprocess as a
     combination that might have a real mismatch risks the classifier
     seeing "SKIPPED" and "FAIL" in the same combined output and
     misclassifying the whole step as UNSUPPORTED, masking the real
     failure). **NOT a full-dimension smoke test** -- every bundle here
     uses reduced scale (`num_experts=8`, `top_k=2`), not the real
     `num_experts=896`/`top_k=16`/`hidden=7168`/`latent=3584`/
     `intermediate=3072` -- a literal full-dimension run is infeasible on
     one chip anyway (confirmed OOM elsewhere in this project, ~59GB) and
     remains a separate, not-yet-done item. Each combination's structured
     per-stage results are written to its own correctness.json.
  8. WP4's real 4-stage profiling breakdown (router+projection / dispatch
     indexing / REAL tokamax.ragged_dot / combine) -- see
     `profile_four_stages_wp4`'s docstring for an important caveat: Stage B
     is timed eagerly, not on the same jitted-device basis as A/C/D, so the
     resulting irregular-share ratio is not yet a clean device-only
     SparseCore-decision number.

Each step runs as its own subprocess (not an in-process function call) so
that one step's crash, OOM, or compile error can never take down the
suite -- every subsequent step still runs. Status per step is one of:
  PASS           -- exit code 0
  FAIL           -- non-zero exit code, no more specific pattern matched
  UNSUPPORTED    -- output contains a NotImplementedError/SKIPPED marker
                    (e.g. Mosaic v1 below its tiling floor) -- with step 7
                    now split per-implementation, an UNSUPPORTED here is
                    genuinely unexpected (every guaranteed-skip combination
                    was excluded from the matrix), so it counts toward the
                    suite's overall failure the same as FAIL/OOM/COMPILE_ERROR.
  OOM            -- output mentions RESOURCE_EXHAUSTED / an out-of-memory pattern
  COMPILE_ERROR  -- output mentions a Mosaic/XLA compilation failure

The suite's own exit code is 1 if ANY step's status is FAIL, OOM,
COMPILE_ERROR, or UNSUPPORTED (0 only if every step is PASS) -- an earlier
version of this file always exited 0 regardless of step failures, which
would have made an automated caller think the whole suite succeeded even
when several steps failed.

Outputs, all under --output-dir:
  environment.json     -- versions, jax.devices(), git commits, GCE metadata
  summary.json          -- structured list of {name, status, returncode, elapsed_s, log_file, note}
  summary.csv            -- same, as CSV
  <step_name>.log         -- full stdout+stderr+command+timing for each step
  latency_sweep.csv / realistic_shard_latency.csv / wp4_profiling.csv --
    structured benchmark data from the corresponding steps (not just text logs)
  memory_budget.csv      -- from the memory budget step
  bf16_direct_compare_outputs/ -- <impl>_outputs.npz + correctness.json
  golden_regression/<combo>.json -- per-(bundle_set,variant,implementation) correctness.json

Usage:
  python run_v6e_experiment_suite.py --output-dir results/2026-09-01-v6e
"""

import argparse
import csv
import json
import pathlib
import platform
import subprocess
import sys
import time
import urllib.request

_HERE = pathlib.Path(__file__).parent
_GOLDEN_DIR = _HERE / "06_kimi_k3_golden_validation"
_RAGGED_DOT_DIR = _HERE / "05_ragged_dot_on_tpu"

_DEFAULT_TIMEOUT_S = 1800  # 30 minutes per step -- generous, but bounded so one hung step doesn't eat the whole session

# The 12 (bundle_set, variant, implementation) combinations actually expected
# to execute a kernel -- deliberately excludes combinations already known to
# hit Mosaic v1's 128-row tiling floor (mosaic v1 against "small" or the
# original "mosaic" bundle), per this file's module docstring.
_GOLDEN_REGRESSION_MATRIX: tuple[tuple[str, str, str], ...] = (
    ("small", "fp32", "xla"),
    ("small", "bf16", "xla"),
    ("mosaic", "fp32", "xla"),
    ("mosaic", "fp32", "mosaic_tpu_v2"),
    ("mosaic", "bf16", "xla"),
    ("mosaic", "bf16", "mosaic_tpu_v2"),
    ("mosaic_wide", "fp32", "xla"),
    ("mosaic_wide", "fp32", "mosaic"),
    ("mosaic_wide", "fp32", "mosaic_tpu_v2"),
    ("mosaic_wide", "bf16", "xla"),
    ("mosaic_wide", "bf16", "mosaic"),
    ("mosaic_wide", "bf16", "mosaic_tpu_v2"),
)

_FAILURE_STATUSES = frozenset({"FAIL", "OOM", "COMPILE_ERROR", "UNSUPPORTED"})


def _classify(returncode: int | None, stdout: str, stderr: str) -> str:
  combined = stdout + stderr
  if returncode == 0:
    return "PASS"
  if "NotImplementedError" in combined or "SKIPPED" in combined:
    return "UNSUPPORTED"
  if "RESOURCE_EXHAUSTED" in combined or "CompileTimeScopedVmemOom" in combined or "out of memory" in combined.lower():
    return "OOM"
  if "Mosaic failed to compile" in combined or "compilation failed" in combined.lower():
    return "COMPILE_ERROR"
  return "FAIL"


def _git_commit(repo_dir: pathlib.Path) -> str | None:
  try:
    result = subprocess.run(
        ["git", "-C", str(repo_dir), "rev-parse", "HEAD"],
        capture_output=True, text=True, timeout=10,
    )
    return result.stdout.strip() if result.returncode == 0 else None
  except Exception:  # noqa: BLE001 -- best-effort only, absence is recorded as None
    return None


def _gce_metadata(path: str) -> str | None:
  """Best-effort GCE instance metadata query -- only succeeds when actually
  running on a GCE VM (silently returns None everywhere else, e.g. locally)."""
  try:
    req = urllib.request.Request(
        f"http://metadata.google.internal/computeMetadata/v1/{path}",
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
      return resp.read().decode().strip()
  except Exception:  # noqa: BLE001
    return None


_JAX_TOKAMAX_PROBE = """
import json
info = {}
try:
    import jax
    info['jax_version'] = jax.__version__
    info['jax_devices'] = [str(d) for d in jax.devices()]
    try:
        info['tpu_device_kind'] = jax.devices()[0].device_kind
    except Exception:
        pass
except Exception as e:
    info['jax_error'] = str(e)
try:
    import jaxlib
    info['jaxlib_version'] = jaxlib.__version__
except Exception as e:
    info['jaxlib_error'] = str(e)
try:
    import tokamax
    info['tokamax_version'] = getattr(tokamax, '__version__', 'unknown')
except Exception as e:
    info['tokamax_error'] = str(e)
print(json.dumps(info))
"""


def _probe_jax_tokamax_env() -> dict:
  """Query jax/jaxlib/tokamax/TPU-device info in a SEPARATE, throwaway process.

  Must NEVER `import jax` in the orchestrator's own long-lived process:
  `jax.devices()` initializes the PJRT TPU client and claims the TPU for that
  process's entire lifetime (a v6e chip allows exactly one process at a time).
  Confirmed on real hardware (2026-09-02): the orchestrator process itself
  held the TPU after this check ran in-process, so every subsequent per-step
  subprocess failed with `ABORTED: The TPU is already in use by process with
  pid <orchestrator-pid>`. Probing in a subprocess that exits immediately
  releases the TPU before any real step needs it.
  """
  try:
    proc = subprocess.run(
        [sys.executable, "-c", _JAX_TOKAMAX_PROBE],
        capture_output=True, text=True, timeout=60,
    )
    line = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else ""
    return json.loads(line)
  except Exception as e:  # noqa: BLE001 -- recording the failure itself is the point
    return {"jax_error": f"failed to probe jax/tokamax in subprocess: {e}"}


def _env_info() -> dict:
  info: dict = {
      "python_version": platform.python_version(),
      "platform": platform.platform(),
  }
  info.update(_probe_jax_tokamax_env())

  info["kernel_lab_git_commit"] = _git_commit(_HERE)
  info["tokamax_git_commit"] = _git_commit(pathlib.Path.home() / "tokamax")
  info["gce_zone"] = _gce_metadata("instance/zone")
  info["gce_machine_type"] = _gce_metadata("instance/machine-type")
  return info


def _run_step(
    name: str,
    cwd: pathlib.Path,
    args: list[str],
    output_dir: pathlib.Path,
    timeout: int = _DEFAULT_TIMEOUT_S,
) -> dict:
  print(f"\n{'=' * 70}\n[suite] running: {name}\n  cwd={cwd}\n  cmd={' '.join(args)}\n{'=' * 70}")
  t0 = time.time()
  log_path = output_dir / f"{name}.log"
  try:
    proc = subprocess.run(args, cwd=str(cwd), capture_output=True, text=True, timeout=timeout)
    elapsed = time.time() - t0
    status = _classify(proc.returncode, proc.stdout, proc.stderr)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"CMD: {' '.join(args)}\nCWD: {cwd}\nRETURNCODE: {proc.returncode}\n"
        f"ELAPSED_S: {elapsed:.1f}\n\n--- STDOUT ---\n{proc.stdout}\n\n--- STDERR ---\n{proc.stderr}\n",
        encoding="utf-8",
    )
    print(f"[suite] {name}: {status} ({elapsed:.1f}s) -- see {log_path.name}")
    return {
        "name": name, "status": status, "returncode": proc.returncode,
        "elapsed_s": round(elapsed, 1), "log_file": log_path.name, "note": "",
    }
  except subprocess.TimeoutExpired as e:
    elapsed = time.time() - t0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        f"CMD: {' '.join(args)}\nCWD: {cwd}\nTIMED OUT after {timeout}s\n\n"
        f"--- STDOUT (partial) ---\n{e.stdout or ''}\n\n--- STDERR (partial) ---\n{e.stderr or ''}\n",
        encoding="utf-8",
    )
    print(f"[suite] {name}: FAIL (timed out after {timeout}s) -- see {log_path.name}")
    return {
        "name": name, "status": "FAIL", "returncode": None,
        "elapsed_s": round(elapsed, 1), "log_file": log_path.name,
        "note": f"timed out after {timeout}s",
    }


def main(output_dir: pathlib.Path) -> bool:
  """Returns True if every step passed (see module docstring for exactly
  what counts as a failure) -- the CLI below turns this into the process
  exit code."""
  output_dir.mkdir(parents=True, exist_ok=True)
  results: list[dict] = []

  # Step 1: environment/device check.
  env = _env_info()
  (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
  env_ok = bool(env.get("jax_devices")) and any("tpu" in d.lower() for d in env.get("jax_devices", []))
  print(f"[suite] environment: {json.dumps(env, indent=2)}")
  results.append({
      "name": "environment_check",
      "status": "PASS" if env_ok else "FAIL",
      "returncode": None, "elapsed_s": 0.0, "log_file": "environment.json",
      "note": "" if env_ok else "no TPU device found in jax.devices()",
  })

  # Step 2: local (CPU-only) routing edge-case unit tests -- fast, no TPU
  # needed, run first so an ordinary logic regression is caught before any
  # expensive TPU-dependent step below spends real VM time rediscovering it.
  results.append(_run_step(
      "local_sharded_routing_edge_cases", _RAGGED_DOT_DIR,
      [sys.executable, "test_sharded_routing_local.py"], output_dir,
  ))

  # Step 2b: memory budget estimate -- also CPU-only/no TPU needed, prints
  # the theoretical (lower-bound) memory table for the shapes about to be
  # exercised below, so an OOM-prone shape is flagged before the expensive
  # steps hit it for real.
  results.append(_run_step(
      "memory_budget_estimate", _RAGGED_DOT_DIR,
      [sys.executable, "memory_budget_estimate.py", "--output-dir", str(output_dir.resolve())],
      output_dir,
  ))

  # Step 3: official snapshot verification.
  results.append(_run_step(
      "official_config_validation", _GOLDEN_DIR,
      [sys.executable, "validate_official_config.py"], output_dir,
  ))
  results.append(_run_step(
      "official_snapshot_verification", _GOLDEN_DIR,
      [sys.executable, "verify_official_snapshot.py"], output_dir,
  ))

  # Step 4: sharded ragged_dot correctness.
  results.append(_run_step(
      "sharded_ragged_dot_correctness", _RAGGED_DOT_DIR,
      [sys.executable, "kimi_k3_latent_moe_ragged_dot.py", "--sharded-ragged-dot-correctness"], output_dir,
  ))

  # Step 5: realistic-distribution latency sweep. --output-dir is ABSOLUTE --
  # this subprocess's cwd is _RAGGED_DOT_DIR, not wherever this orchestrator
  # itself was invoked from.
  results.append(_run_step(
      "realistic_shard_latency_sweep", _RAGGED_DOT_DIR,
      [sys.executable, "kimi_k3_latent_moe_ragged_dot.py", "--realistic-shard-latency-sweep",
       "--output-dir", str(output_dir.resolve())],
      output_dir,
  ))

  # Step 6: direct same-device bf16 cross-implementation comparison.
  bf16_npz_dir = (output_dir / "bf16_direct_compare_outputs").resolve()
  results.append(_run_step(
      "bf16_cross_implementation_direct_compare", _GOLDEN_DIR,
      [sys.executable, "compare_bf16_implementations_direct.py", "--bundle-set", "mosaic_wide",
       "--output-dir", str(bf16_npz_dir)],
      output_dir,
  ))

  # Step 7: the reduced-scale golden-validation regression battery, one
  # step PER (bundle_set, variant, implementation) combination -- see this
  # file's module docstring for why this replaced a single combined
  # subprocess invocation (an expected skip and a real mismatch could land
  # in the same combined stdout/stderr and get misclassified as one
  # UNSUPPORTED status, masking the real failure).
  golden_regression_dir = (output_dir / "golden_regression").resolve()
  for bundle_set, variant, implementation in _GOLDEN_REGRESSION_MATRIX:
    step_name = f"reduced_scale_golden_regression__{bundle_set}__{variant}__{implementation}"
    json_path = golden_regression_dir / f"{bundle_set}__{variant}__{implementation}.json"
    results.append(_run_step(
        step_name, _GOLDEN_DIR,
        [sys.executable, "test_ragged_dot_against_pytorch_golden.py",
         "--bundle-set", bundle_set, "--variant", variant, "--implementation", implementation,
         "--json-output", str(json_path)],
        output_dir,
    ))

  # Step 8: WP4's real 4-stage profiling. Runs once per EXPLICIT implementation
  # -- never relying on --wp4-implementation's default, since tokamax's own
  # `implementation=None` resolution order tries Mosaic v1 before xla on TPU
  # (confirmed via tokamax/_src/ops/ragged_dot/api.py), which would silently
  # profile the slow, unautotuned backend and understate Stage B/D's relative
  # cost against a fast Stage C.
  for wp4_impl in ("xla", "mosaic_tpu_v2"):
    results.append(_run_step(
        f"wp4_four_stage_profiling__{wp4_impl}", _RAGGED_DOT_DIR,
        [sys.executable, "kimi_k3_latent_moe_ragged_dot.py", "--wp4-profile",
         "--wp4-implementation", wp4_impl,
         "--output-dir", str(output_dir.resolve())],
        output_dir,
    ))

  # Save JSON + CSV summaries.
  (output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
  fieldnames = ["name", "status", "returncode", "elapsed_s", "log_file", "note"]
  with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    for r in results:
      writer.writerow(r)

  print(f"\n{'=' * 70}\n[suite] SUMMARY\n{'=' * 70}")
  for r in results:
    print(f"  {r['status']:>12}  {r['name']:<55} ({r['elapsed_s']}s)")
  print(f"\n[suite] full results written to {output_dir}/summary.json and summary.csv")
  print(f"[suite] per-step logs (full stdout/stderr) in {output_dir}/*.log")

  has_failure = any(r["status"] in _FAILURE_STATUSES for r in results)
  if has_failure:
    print(
        "\n[suite] AT LEAST ONE STEP DID NOT PASS (FAIL/OOM/COMPILE_ERROR/UNSUPPORTED counted as "
        "failure) -- see summary.csv for which."
    )
  else:
    print("\n[suite] ALL STEPS PASSED.")
  return not has_failure


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-dir", type=pathlib.Path, required=True,
      help="directory to write environment.json, summary.json/.csv, and per-step .log/.csv/.json "
      "files to (created if it doesn't exist) -- e.g. results/2026-09-01-v6e",
  )
  args = parser.parse_args()
  all_passed = main(args.output_dir)
  sys.exit(0 if all_passed else 1)
