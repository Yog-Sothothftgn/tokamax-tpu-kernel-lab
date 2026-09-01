"""One-shot v6e experiment suite entry point.

The point of this file: TPU v6e capacity in this project's zone has been
genuinely scarce (real stockouts lasting hours to days, confirmed not a
Google-side incident -- see project notes). Every prior session spent part
of its scarce VM time hunting down which script/flag to run next. This
script runs the FULL battery in one shot, in a fixed order, so a VM session
collects as much data as possible even if some individual step fails --
one step's OOM or compile error never blocks the rest.

Runs, in order:
  1. Environment/device check (jax/tokamax versions, jax.devices()).
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
     mosaic_tpu_v2's bf16 outputs against EACH OTHER -- not just each vs.
     the golden PyTorch reference separately (closes a long-flagged gap:
     prior "shared bf16 precision effect" conclusions were only ever an
     inference from matching summary statistics across separate runs).
  7. The full existing golden-validation battery: every bundle set (small/
     mosaic/mosaic_wide) x every dtype (fp32/bf16) x every implementation
     (xla/mosaic/mosaic_tpu_v2) -- the broadest correctness sweep this
     project currently has, referred to here as the "full-dimension smoke
     test" since a literal full-896-expert/unsharded run is infeasible on
     one chip (confirmed OOM elsewhere in this project, ~59GB).
  8. WP4's real 4-stage profiling breakdown (router+projection / dispatch
     indexing / REAL tokamax.ragged_dot / combine).

Each step runs as its own subprocess (not an in-process function call) so
that one step's crash, OOM, or compile error can never take down the
suite -- every subsequent step still runs. Status per step is one of:
  PASS           -- exit code 0
  FAIL           -- non-zero exit code, no more specific pattern matched
  UNSUPPORTED    -- output contains a NotImplementedError/SKIPPED marker
                    (e.g. Mosaic v1 below its tiling floor)
  OOM            -- output mentions RESOURCE_EXHAUSTED / an out-of-memory pattern
  COMPILE_ERROR  -- output mentions a Mosaic/XLA compilation failure

Outputs, all under --output-dir:
  environment.json     -- jax/tokamax versions, jax.devices() (also step 1's own record)
  summary.json          -- structured list of {name, status, returncode, elapsed_s, log_file, note}
  summary.csv            -- same, as CSV
  <step_name>.log         -- full stdout+stderr+command+timing for each step

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

_HERE = pathlib.Path(__file__).parent
_GOLDEN_DIR = _HERE / "06_kimi_k3_golden_validation"
_RAGGED_DOT_DIR = _HERE / "05_ragged_dot_on_tpu"

_DEFAULT_TIMEOUT_S = 1800  # 30 minutes per step -- generous, but bounded so one hung step doesn't eat the whole session


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


def _env_info() -> dict:
  info: dict = {
      "python_version": platform.python_version(),
      "platform": platform.platform(),
  }
  try:
    import jax
    info["jax_version"] = jax.__version__
    info["jax_devices"] = [str(d) for d in jax.devices()]
  except Exception as e:  # noqa: BLE001 -- recording the failure itself is the point
    info["jax_error"] = str(e)
  try:
    import tokamax
    info["tokamax_version"] = getattr(tokamax, "__version__", "unknown")
  except Exception as e:  # noqa: BLE001
    info["tokamax_error"] = str(e)
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


def main(output_dir: pathlib.Path) -> None:
  output_dir.mkdir(parents=True, exist_ok=True)
  results: list[dict] = []

  # Step 1: environment/device check.
  env = _env_info()
  (output_dir / "environment.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
  env_ok = bool(env.get("jax_devices")) and any("Tpu" in d for d in env.get("jax_devices", []))
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
      [sys.executable, "memory_budget_estimate.py"], output_dir,
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

  # Step 5: realistic-distribution latency sweep.
  results.append(_run_step(
      "realistic_shard_latency_sweep", _RAGGED_DOT_DIR,
      [sys.executable, "kimi_k3_latent_moe_ragged_dot.py", "--realistic-shard-latency-sweep"], output_dir,
  ))

  # Step 6: direct same-device bf16 cross-implementation comparison.
  results.append(_run_step(
      "bf16_cross_implementation_direct_compare", _GOLDEN_DIR,
      [sys.executable, "compare_bf16_implementations_direct.py", "--bundle-set", "mosaic_wide"], output_dir,
  ))

  # Step 7: the full existing golden-validation battery (every bundle set x
  # dtype x implementation) -- the broadest correctness sweep currently
  # available; see this file's module docstring for why this stands in for
  # a literal full-896-expert "full-dimension" smoke test.
  results.append(_run_step(
      "full_golden_validation_battery", _GOLDEN_DIR,
      [sys.executable, "test_ragged_dot_against_pytorch_golden.py",
       "--bundle-set", "both", "--variant", "both", "--implementation", "all"],
      output_dir, timeout=2400,
  ))

  # Step 8: WP4's real 4-stage profiling.
  results.append(_run_step(
      "wp4_four_stage_profiling", _RAGGED_DOT_DIR,
      [sys.executable, "kimi_k3_latent_moe_ragged_dot.py", "--wp4-profile"], output_dir,
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
    print(f"  {r['status']:>12}  {r['name']:<40} ({r['elapsed_s']}s)")
  print(f"\n[suite] full results written to {output_dir}/summary.json and summary.csv")
  print(f"[suite] per-step logs (full stdout/stderr) in {output_dir}/*.log")


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--output-dir", type=pathlib.Path, required=True,
      help="directory to write environment.json, summary.json/.csv, and per-step .log files to "
      "(created if it doesn't exist) -- e.g. results/2026-09-01-v6e",
  )
  args = parser.parse_args()
  main(args.output_dir)
