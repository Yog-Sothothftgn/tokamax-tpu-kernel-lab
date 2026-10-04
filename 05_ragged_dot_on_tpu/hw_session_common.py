"""Shared helpers for the next v6e hardware session (experiments A, B, C).

* `record_session_environment`: saves the code version (commit, uncommitted file
  list and the full uncommitted diff as a patch file), JAX / jaxlib / libtpu /
  Tokamax versions, device info and, when given, a provenance record of the
  real-checkpoint files that were used (path, size, sha256). Commit dates are
  recorded only as metadata; nothing here infers when an experiment was run
  from them.
* `rotated_timing`: the project's established timing protocol -- compile and
  warm up outside the timed region, rotate the candidate order every round,
  explicit jit arguments (no closure-captured data), report the pipelined
  convention (consecutive dispatches, one block at the end, bounded in flight)
  and the per-call convention (block after every call) separately, with the raw
  per-round rows saved.
* `error_report`: error metrics that say what they are (see its docstring).
"""

import collections
import csv
import datetime
import hashlib
import json
import pathlib
import platform
import subprocess
import time

import jax
import jax.numpy as jnp

HERE = pathlib.Path(__file__).resolve().parent
REPO_ROOT = HERE.parent


def _git(*args: str) -> str:
  try:
    r = subprocess.run(["git", "-C", str(REPO_ROOT), *args], capture_output=True, text=True, timeout=60)
    return r.stdout.strip() if r.returncode == 0 else f"<git failed: {r.stderr.strip()}>"
  except Exception as e:  # noqa: BLE001
    return f"<git unavailable: {e}>"


def _pkg_version(*names: str) -> str:
  from importlib import metadata
  for n in names:
    try:
      return f"{n}=={metadata.version(n)}"
    except metadata.PackageNotFoundError:
      continue
  return "not installed"


def sha256_file(path: pathlib.Path, chunk: int = 1 << 20) -> str:
  h = hashlib.sha256()
  with open(path, "rb") as f:
    while block := f.read(chunk):
      h.update(block)
  return h.hexdigest()


def record_session_environment(results_dir: pathlib.Path, tag: str, weight_files: list | None = None) -> dict:
  results_dir.mkdir(parents=True, exist_ok=True)
  porcelain = _git("status", "--porcelain")
  diff = _git("diff", "HEAD")
  patch_path = results_dir / f"uncommitted_{tag}.patch"
  patch_path.write_text(diff + "\n", encoding="utf-8")
  tokamax_info = {}
  try:
    import tokamax  # noqa: PLC0415
    root = pathlib.Path(tokamax.__file__).resolve().parent.parent
    tokamax_info = {
        "import_path": str(pathlib.Path(tokamax.__file__).resolve()),
        "commit": subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "dirty": subprocess.run(["git", "-C", str(root), "status", "--porcelain"], capture_output=True, text=True).stdout.strip(),
        "distribution": _pkg_version("tokamax"),
    }
  except Exception as e:  # noqa: BLE001
    tokamax_info = {"error": f"{type(e).__name__}: {e}"}
  dev = jax.devices()[0]
  env = {
      "tag": tag,
      "recorded_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
      "python": platform.python_version(),
      "platform": platform.platform(),
      "jax": jax.__version__,
      "jaxlib": _pkg_version("jaxlib"),
      "libtpu": _pkg_version("libtpu", "libtpu-nightly"),
      "devices": [str(d) for d in jax.devices()],
      "device_kind": dev.device_kind,
      "kernel_lab": {
          "commit": _git("rev-parse", "HEAD"),
          "commit_date_metadata_only": _git("log", "-1", "--format=%cI"),
          "uncommitted_files": porcelain,
          "uncommitted_diff_patch": str(patch_path.name),
      },
      "tokamax": tokamax_info,
  }
  try:
    from jax.experimental.pallas import tpu as pltpu  # noqa: PLC0415
    env["tpu_info"] = repr(pltpu.get_tpu_info())
  except Exception as e:  # noqa: BLE001
    env["tpu_info"] = f"unavailable: {type(e).__name__}: {e}"
  if weight_files:
    env["weight_files"] = [
        {"path": str(p), "bytes": pathlib.Path(p).stat().st_size, "sha256": sha256_file(pathlib.Path(p))}
        for p in weight_files
    ]
  (results_dir / f"env_{tag}.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
  print(f"[env:{tag}] " + json.dumps({k: env[k] for k in ("jax", "jaxlib", "libtpu", "device_kind")}) +
        f" commit={env['kernel_lab']['commit'][:8]} uncommitted_files={'yes' if porcelain else 'no'}")
  return env


def _median(v):
  s = sorted(v)
  n = len(s)
  return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def rotated_timing(fns: dict, args: tuple, label: str, raw_rows: list, rounds: int = 10, repeats: int = 20,
                   max_inflight: int = 4, point: dict | None = None) -> dict:
  """fns: name -> callable taking *args; the FIRST entry is the reference the
  speedups are relative to (speedup = reference latency / candidate latency,
  below 1 = slower than the reference). Returns {name: summary}."""
  names = list(fns)
  jitted = {n: jax.jit(f) for n, f in fns.items()}
  for f in jitted.values():  # compile + warm up outside the timed region
    jax.block_until_ready(f(*args))
    jax.block_until_ready(f(*args))

  def pipelined(f):
    pend = collections.deque()
    t0 = time.perf_counter()
    for _ in range(repeats):
      pend.append(f(*args))
      if len(pend) > max_inflight:
        jax.block_until_ready(pend.popleft())
    while pend:
      jax.block_until_ready(pend.popleft())
    return (time.perf_counter() - t0) / repeats * 1000

  def blocking(f):
    t0 = time.perf_counter()
    for _ in range(repeats):
      jax.block_until_ready(f(*args))
    return (time.perf_counter() - t0) / repeats * 1000

  pipe = {n: [] for n in names}
  blk = {n: [] for n in names}
  for r in range(rounds):
    rot = r % len(names)
    for pos, n in enumerate(names[rot:] + names[:rot]):
      p, b = pipelined(jitted[n]), blocking(jitted[n])
      pipe[n].append(p)
      blk[n].append(b)
      raw_rows.append({**(point or {}), "label": label, "round": r, "order_pos": pos, "candidate": n,
                       "pipelined_ms": f"{p:.5f}", "per_call_ms": f"{b:.5f}"})
  ref = names[0]
  out = {}
  for n in names:
    out[n] = dict(
        pipelined_median_ms=_median(pipe[n]), pipelined_min_ms=min(pipe[n]), pipelined_max_ms=max(pipe[n]),
        per_call_median_ms=_median(blk[n]), per_call_min_ms=min(blk[n]), per_call_max_ms=max(blk[n]),
        speedup_vs_ref_pipelined=_median(pipe[ref]) / _median(pipe[n]),
        speedup_vs_ref_per_call=_median(blk[ref]) / _median(blk[n]),
    )
  print(f"\n[{label}] {rounds} rounds x {repeats} calls, rotated; reference = {ref}; speedup = reference / candidate")
  for n in names:
    s = out[n]
    print(f"  {n:34s} pipelined {s['pipelined_median_ms']:.4f}ms (min {s['pipelined_min_ms']:.4f} "
          f"max {s['pipelined_max_ms']:.4f}) x{s['speedup_vs_ref_pipelined']:.3f} | per-call "
          f"{s['per_call_median_ms']:.4f}ms (min {s['per_call_min_ms']:.4f} max {s['per_call_max_ms']:.4f}) "
          f"x{s['speedup_vs_ref_per_call']:.3f}")
  return out


def write_csv(path: pathlib.Path, rows: list[dict]) -> None:
  if not rows:
    return
  path.parent.mkdir(parents=True, exist_ok=True)
  with open(path, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)
  print(f"[csv] wrote {path}")


def error_report(out: jax.Array, ref: jax.Array) -> dict:
  """Error metrics, each labelled with its exact definition (so a bare
  'error percent' is never printed):
    max_abs_err        max |out - ref|
    rmse               sqrt(mean((out - ref)^2))
    ref_std            std of the reference output
    nrmse_vs_ref_std   rmse / ref_std
    relative_max_diff  max_abs_err / ref_std  -- THE PROJECT'S EXISTING METRIC. Its
                       denominator is the reference output's standard deviation, so
                       it is NOT an element-wise relative error.
    max_elem_rel_err   max |out - ref| / max(|ref|, 1e-3 * ref_std)  (element-wise,
                       floored so near-zero references do not dominate)
    nonfinite_out      number of NaN/Inf in out
    n_diff             number of elements whose bf16 values differ at all
  """
  o = out.astype(jnp.float32)
  r = ref.astype(jnp.float32)
  d = jnp.abs(o - r)
  ref_std = float(jnp.std(r)) + 1e-12
  rmse = float(jnp.sqrt(jnp.mean(d * d)))
  max_abs = float(jnp.max(d))
  floor = 1e-3 * ref_std
  return {
      "max_abs_err": max_abs,
      "rmse": rmse,
      "ref_std": ref_std,
      "nrmse_vs_ref_std": rmse / ref_std,
      "relative_max_diff": max_abs / ref_std,
      "max_elem_rel_err": float(jnp.max(d / jnp.maximum(jnp.abs(r), floor))),
      "nonfinite_out": int(jnp.sum(~jnp.isfinite(o))),
      "n_diff": int(jnp.sum(o != r)),
      "n_elements": int(o.size),
  }
