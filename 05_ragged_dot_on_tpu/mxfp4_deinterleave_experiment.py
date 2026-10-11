"""Single-variable experiment D: is the outside even / odd split of the activations the
main cost of the online MXFP4 prototype?

Background (hardware session, experiment C, pipelined ms, gate matrix, expert 0, bm = M, bn = 256):
                       M = 128    M = 2048
  XLA, pre-dequantized   0.0426     0.0691
  Pallas, pre-dequant.   0.0415     0.0756
  Pallas, online MXFP4   0.0811     0.6346
Arithmetic (mxfp4_cost_model.py): the grid is (1, 12) in both cases, so the dequantization (11,010,048
elements) is done once whatever M is; it is therefore at most 81 us = 12.8 percent of the M = 2048 call,
and at least 554 us of that call is M-dependent. The excess of the online kernel over the Pallas
pre-dequantized matmul grows 14.1x for 16x the rows (39.6 -> 559.0 us). Remaining M-dependent suspects:
  (H1) the stride-2 slicing of x done OUTSIDE the kernel but inside the timed call, and
  (H2) the dot structure inside the kernel (two K/2 transposed-rhs dots).
This experiment tests H1 only.

HYPOTHESIS H1: the outside deinterleave of x (x[:, 0::2], x[:, 1::2]) accounts for most of the
M = 2048 time of the online kernel.

THE ONE VARIABLE: whether that split is inside the timed call.
  online_original   exactly the kernel call of experiment C, split inside the call
  online_presplit   the SAME pallas_call with the same tile; x_even / x_odd are computed once
                    OUTSIDE the timed region (materialised arrays) and passed as jit arguments
Nothing else differs (same kernel, weights, scales, tile, M, protocol). Both outputs must be
bit-identical (same arithmetic); this is checked before any timing. Calibration candidates, not part of
the comparison: `deinterleave_only` (the two slices alone) and `dispatch_floor` (a tiny op, the cost of
one separate dispatch), plus the XLA pre-dequantized matmul for context.

PRE-REGISTERED DECISION (written before the run; evaluated at M = 2048 on the pipelined medians, with
T0 = online_original and T1 = online_presplit measured in the SAME process, not the old numbers;
f = (T0 - T1) / T0; run-to-run noise of this protocol is 1-3 percent):
  f >= 0.50   SUPPORTS H1: moving the split out of the call removes at least half of the time. The
              remaining fraction is then at most the dot structure plus dequantization.
  f <= 0.10   REFUTES H1 as a main cause: the split costs at most a tenth of the call; look at H2 next.
  otherwise   PARTIAL: the split explains a fraction f and something else explains the rest.
  Consistency check (reported, not decisive): (T0 - T1) should be close to
  (deinterleave_only - dispatch_floor); a large mismatch means the split overlaps with other work.
M = 128 is run for completeness; no verdict is drawn from it (x is only 0.9 MB there).
If H1 is refuted or partial, the next single variable (not implemented here) is the contraction form:
keep the split outside the call and replace the two transposed-rhs dots by plain (K/2, bn) right-hand sides
(pre-transposed packed layout prepared as weight-format preparation), everything else unchanged.

Run on the TPU (needs wp_kv6_real_weights/layer1_expert0; fetch with
`python fetch_real_mxfp4_expert_weights.py --num-experts 1`):
  python3 -u mxfp4_deinterleave_experiment.py 2>&1 | tee hw_session_results/6_D_deinterleave.log
CPU logic check (correctness of the two entry points and of the decision rule; no timing):
  python mxfp4_deinterleave_experiment.py cpu_smoke
"""

import json
import pathlib
import sys
import traceback

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hw_session_common as hc  # noqa: E402
import mxfp4_online_dequant_matmul_pallas as om  # noqa: E402

RESULTS_DIR = HERE / "hw_session_results"
WEIGHTS = HERE / "wp_kv6_real_weights" / "layer1_expert0"
SUPPORT_MIN_FRACTION = 0.50
REFUTE_MAX_FRACTION = 0.10
TOLERANCE = 0.05


def decide(t0: float, t1: float) -> dict:
  f = (t0 - t1) / t0
  if f >= SUPPORT_MIN_FRACTION:
    verdict = "SUPPORTS H1 (the outside split is at least half of the call)"
  elif f <= REFUTE_MAX_FRACTION:
    verdict = "REFUTES H1 as a main cause (the split is at most a tenth of the call); test H2 next"
  else:
    verdict = "PARTIAL (the split explains a fraction f; something else explains the rest)"
  return {"T0_original_ms": t0, "T1_presplit_ms": t1, "removed_fraction_f": f, "verdict": verdict,
          "thresholds": {"supports_if_f_at_least": SUPPORT_MIN_FRACTION, "refutes_if_f_at_most": REFUTE_MAX_FRACTION}}


def make_x(m: int):
  return (jax.random.normal(jax.random.key(hash(m) % (2**31)), (m, om.LATENT_SIZE)) * 0.02).astype(jnp.bfloat16)


def run(cpu_smoke: bool = False):
  RESULTS_DIR.mkdir(exist_ok=True)
  hc.record_session_environment(RESULTS_DIR, "D", [WEIGHTS / "w1.weight_packed.npy", WEIGHTS / "w1.weight_scale.npy"])
  packed = jnp.asarray(np.load(WEIGHTS / "w1.weight_packed.npy"))
  scale = jnp.asarray(np.load(WEIGHTS / "w1.weight_scale.npy"))
  assert int(scale.min()) >= 1
  _, scale_p = om.prepare_weights(packed, scale)
  w_deq_nk = om.reference_dequant_bf16(packed, scale)
  w_kn = jnp.asarray(w_deq_nk.T)
  jax.block_until_ready((scale_p, w_kn))

  results, raw = {}, []
  for m in ((16,) if cpu_smoke else (2048, 128)):
    bm, bn = m, 256
    x = make_x(m)
    x_even, x_odd = jax.jit(om.deinterleave)(x)  # prepared OUTSIDE the timed region
    jax.block_until_ready((x_even, x_odd))
    args = (x, x_even, x_odd, packed, scale_p, w_kn)

    fns = {
        "xla_matmul_predequantized(context)": lambda x_, xe, xo, p_, s_, wk: jnp.dot(
            x_, wk, preferred_element_type=jnp.float32).astype(x_.dtype),
        "online_original": lambda x_, xe, xo, p_, s_, wk: om.online_dequant_matmul(x_, p_, s_, bm=bm, bn=bn),
        "online_presplit": lambda x_, xe, xo, p_, s_, wk: om.online_dequant_matmul_presplit(xe, xo, p_, s_, bm=bm, bn=bn),
        "deinterleave_only(calibration)": lambda x_, xe, xo, p_, s_, wk: om.deinterleave(x_),
        "dispatch_floor(calibration)": lambda x_, xe, xo, p_, s_, wk: x_[:8, :128] + jnp.bfloat16(0),
    }

    # Correctness first: the two entry points must be bit-identical, and close to XLA.
    out0 = jax.jit(fns["online_original"])(*args)
    out1 = jax.jit(fns["online_presplit"])(*args)
    ref = jax.jit(fns["xla_matmul_predequantized(context)"])(*args)
    jax.block_until_ready((out0, out1, ref))
    identical = bool(jnp.array_equal(out0.view(jnp.uint16), out1.view(jnp.uint16)))
    rep = hc.error_report(out1, ref)
    ok = identical and rep["relative_max_diff"] < TOLERANCE and rep["nonfinite_out"] == 0
    print(f"[D correctness] M={m}: original == presplit bit for bit: {identical}; presplit vs XLA "
          f"relative_max_diff(=max_abs/ref_std)={rep['relative_max_diff']:.4f} nonfinite={rep['nonfinite_out']} -> {'OK' if ok else 'FAIL'}")
    results[str(m)] = {"identical_outputs": identical, "presplit_vs_xla": rep}
    if not ok or cpu_smoke:
      if not ok:
        print(f"[D] M={m}: correctness failed; no timing for this M")
      continue

    summ = hc.rotated_timing(fns, args, f"D M={m}", raw, point={"experiment": "D", "M": m, "bm": bm, "bn": bn})
    p = {k: v["pipelined_median_ms"] for k, v in summ.items()}
    b = {k: v["per_call_median_ms"] for k, v in summ.items()}
    entry = results[str(m)]
    entry["pipelined_median_ms"], entry["per_call_median_ms"] = p, b
    entry["decision_pipelined"] = decide(p["online_original"], p["online_presplit"])
    entry["decision_per_call"] = decide(b["online_original"], b["online_presplit"])
    d_net = p["deinterleave_only(calibration)"] - p["dispatch_floor(calibration)"]
    entry["consistency"] = {"T0_minus_T1_ms": p["online_original"] - p["online_presplit"],
                            "deinterleave_only_minus_dispatch_floor_ms": d_net}
    tag = "PRIMARY VERDICT" if m == 2048 else "informational (no verdict drawn)"
    print(f"\n[D result, M={m}, {tag}] T0(original)={p['online_original']:.4f} ms, T1(presplit)={p['online_presplit']:.4f} ms, "
          f"f={entry['decision_pipelined']['removed_fraction_f']:.3f} -> {entry['decision_pipelined']['verdict']}\n"
          f"    per-call convention: f={entry['decision_per_call']['removed_fraction_f']:.3f}\n"
          f"    consistency: T0-T1={entry['consistency']['T0_minus_T1_ms']:.4f} ms vs deinterleave_only-dispatch_floor={d_net:.4f} ms")
  (RESULTS_DIR / "D_result.json").write_text(json.dumps(results, indent=2, default=str))
  hc.write_csv(RESULTS_DIR / "D_timing_raw.csv", raw)


def cpu_logic_check():
  # Decision rule on synthetic numbers (boundaries and the three outcomes).
  cases = [(0.6346, 0.10, "SUPPORTS"), (0.6346, 0.60, "REFUTES"), (0.6346, 0.40, "PARTIAL"),
           (1.0, 0.5, "SUPPORTS"), (1.0, 0.9, "REFUTES")]
  for t0, t1, expect in cases:
    got = decide(t0, t1)["verdict"].split()[0]
    assert got == expect, (t0, t1, got, expect)
  print("[cpu_smoke] decision rule: OK")


if __name__ == "__main__":
  smoke = len(sys.argv) > 1 and sys.argv[1] == "cpu_smoke"
  print(f"devices: {jax.devices()}  jax: {jax.__version__}  cpu_smoke={smoke}")
  try:
    if smoke:
      cpu_logic_check()
    run(cpu_smoke=smoke)
  except Exception:  # noqa: BLE001
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "D_fatal_error.txt").write_text(traceback.format_exc())
    raise
