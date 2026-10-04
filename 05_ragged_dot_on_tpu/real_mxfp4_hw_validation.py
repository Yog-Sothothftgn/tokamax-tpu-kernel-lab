"""Hardware validation with REAL Kimi K3 MXFP4 expert weights (v6e session).

Two independent experiments, each run in its own process:

  B  (python real_mxfp4_hw_validation.py B)
     The hand-written fused gate + up + SiTU-GLU Pallas kernel
     (own_gateup_situ_pallas_kernel.fused_gateup_situ) fed with REAL checkpoint
     weights that were PRE-DEQUANTIZED to BF16 outside the kernel and outside the
     timed region. This closes the gap left by commit 46a706d, whose real-weight
     check only ran in interpret mode.  Label for every B result:
         "Real checkpoint weights, pre-dequantized to BF16."
     It is NOT online dequantization. No down projection, routing, combine or full
     MoE; activations are fixed-seed SYNTHETIC data, not real model activations.
     M = the number of input rows given to ONE expert, not a model batch size or
     sequence length.

  C  (python real_mxfp4_hw_validation.py C)
     The online-dequantization prototype (mxfp4_online_dequant_matmul_pallas.py):
     one expert, ONE matrix (gate / w1), packed MXFP4 + scales read by the kernel,
     dequantized inside it on every call. REPORTED AS A CORRECTNESS PROTOTYPE: no
     performance expectation is stated or assumed; stage 4 timing is recorded as a
     measurement only. Separate stages that must not be merged: (0) the zero-padded
     scale's valid range, (1) in-kernel dequantization bit-exact on real weights,
     checked on the complete matrix in both the (N, K) and (K, N) layouts, (2) matmul
     correct against the pre-dequantized baselines under the explicitly documented
     precision conventions (PRECISION_CONVENTIONS below), (3) compiled module reads
     compressed inputs, (4) timing vs two pre-dequantized baselines. Gate/up fusion
     with SiTU is NOT part of C.

Experts: 0 (all M, timing), 1 and 2 (correctness only). The files must exist
under wp_kv6_real_weights/layer1_expert<N>/ ; on a fresh VM run first
  python fetch_real_mxfp4_expert_weights.py --num-experts 3
(public Hugging Face repo, ~17.5 MB per expert). Layer = 1 for every expert.

Correctness metric definitions are in hw_session_common.error_report. In
particular `relative_max_diff` (the project's existing metric) divides by the
STANDARD DEVIATION of the reference output, so it is not an element-wise
relative error. The pass criterion is the project's existing one
(relative_max_diff < 0.05, no NaN/Inf) and is not loosened here. The primary
reference is the EAGER plain-JAX reference (the project's established
convention); the jit-compiled XLA output is reported separately because the
two can differ.

Results go to hw_session_results/ (env JSON + uncommitted patch, raw per-round
timings, summary CSVs, correctness JSON).
"""

import functools
import json
import pathlib
import sys

import jax
import jax.numpy as jnp
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import hw_session_common as hc  # noqa: E402
import own_gateup_situ_pallas_kernel as gk  # noqa: E402
import mxfp4_online_dequant_matmul_pallas as om  # noqa: E402
from own_single_matmul_pallas_kernel import pallas_matmul  # noqa: E402

RESULTS_DIR = HERE / "hw_session_results"
WEIGHTS_DIR = HERE / "wp_kv6_real_weights"
LAYER = 1
LABEL_B = "Real checkpoint weights, pre-dequantized to BF16"
PRECISION_CONVENTIONS = {
    "xla_matmul_predequantized": "bf16 activations x pre-dequantized bf16 weights, jnp.dot with "
                                 "preferred_element_type=float32 (XLA picks the accumulation order), ONE bf16 rounding of "
                                 "the float32 result",
    "pallas_matmul_predequantized": "same bf16 inputs; the existing hand-written matmul with a single full-K tile "
                                    "(bk=3584): one dot, float32 accumulation in a VMEM scratch, ONE bf16 rounding on write",
    "pallas_online_dequant_mxfp4": "weights dequantized inside the kernel to bf16 values that stage 1 proves bit-identical "
                                   "to the pre-dequantized weights; bf16 activations split into even / odd K; two dots with "
                                   "float32 accumulation summed in float32; ONE bf16 rounding. Its float32 summation order "
                                   "differs from the other two, so results may differ by rounding",
    "f32_reference": "float32 activations x the same bf16 weights upcast to float32, float32 result (never rounded to "
                     "bf16); used only to measure errors",
}
TOLERANCE = 0.05
# Tile configs already confirmed to compile on v6e (gateup_situ_token_sweep.csv): bm = M, except M=4096 -> 1024.
KNOWN_GOOD_BM = {1: 1, 128: 128, 2048: 2048, 4096: 1024}
BN_FUSED = 256


def expert_files(expert: int, names=("w1", "w3")) -> list[pathlib.Path]:
  d = WEIGHTS_DIR / f"layer{LAYER}_expert{expert}"
  return [d / f"{n}.{kind}.npy" for n in names for kind in ("weight_packed", "weight_scale")]


def make_x(m: int) -> jax.Array:
  key = jax.random.key(hash(m) % (2**31))  # same convention as own_gateup_situ_pallas_kernel.check
  return (jax.random.normal(key, (m, gk.LATENT_SIZE)) * 0.02).astype(jnp.bfloat16)


def weight_provenance(expert: int) -> dict:
  out = {"layer": LAYER, "expert": expert}
  for n in ("w1", "w3"):
    packed = np.load(WEIGHTS_DIR / f"layer{LAYER}_expert{expert}" / f"{n}.weight_packed.npy")
    scale = np.load(WEIGHTS_DIR / f"layer{LAYER}_expert{expert}" / f"{n}.weight_scale.npy")
    out[n] = {"packed_shape": list(packed.shape), "scale_shape": list(scale.shape),
              "scale_min": int(scale.min()), "scale_max": int(scale.max())}
  return out


# ----------------------------------------------------------------------------
# Experiment B
# ----------------------------------------------------------------------------

def _pallas_fn(bm: int):
  return functools.partial(gk.fused_gateup_situ, bm=bm, bk=gk.LATENT_SIZE, bn=BN_FUSED)


def _xla_fn():
  return functools.partial(gk._reference_gateup_situ, beta=gk.SITU_BETA, linear_beta=gk.SITU_LINEAR_BETA,
                           round_dtype=jnp.bfloat16)


def _compile_with_bm_fallback(m: int, x, w_gate, w_up):
  """Known-good bm first, then descending divisors; records which worked."""
  first = KNOWN_GOOD_BM.get(m, min(m, 2048))
  cands = [first] + [c for c in (2048, 1024, 512, 256, 128, 64, 32, 16, 8, 4, 2, 1) if c != first and c <= m and m % c == 0]
  last = None
  for bm in cands:
    try:
      f = _pallas_fn(bm)
      out = jax.jit(f)(x, w_gate, w_up)
      jax.block_until_ready(out)
      return bm, f, out, None
    except Exception as e:  # noqa: BLE001 -- a failure at some tile size is data, kept in full
      last = f"bm={bm}: {type(e).__name__}: {str(e)[:600]}"
      print(f"[compile] m={m} {last}")
  return None, None, None, last


def run_B(cpu_smoke: bool = False) -> None:
  RESULTS_DIR.mkdir(exist_ok=True)
  experts = (0, 1, 2)
  hc.record_session_environment(RESULTS_DIR, "B", [p for e in experts for p in expert_files(e)])
  print(f"=== Experiment B: {LABEL_B} (fused gate+up+SiTU, hand-written Pallas) ===")
  provenance = {e: weight_provenance(e) for e in experts}
  for e in experts:
    print(f"[weights] {provenance[e]}")
    assert provenance[e]["w1"]["scale_min"] >= 1 and provenance[e]["w3"]["scale_min"] >= 1
  correctness = []
  gap_rows = []
  loaded = {}
  m_plan = {0: (128, 2048, 1, 4096), 1: (128, 2048), 2: (128, 2048)}
  if cpu_smoke:
    m_plan = {0: (8,)}
  for e in experts:
    if cpu_smoke and e != 0:
      continue
    w_gate, w_up = gk.load_real_expert_gate_up(expert_idx=e, layer=LAYER)  # dequantized once, outside any timing
    jax.block_until_ready((w_gate, w_up))
    loaded[e] = (w_gate, w_up)
    print(f"[weights] expert {e}: dequantized bf16 gate/up {w_gate.shape} std={float(jnp.std(w_gate.astype(jnp.float32))):.4f} "
          f"nonfinite={int(jnp.sum(~jnp.isfinite(w_gate.astype(jnp.float32))))}")
    for m in m_plan[e]:
      if e == 0 and m in (1, 4096) and not all(
          r["status"] == "OK" for r in correctness if r["expert"] == 0 and r["M"] in (128, 2048)):
        print(f"[B] expert 0 M={m} skipped: M=128 and M=2048 did not both pass first (plan order)")
        continue
      x = make_x(m)
      bm, f, out, err = _compile_with_bm_fallback(m, x, w_gate, w_up)
      row = {"label": LABEL_B, "layer": LAYER, "expert": e, "M": m, "bm": bm, "bk": gk.LATENT_SIZE, "bn": BN_FUSED}
      if f is None:
        row.update({"status": "COMPILE_FAILED", "error": err})
        print(f"[B] expert {e} M={m}: COMPILE FAILED {err}")
        correctness.append(row)
        continue
      eager_ref = gk._reference_gateup_situ(x, w_gate, w_up, gk.SITU_BETA, gk.SITU_LINEAR_BETA, jnp.bfloat16)
      xla_jit = jax.jit(_xla_fn())(x, w_gate, w_up)
      rep_eager = hc.error_report(out, eager_ref)
      rep_jit = hc.error_report(out, xla_jit)
      rep_ref_noise = hc.error_report(xla_jit, eager_ref)
      # Gap between the two REFERENCES themselves (jit-compiled XLA vs the eager plain-JAX path), on the
      # final output and on each intermediate matmul, so it is visible where the gap arises.
      mm = jax.jit(lambda a, w: (a @ w).astype(jnp.bfloat16))
      gap_gate = hc.error_report(mm(x, w_gate), (x @ w_gate).astype(jnp.bfloat16))
      gap_up = hc.error_report(mm(x, w_up), (x @ w_up).astype(jnp.bfloat16))
      for stage, rep in (("final_output", rep_ref_noise), ("gate_matmul_bf16", gap_gate), ("up_matmul_bf16", gap_up)):
        gap_rows.append({"expert": e, "M": m, "stage": stage + " (jit XLA vs eager reference)",
                         "max_abs_err": f"{rep['max_abs_err']:.4e}", "rmse": f"{rep['rmse']:.4e}",
                         "nrmse_vs_ref_std": f"{rep['nrmse_vs_ref_std']:.4f}",
                         "relative_max_diff_(max_abs/ref_std)": f"{rep['relative_max_diff']:.4f}",
                         "n_diff": rep["n_diff"], "n_elements": rep["n_elements"]})
        print(f"[B reference gap] expert {e} M={m} {stage}: max_abs={rep['max_abs_err']:.4e} rmse={rep['rmse']:.4e} "
              f"relative_max_diff={rep['relative_max_diff']:.4f} n_diff={rep['n_diff']}/{rep['n_elements']}")
      ok = rep_eager["relative_max_diff"] < TOLERANCE and rep_eager["nonfinite_out"] == 0
      row.update({"status": "OK" if ok else "FAIL", "vs_eager_reference": rep_eager, "vs_jit_xla": rep_jit,
                  "jit_xla_vs_eager_reference": rep_ref_noise,
                  "reference_gap_gate_matmul": gap_gate, "reference_gap_up_matmul": gap_up})
      correctness.append(row)
      print(f"[B] expert {e} M={M_str(m)} bm={bm}: {row['status']} | vs eager ref: max_abs={rep_eager['max_abs_err']:.4e} "
            f"rmse={rep_eager['rmse']:.4e} nrmse(/ref_std)={rep_eager['nrmse_vs_ref_std']:.4f} "
            f"relative_max_diff(=max_abs/ref_std)={rep_eager['relative_max_diff']:.4f} "
            f"max_elem_rel={rep_eager['max_elem_rel_err']:.3f} nonfinite={rep_eager['nonfinite_out']} "
            f"n_diff={rep_eager['n_diff']}/{rep_eager['n_elements']} | jit-XLA vs eager "
            f"relative_max_diff={rep_ref_noise['relative_max_diff']:.4f}")
  (RESULTS_DIR / "B_correctness.json").write_text(json.dumps({"provenance": provenance, "rows": correctness}, indent=2))
  hc.write_csv(RESULTS_DIR / "B_reference_gap.csv", gap_rows)

  if cpu_smoke:
    print("cpu smoke done (no timing)")
    return
  # Timing: expert 0 only, only for M that passed correctness.
  w_gate, w_up = loaded[0]
  raw_rows, summary_rows = [], []
  for row in [r for r in correctness if r["expert"] == 0]:
    if row["status"] != "OK":
      print(f"[B timing] M={row['M']} skipped: correctness status {row['status']}")
      continue
    m, bm = row["M"], row["bm"]
    x = make_x(m)
    fns = {"xla_unfused_reference": _xla_fn(), "pallas_fused": _pallas_fn(bm)}
    summ = hc.rotated_timing(fns, (x, w_gate, w_up), f"B timing M={m} ({LABEL_B})", raw_rows,
                             point={"experiment": "B", "expert": 0, "M": m, "bm": bm})
    s_x, s_p = summ["xla_unfused_reference"], summ["pallas_fused"]
    summary_rows.append({"label": LABEL_B, "expert": 0, "M": m, "tile": f"bm={bm},bk={gk.LATENT_SIZE},bn={BN_FUSED}",
                         "correctness": row["status"], "xla_pipelined_ms": f"{s_x['pipelined_median_ms']:.4f}",
                         "pallas_pipelined_ms": f"{s_p['pipelined_median_ms']:.4f}",
                         "speedup_pipelined": f"{s_p['speedup_vs_ref_pipelined']:.3f}",
                         "xla_per_call_ms": f"{s_x['per_call_median_ms']:.4f}",
                         "pallas_per_call_ms": f"{s_p['per_call_median_ms']:.4f}",
                         "speedup_per_call": f"{s_p['speedup_vs_ref_per_call']:.3f}"})
  hc.write_csv(RESULTS_DIR / "B_timing_raw.csv", raw_rows)
  hc.write_csv(RESULTS_DIR / "B_timing_summary.csv", summary_rows)


def M_str(m: int) -> str:
  return f"{m}"


# ----------------------------------------------------------------------------
# Experiment C
# ----------------------------------------------------------------------------

def run_C(cpu_smoke: bool = False) -> None:
  RESULTS_DIR.mkdir(exist_ok=True)
  e = 0
  files = expert_files(e, names=("w1",))
  hc.record_session_environment(RESULTS_DIR, "C", files)
  print("=== Experiment C (CORRECTNESS PROTOTYPE): in-kernel MXFP4 dequantization, ONE matrix (gate/w1), "
        "expert 0, layer 1 ===")
  packed = jnp.asarray(np.load(files[0]))
  scale = jnp.asarray(np.load(files[1]))
  print(f"[weights] packed {packed.shape} {packed.dtype}, scale {scale.shape} {scale.dtype}, scale byte range "
        f"{int(scale.min())}..{int(scale.max())}")
  assert int(scale.min()) >= 1, "scale byte 0 is a documented numerical difference of the kernel"
  status = {"report_as": "correctness prototype (timing is recorded as a measurement only; no performance "
                         "expectation is stated or assumed)",
            "precision_conventions": PRECISION_CONVENTIONS,
            "default_matmul_precision_config": str(getattr(jax.config, "jax_default_matmul_precision", "unavailable")),
            "stage0_scale_padding": None, "stage1_dequant_bit_exact": None, "stage1_value_exact": None, "stage2_matmul_correct": None,
            "stage3_reads_compressed": None, "stage4_timed": False}
  print("[C] precision conventions of the three matmuls:\n" + json.dumps(PRECISION_CONVENTIONS, indent=2))
  _, scale_p = om.prepare_weights(packed, scale)
  status["stage0_scale_padding"] = om.check_scale_padding(scale, scale_p, packed.shape[1])
  print(f"[C stage0] zero-padded scale: {status['stage0_scale_padding']}")
  if not status["stage0_scale_padding"]["ok"]:
    (RESULTS_DIR / "C_status.json").write_text(json.dumps(status, indent=2))
    print("C STOPS: scale padding check failed")
    return
  w_deq_nk = om.reference_dequant_bf16(packed, scale)          # (N, K) bf16, pre-dequantized once, outside timing
  w_kn = jnp.asarray(w_deq_nk.T)                                # (K, N) for the pre-dequantized matmuls
  jax.block_until_ready((w_kn, scale_p))

  # Stage 1: dequantization alone, on hardware, bit for bit against the verified dequantizer.
  try:
    for bn in (256, 128):
      try:
        lo, hi = om.dequant_only(packed, scale_p, bn=bn)
        jax.block_until_ready((lo, hi))
        break
      except Exception as ex:  # noqa: BLE001
        print(f"[C stage1] bn={bn} failed: {type(ex).__name__}: {str(ex)[:800]}")
        lo = None
    if lo is None:
      raise RuntimeError("dequant_only failed to compile at every tile size")
    exact_halves = bool(jnp.array_equal(lo.view(jnp.uint16), w_deq_nk[:, 0::2].view(jnp.uint16))) and bool(
        jnp.array_equal(hi.view(jnp.uint16), w_deq_nk[:, 1::2].view(jnp.uint16)))
    # Equivalent-layout check of the COMPLETE dequantized matrix: re-interleave the kernel's low / high
    # halves into the (N, K) layout and compare every element's raw bits with the pre-dequantized
    # weights, then the transposed (K, N) layout the pre-dequantized matmuls actually consume.
    full_nk = jnp.stack([lo, hi], axis=-1).reshape(w_deq_nk.shape)
    n_bad_nk = int(jnp.sum(full_nk.view(jnp.uint16) != w_deq_nk.view(jnp.uint16)))
    n_bad_kn = int(jnp.sum(jnp.asarray(full_nk.T).view(jnp.uint16) != w_kn.view(jnp.uint16)))
    exact = exact_halves and n_bad_nk == 0 and n_bad_kn == 0
    # Classify any bit differences (first hardware run: 634,681 elements differed). Value equality uses
    # bf16 comparison, under which -0.0 == +0.0; the sign of a zero cannot change a matmul result.
    bits_k, bits_r = full_nk.view(jnp.uint16), w_deq_nk.view(jnp.uint16)
    differ = bits_k != bits_r
    value_equal = full_nk == w_deq_nk
    n_value_mismatch = int(jnp.sum(~value_equal))
    n_zero_sign_only = int(jnp.sum(differ & value_equal & (full_nk == 0)))
    n_kernel_pos0_ref_neg0 = int(jnp.sum((bits_k == 0x0000) & (bits_r == 0x8000)))
    pk = np.asarray(packed)
    n_nibble8_in_checkpoint = int(((pk & 0xF) == 8).sum() + (((pk >> 4) & 0xF) == 8).sum())  # FP4 "-0"
    status["stage1_dequant_bit_exact"] = exact
    status["stage1_value_exact"] = n_value_mismatch == 0
    status["stage1_differences_are_only_zero_sign"] = (n_bad_nk == n_zero_sign_only)
    status["stage1_detail"] = {
        "halves_bit_exact": exact_halves, "full_NK_layout_mismatching_elements": n_bad_nk,
        "full_KN_layout_mismatching_elements": n_bad_kn, "elements": int(w_deq_nk.size),
        "value_mismatching_elements_(-0.0==+0.0)": n_value_mismatch,
        "bit_differences_that_are_only_zero_sign": n_zero_sign_only,
        "kernel_+0.0_where_reference_-0.0": n_kernel_pos0_ref_neg0,
        "checkpoint_nibbles_equal_to_8_(FP4_negative_zero)": n_nibble8_in_checkpoint,
        "nonfinite_in_dequantized": int(jnp.sum(~jnp.isfinite(w_deq_nk.astype(jnp.float32))))}
    print(f"[C stage1] in-kernel dequantization vs verified dequantizer, raw bf16 bits: {'EXACT' if exact else 'MISMATCH'} "
          f"| value-exact: {status['stage1_value_exact']} | differences only in the sign of zero: "
          f"{status['stage1_differences_are_only_zero_sign']} | {status['stage1_detail']}")
  except Exception as ex:  # noqa: BLE001
    status["stage1_dequant_bit_exact"] = f"ERROR {type(ex).__name__}"
    print(f"[C stage1] ERROR (full text follows)\n{ex}")
  (RESULTS_DIR / "C_status.json").write_text(json.dumps(status, indent=2))
  if status.get("stage1_value_exact") is not True:
    print("C STOPS after stage 1 (dequantized VALUES differ from the verified dequantizer); no matmul stage.")
    return
  if status["stage1_dequant_bit_exact"] is not True:
    print("[C] stage 1 is NOT bit-exact (reported as such above); the values are equal, so stage 2 proceeds "
          "(a zero's sign cannot change a matmul result) and stage 1 stays reported as a bit mismatch.")

  m_list = (8,) if cpu_smoke else (128, 2048)
  raw_rows, summary_rows, corr_rows = [], [], []
  for m in m_list:
    x = make_x(m)
    args = (x, w_kn, packed, scale_p)
    online_fn = None
    for bm, bn in ((m, 256), (m, 128), (max(m // 2, 8), 128), (max(m // 4, 8), 128)):
      if m % bm or bn > om.INTERMEDIATE_SIZE:
        continue
      try:
        cand = functools.partial(lambda x_, wk, p_, s_, bn_=bn, bm_=bm: om.online_dequant_matmul(x_, p_, s_, bm=bm_, bn=bn_))
        out_c = jax.jit(cand)(*args)
        jax.block_until_ready(out_c)
        online_fn, bn_used = cand, bn
        break
      except Exception as ex:  # noqa: BLE001
        print(f"[C stage2] M={m} bm={bm} bn={bn} compile/run failed (full text): {type(ex).__name__}: {str(ex)[:1500]}")
    if online_fn is None:
      corr_rows.append({"M": m, "status": "COMPILE_FAILED"})
      status["stage2_matmul_correct"] = False
      continue
    xla_fn = lambda x_, wk, p_, s_: jnp.dot(x_, wk, preferred_element_type=jnp.float32).astype(x_.dtype)  # noqa: E731
    pal_fn = lambda x_, wk, p_, s_: pallas_matmul(x_, wk, bm=bm, bk=gk.LATENT_SIZE, bn=bn_used)  # noqa: E731
    out_x = jax.jit(xla_fn)(*args)
    out_p = jax.jit(pal_fn)(*args)
    hp = jnp.dot(x.astype(jnp.float32), w_deq_nk.astype(jnp.float32).T)  # f32 reference on the same bf16 weights
    rep_online = hc.error_report(out_c, out_x)
    rep_online_hp = hc.error_report(out_c, hp)
    rep_pal = hc.error_report(out_p, out_x)
    ok = rep_online["relative_max_diff"] < TOLERANCE and rep_online["nonfinite_out"] == 0
    corr_rows.append({"M": m, "bm": bm, "bn": bn_used, "status": "OK" if ok else "FAIL",
                      "online_vs_xla_predequantized": rep_online, "online_vs_f32_reference": rep_online_hp,
                      "pallas_predequantized_vs_xla_predequantized": rep_pal})
    print(f"[C stage2] M={m} bm={bm} bn={bn_used}: {'OK' if ok else 'FAIL'} online vs XLA(pre-dequantized): "
          f"max_abs={rep_online['max_abs_err']:.4e} rmse={rep_online['rmse']:.4e} "
          f"relative_max_diff(=max_abs/ref_std)={rep_online['relative_max_diff']:.4f} "
          f"nonfinite={rep_online['nonfinite_out']} n_diff={rep_online['n_diff']}/{rep_online['n_elements']}")
    status["stage2_matmul_correct"] = ok if status["stage2_matmul_correct"] is None else (status["stage2_matmul_correct"] and ok)
    if m == m_list[0]:
      insp = om.inspect_compiled(online_fn, args, om.INTERMEDIATE_SIZE, om.LATENT_SIZE)
      status["stage3_reads_compressed"] = insp
      print(f"[C stage3] compiled-module inspection: {insp}")
    if not ok:
      print(f"[C] M={m}: correctness failed -> no timing for this M (locate the problem first)")
      continue
    fns = {"xla_matmul_predequantized": xla_fn, "pallas_matmul_predequantized": pal_fn,
           "pallas_online_dequant_mxfp4": online_fn}
    summ = hc.rotated_timing(fns, args, f"C timing M={m}", raw_rows,
                             point={"experiment": "C", "expert": 0, "M": m, "bm": bm, "bn": bn_used})
    status["stage4_timed"] = True
    for name, s in summ.items():
      summary_rows.append({"M": m, "tile": f"bm={bm},bn={bn_used}", "candidate": name,
                           "pipelined_ms": f"{s['pipelined_median_ms']:.4f}", "per_call_ms": f"{s['per_call_median_ms']:.4f}",
                           "speedup_vs_xla_pipelined": f"{s['speedup_vs_ref_pipelined']:.3f}",
                           "speedup_vs_xla_per_call": f"{s['speedup_vs_ref_per_call']:.3f}"})
  (RESULTS_DIR / "C_correctness.json").write_text(json.dumps(corr_rows, indent=2))
  (RESULTS_DIR / "C_status.json").write_text(json.dumps(status, indent=2))
  hc.write_csv(RESULTS_DIR / "C_timing_raw.csv", raw_rows)
  hc.write_csv(RESULTS_DIR / "C_timing_summary.csv", summary_rows)
  print("\nC STATUS (report each stage separately):\n" + json.dumps(status, indent=2))


if __name__ == "__main__":
  which = sys.argv[1] if len(sys.argv) > 1 else "B"
  smoke = len(sys.argv) > 2 and sys.argv[2] == "cpu_smoke"
  print(f"devices: {jax.devices()}  jax: {jax.__version__}  experiment: {which}  cpu_smoke={smoke}")
  if which not in ("B", "C"):
    raise SystemExit("usage: real_mxfp4_hw_validation.py [B|C] [cpu_smoke]")
  try:
    (run_B if which == "B" else run_C)(cpu_smoke=smoke)
  except Exception:  # noqa: BLE001 -- recorded, then re-raised so the session driver sees the failure
    import traceback
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / f"{which}_fatal_error.txt").write_text(traceback.format_exc())
    raise
