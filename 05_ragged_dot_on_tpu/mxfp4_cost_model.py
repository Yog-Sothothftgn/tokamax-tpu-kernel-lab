"""Arithmetic comparison of the three gate-matmul paths measured in the hardware
session (experiment C): XLA matmul on pre-dequantized BF16 weights, the hand-written
Pallas matmul on the same BF16 weights, and the in-kernel MXFP4 dequantization
prototype. Pure arithmetic from the shapes, the tile configuration that was run and
the TIMINGS MEASURED ON HARDWARE (copied below from 5b_C_rerun.log, pipelined
medians, ms); no TPU needed.

  python mxfp4_cost_model.py            # prints the page and writes online_mxfp4_vs_bf16_page.md

Nominal chip numbers come from the logged `get_tpu_info()` of the session:
HBM 1.64e12 B/s, bf16 MXU 9.2e14 FLOP/s. Floors computed from them are lower
bounds for a perfectly overlapped kernel, not predictions.
"""

import pathlib

K, N, BN, GROUP = 3584, 3072, 256, 32
HBM_BPS, MXU_FLOPS = 1.64e12, 9.2e14
KH = K // 2
GP = ((K // GROUP + 127) // 128) * 128  # padded scale groups (128)

# Measured on the v6e (5b_C_rerun.log): pipelined median ms, per-call median ms.
MEASURED = {
    128: {"xla_predeq": (0.0426, 0.1320), "pallas_predeq": (0.0415, 0.1396), "online": (0.0811, 0.1969)},
    2048: {"xla_predeq": (0.0691, 0.1849), "pallas_predeq": (0.0756, 0.1905), "online": (0.6346, 0.7596)},
}


def mb(b):
  return f"{b / 1e6:.2f} MB"


def page() -> str:
  out = []
  w = out.append
  w("# Online MXFP4 vs BF16 baselines: operations, data volume, tiling (gate matrix, one expert)\n")
  w(f"Shapes: x (M, {K}) bf16, W (N = {N}, K = {K}); y = x @ W^T (M, {N}) bf16. Tile run on hardware: bm = M, bn = {BN}, full K, "
    f"so the grid has 1 m-tile x {N // BN} n-tiles in both measured cases (M = 128 and M = 2048).\n")
  w("## 1. What each path does per call\n")
  w("| | XLA, pre-dequantized | Pallas, pre-dequantized | Pallas, online MXFP4 |")
  w("|---|---|---|---|")
  w(f"| Weights in HBM | bf16 W (K, N), dequantized once outside the call | same array | packed u8 (N, K/2) = ({N}, {KH}) + scale u8 (N, {GP}) (zero-padded from {K // GROUP}) |")
  w("| Outside-kernel work in the call | none (one fused dot) | none | two stride-2 slices of x -> x_even, x_odd, each (M, K/2) |")
  w(f"| In-kernel work per n-tile (12 of them) | - | one dot (M, K) x (K, {BN}) | decode {BN}x{KH} low + {BN}x{KH} high nibbles (int ops + FP4 -> float), "
    f"expand scales with a ({BN}, {GP}) x ({GP}, {KH}) 0/1 matmul, multiply, cast to bf16; then TWO transposed-rhs dots (M, K/2) x ({BN}, K/2)^T, float32 accumulation, add, cast |")
  w("| Accumulation | float32, XLA order | float32 scratch, one dot | float32, two dots summed |")
  w("| Rounding of the result | one bf16 rounding | one | one |")
  w("")
  w("## 2. Data volume per call (bytes; exact from shapes)\n")
  w("| | M = 128 | M = 2048 |")
  w("|---|---|---|")
  rows = {}
  for m in (128, 2048):
    xb = m * K * 2
    yb = m * N * 2
    wb_bf16 = K * N * 2
    wb_q = N * KH + N * GP
    rows[m] = dict(
        xb=xb, yb=yb, wb_bf16=wb_bf16, wb_q=wb_q,
        base=wb_bf16 + xb + yb,
        online=wb_q + 3 * xb + yb,  # x: read by the slices, written (even+odd), read by the kernel
    )
  w(f"| Weights read, BF16 paths | {mb(rows[128]['wb_bf16'])} | {mb(rows[2048]['wb_bf16'])} |")
  w(f"| Weights read, online (packed + padded scale) | {mb(rows[128]['wb_q'])} ({rows[128]['wb_bf16'] / rows[128]['wb_q']:.2f}x fewer) | {mb(rows[2048]['wb_q'])} |")
  w(f"| x (bf16) | {mb(rows[128]['xb'])} | {mb(rows[2048]['xb'])} |")
  w(f"| x traffic, BF16 paths (read once) | {mb(rows[128]['xb'])} | {mb(rows[2048]['xb'])} |")
  w(f"| x traffic, online (read + write the two halves + read in kernel) | {mb(3 * rows[128]['xb'])} | {mb(3 * rows[2048]['xb'])} |")
  w(f"| y written | {mb(rows[128]['yb'])} | {mb(rows[2048]['yb'])} |")
  w(f"| **Minimum HBM traffic, BF16 paths** | {mb(rows[128]['base'])} | {mb(rows[2048]['base'])} |")
  w(f"| **Minimum HBM traffic, online** | {mb(rows[128]['online'])} | {mb(rows[2048]['online'])} |")
  w("")
  w("## 3. Floors from nominal chip numbers, and what was measured (pipelined ms)\n")
  w("| | M = 128 | M = 2048 |")
  w("|---|---|---|")
  fl = {}
  for m in (128, 2048):
    flops = 2 * m * K * N
    fl[m] = dict(flops=flops, mxu=flops / MXU_FLOPS * 1e3, hbm_base=rows[m]["base"] / HBM_BPS * 1e3,
                 hbm_online=rows[m]["online"] / HBM_BPS * 1e3)
  w(f"| FLOPs (2 M K N) | {fl[128]['flops'] / 1e9:.2f} G | {fl[2048]['flops'] / 1e9:.2f} G |")
  w(f"| MXU floor | {fl[128]['mxu'] * 1e3:.1f} us | {fl[2048]['mxu'] * 1e3:.1f} us |")
  w(f"| HBM floor, BF16 paths / online | {fl[128]['hbm_base'] * 1e3:.1f} / {fl[128]['hbm_online'] * 1e3:.1f} us | {fl[2048]['hbm_base'] * 1e3:.1f} / {fl[2048]['hbm_online'] * 1e3:.1f} us |")
  for key, label in (("xla_predeq", "XLA, pre-dequantized"), ("pallas_predeq", "Pallas, pre-dequantized"), ("online", "Pallas, online MXFP4")):
    a, b = MEASURED[128][key][0], MEASURED[2048][key][0]
    w(f"| Measured: {label} | {a * 1e3:.1f} us | {b * 1e3:.1f} us ({fl[2048]['flops'] / (b * 1e-3) / 1e12:.0f} TFLOP/s) |")
  w("")
  w("## 4. What the arithmetic rules out and narrows\n")
  t_on_128, t_on_2048 = MEASURED[128]["online"][0], MEASURED[2048]["online"][0]
  t_pre_128, t_pre_2048 = MEASURED[128]["pallas_predeq"][0], MEASURED[2048]["pallas_predeq"][0]
  extra_128, extra_2048 = t_on_128 - t_pre_128, t_on_2048 - t_pre_2048
  slope = (extra_2048 - extra_128) / (2048 - 128)  # ms per row
  intercept = extra_128 - slope * 128
  w("**Structural fact.** In both measured cases the grid is (1, 12): the weights are dequantized exactly once, "
    "the same 11,010,048 elements, whatever M is. The dequantization work therefore does not depend on M. "
    "(Assumption, not verified: the compiler materialises each dequantized tile once per grid step and does not "
    "recompute it per row block of the dot.)\n")
  w(f"**Bound on the dequantization cost D.** D is part of the online kernel's time at M = 128, so D <= {t_on_128 * 1e3:.1f} us. "
    f"At M = 2048 the same D sits inside {t_on_2048 * 1e3:.1f} us, so dequantization is at most {t_on_128 / t_on_2048 * 100:.1f} percent of that call and "
    f"at least {(t_on_2048 - t_on_128) * 1e3:.0f} us ({(1 - t_on_128 / t_on_2048) * 100:.1f} percent) of it is M-dependent work. "
    "(Under the extra assumption that the online kernel cannot save more than the HBM time of the weight bytes it skips, "
    f"{(rows[128]['wb_bf16'] - rows[128]['wb_q']) / HBM_BPS * 1e6:.1f} us, the bound tightens to about "
    f"{(extra_128 * 1e3 + (rows[128]['wb_bf16'] - rows[128]['wb_q']) / HBM_BPS * 1e6):.0f} us.)\n")
  w(f"**Linear scaling.** Excess of the online kernel over the Pallas pre-dequantized matmul (same grid, same tile, same MXU FLOPs): "
    f"{extra_128 * 1e3:.1f} us at M = 128 and {extra_2048 * 1e3:.1f} us at M = 2048, i.e. {extra_2048 / extra_128:.1f}x for 16x the rows. "
    f"A two-point line gives {slope * 1e6:.0f} ns per row with an intercept of {intercept * 1e3:.1f} us (two points only; a "
    "description, not a fitted model). A constant dequantization cost cannot produce this.\n")
  w("**What each remaining M-dependent suspect would have to do.**")
  xtraffic = 2 * rows[2048]["xb"]
  w(f"* Outside stride-2 slicing of x: it moves {mb(xtraffic)} extra at M = 2048 ({HBM_BPS and xtraffic / HBM_BPS * 1e6:.0f} us at nominal HBM bandwidth). "
    f"To explain the {extra_2048 * 1e3:.0f} us excess it would have to run at {xtraffic / (extra_2048 * 1e-3) / 1e9:.0f} GB/s, "
    f"{xtraffic / (extra_2048 * 1e-3) / HBM_BPS * 100:.1f} percent of nominal bandwidth. Implausible for a plain copy, possible for a lane-shuffle relayout. "
    "Not excluded.")
  w(f"* Two K/2 transposed-rhs dots: the online kernel's {fl[2048]['flops'] / 1e9:.1f} GFLOP in {t_on_2048:.4f} ms is "
    f"{fl[2048]['flops'] / (t_on_2048 * 1e-3) / 1e12:.0f} TFLOP/s, {fl[2048]['flops'] / (t_on_2048 * 1e-3) / MXU_FLOPS * 100:.1f} percent of the nominal MXU rate, against "
    f"{fl[2048]['flops'] / (t_pre_2048 * 1e-3) / 1e12:.0f} TFLOP/s ({fl[2048]['flops'] / (t_pre_2048 * 1e-3) / MXU_FLOPS * 100:.0f} percent) for the pre-dequantized Pallas matmul. "
    "Not excluded.")
  w("* Weight bytes (3.7x fewer) and the grid shape are not suspects: both are better or identical to the BF16 Pallas path.")
  w("")
  w("Status after this page: dequantization arithmetic is ruled out as the dominant cost (at most ~13 percent by the bound above). "
    "Remaining: the outside deinterleave of x and the dot structure inside the kernel. See mxfp4_deinterleave_experiment.py "
    "for the single-variable test of the first.")
  return "\n".join(out) + "\n"


if __name__ == "__main__":
  text = page()
  print(text)
  (pathlib.Path(__file__).resolve().parent / "online_mxfp4_vs_bf16_page.md").write_text(text, encoding="utf-8")
