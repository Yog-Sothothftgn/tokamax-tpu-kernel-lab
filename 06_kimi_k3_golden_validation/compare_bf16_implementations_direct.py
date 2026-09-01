"""Direct, same-device, element-wise comparison between tokamax.ragged_dot's
`xla`/`mosaic` (v1)/`mosaic_tpu_v2` bf16 outputs on the SAME inputs, in the
SAME run.

Closes a gap flagged repeatedly in this project's README/memory: every
prior bf16 finding only compared each implementation's `max_abs_diff`
AGAINST the golden PyTorch reference separately, then noted that the
resulting numbers happened to match across implementations -- concluding
"this suggests a shared bf16/backend precision effect" from matching
SUMMARY STATISTICS computed in different runs, never from a direct
tensor-level diff between the implementations' own raw outputs. This
script computes all available implementations in one process on the same
device and diffs their outputs against EACH OTHER directly, so that
inference can finally be checked against real evidence instead of just
being repeated with a caveat.

Not a pass/fail tolerance check like `test_ragged_dot_against_pytorch_golden.py`'s
`run_one` -- this is a diagnostic comparison. It reports the actual direct
diff; interpreting whether that diff is "expected bf16 noise" or something
else is a judgment call for whoever reads the output, informed by this
number plus the existing vs-golden numbers.

Has a real tokamax dependency (via `_run_instrumented_ragged_dot`) -- CANNOT
be verified locally, must run on the v6e TPU VM.

Usage:
  python compare_bf16_implementations_direct.py --bundle-set mosaic_wide
"""

import argparse
import itertools
import pathlib
import sys

import jax  # noqa: E402
import jax.numpy as jnp
import numpy as np

_HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(_HERE.parent / "05_ragged_dot_on_tpu"))
sys.path.insert(0, str(_HERE))

from test_jax_reference_against_pytorch_golden import _load_bundle, _pt_weights_to_jax  # noqa: E402
from test_ragged_dot_against_pytorch_golden import (  # noqa: E402
    BUNDLE_SETS,
    _run_instrumented_ragged_dot,
)

# Stages worth comparing directly: expert_up_output is where the bf16
# residual first appears in every prior run (see README's Results section);
# final_output is the end-to-end user-visible quantity.
_STAGES_TO_COMPARE = ("expert_up_output", "final_output")


def compare_bf16_implementations(bundle_set: str = "mosaic_wide") -> bool:
  spec = BUNDLE_SETS[bundle_set]
  bundle_dir, config = spec["bf16"]
  implementations = spec["default_implementations"]

  inputs, weights_npz, _outputs, _config_kwargs, metadata = _load_bundle(bundle_dir)
  weights = _pt_weights_to_jax(weights_npz, config, dtype=jnp.bfloat16)
  hidden_states = jnp.array(inputs["hidden_states"][0], dtype=jnp.bfloat16)

  print(f"[bf16-direct-compare] bundle={bundle_dir.name} commit={metadata['source_provenance']['pinned_commit']}")

  outputs = {}
  for impl in implementations:
    try:
      # bf16 -> "default" matmul precision, same as run_one's established
      # pattern (forcing HIGHEST for bf16 crashes mosaic_tpu_v2's compiler,
      # confirmed on hardware -- see _run_instrumented_ragged_dot's docstring).
      with jax.default_matmul_precision("default"):
        final_output, intermediates = _run_instrumented_ragged_dot(
            hidden_states, weights, config, implementation=impl
        )
      outputs[impl] = {
          stage: np.asarray(intermediates[stage]) if stage != "final_output" else np.asarray(final_output)
          for stage in _STAGES_TO_COMPARE
      }
      print(f"[bf16-direct-compare] implementation={impl!r}: computed OK")
    except NotImplementedError as e:
      print(f"[bf16-direct-compare] implementation={impl!r}: SKIPPED ({e})")

  computed = list(outputs.keys())
  if len(computed) < 2:
    print(
        f"[bf16-direct-compare] only {len(computed)} implementation(s) computed successfully "
        "-- need at least 2 to compare directly"
    )
    return False

  print(f"\n[bf16-direct-compare] direct pairwise element-wise comparisons ({len(computed)} implementations computed):")
  for impl_a, impl_b in itertools.combinations(computed, 2):
    for stage in _STAGES_TO_COMPARE:
      a = outputs[impl_a][stage].astype(np.float32)
      b = outputs[impl_b][stage].astype(np.float32)
      max_diff = float(np.max(np.abs(a - b)))
      identical = max_diff == 0.0
      print(
          f"  {stage}: {impl_a!r} vs {impl_b!r} max_abs_diff={max_diff:.3e} "
          f"{'(bit-identical)' if identical else ''}"
      )

  return True


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument(
      "--bundle-set",
      choices=["mosaic", "mosaic_wide"],
      default="mosaic_wide",
      help="'mosaic' (n=20, M=40) can only compute xla (mosaic/mosaic_tpu_v2 below the tiling "
      "floor there); 'mosaic_wide' (n=64, M=128, default) can compute all three, giving the "
      "most complete pairwise comparison",
  )
  args = parser.parse_args()
  ok = compare_bf16_implementations(bundle_set=args.bundle_set)
  sys.exit(0 if ok else 1)
