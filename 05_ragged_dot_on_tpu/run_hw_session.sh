#!/usr/bin/env bash
# Hardware session driver: every step is its own process, a failure does not stop the
# later steps. Order follows the plan: pre-flight, A, B, C. Logs and exit codes go to
# hw_session_results/. Run from 05_ragged_dot_on_tpu with the venv active:
#   bash run_hw_session.sh
set -u
cd "$(dirname "$0")"
OUT=hw_session_results
mkdir -p "$OUT"
run() {
  name=$1; shift
  echo; echo "================ $name ================"
  "$@" 2>&1 | tee "$OUT/$name.log"
  echo "$name exit_status=${PIPESTATUS[0]}" | tee -a "$OUT/session_status.txt"
}
run 0_preflight python3 preflight_hw_session.py
run 1_fetch_weights python3 fetch_real_mxfp4_expert_weights.py --num-experts 3
run 2_preflight_after_fetch python3 preflight_hw_session.py
run 3_A_tokamax_narrow python3 -u tokamax_narrow_gather_experiment.py
run 4_B_real_weights_fused python3 -u real_mxfp4_hw_validation.py B
run 5_C_online_dequant python3 -u real_mxfp4_hw_validation.py C
echo; echo "session finished; results in $OUT/ (copy them back before releasing the VM)"
