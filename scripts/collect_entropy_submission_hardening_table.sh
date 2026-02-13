#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <hardening_out_root>" >&2
  exit 2
fi

root="$1"
real_dir="$root/realdata_long_mapvar_256c"
abl_dir="$root/ablation_suite"
syn_dir="$root/synthetic_closure_bh_sbc"

tsv="$root/submission_hardening_summary.tsv"
md="$root/submission_hardening_summary.md"

mkdir -p "$root"

status_from_runtime() {
  local runtime="$1"
  local stage_dir="$2"
  local report_md="$stage_dir/report.md"
  if [ -f "$runtime" ]; then
    local has_exit=0
    local has_signal=0
    local has_exception=0
    local has_report=0
    rg -q "EXIT normal" "$runtime" && has_exit=1 || true
    rg -q "SIGNAL" "$runtime" && has_signal=1 || true
    rg -q "EXCEPTION|Traceback" "$runtime" && has_exception=1 || true
    [ -f "$report_md" ] && has_report=1

    if [ "$has_report" -eq 1 ] && [ "$has_exit" -eq 1 ] && [ "$has_exception" -eq 0 ]; then
      echo "done"
      return
    fi
    if pgrep -af "$(basename "$stage_dir")" >/dev/null 2>&1; then
      echo "running"
      return
    fi
    if [ "$has_report" -eq 1 ]; then
      echo "done"
      return
    fi
    if [ "$has_exception" -eq 1 ]; then
      echo "failed"
      return
    fi
    if [ "$has_signal" -eq 1 ]; then
      echo "interrupted"
      return
    fi
    echo "incomplete"
    return
  fi
  echo "pending"
}

fmt() {
  local v="$1"
  if [ -z "$v" ] || [ "$v" = "null" ] || [ "$v" = "NA" ]; then
    echo "NA"
  else
    printf "%s" "$v"
  fi
}

real_status="$(status_from_runtime "$real_dir/runtime.log" "$real_dir")"
abl_status="$(status_from_runtime "$abl_dir/runtime.log" "$abl_dir")"
if [ "$abl_status" = "pending" ] && [ -f "$abl_dir/report.md" ]; then
  abl_status="done"
fi
syn_status="$(status_from_runtime "$syn_dir/runtime.log" "$syn_dir")"
if [ "$syn_status" = "pending" ] && [ -f "$syn_dir/report.md" ]; then
  syn_status="done"
fi

real_h0="NA"
real_h0_band="NA"
real_om="NA"
real_acc="NA"
real_ess="NA"
real_map_delta="NA"
real_map_abs="NA"
if [ -f "$real_dir/tables/summary.json" ]; then
  real_h0="$(jq -r '.posterior.H0.p50 // "NA"' "$real_dir/tables/summary.json")"
  real_h0_band="$(jq -r '"[" + ((.posterior.H0.p16 // "NA")|tostring) + ", " + ((.posterior.H0.p84 // "NA")|tostring) + "]"' "$real_dir/tables/summary.json")"
  real_om="$(jq -r '.posterior.omega_m0.p50 // "NA"' "$real_dir/tables/summary.json")"
  real_acc="$(jq -r '.mu_sampler.acceptance_fraction_mean // "NA"' "$real_dir/tables/summary.json")"
  real_ess="$(jq -r '.mu_sampler.ess_min // "NA"' "$real_dir/tables/summary.json")"
fi
if [ -f "$real_dir/tables/proximity.json" ]; then
  real_map_delta="$(jq -r 'if (.mapping_sensitivity and (.mapping_sensitivity.deltas|length>0)) then ([.mapping_sensitivity.deltas[][1]]|max) else "NA" end' "$real_dir/tables/proximity.json")"
  real_map_abs="$(jq -r 'if (.mapping_sensitivity and (.mapping_sensitivity.deltas|length>0)) then ([.mapping_sensitivity.deltas[][2]]|max) else "NA" end' "$real_dir/tables/proximity.json")"
fi

abl_cases="NA"
abl_bh_range="NA"
abl_kbest="NA"
if [ -f "$abl_dir/tables/ablations.json" ]; then
  abl_cases="$(jq -r 'length' "$abl_dir/tables/ablations.json")"
  abl_bh_range="$(jq -r '"[" + (([.[].D2_bh]|min)|tostring) + ", " + (([.[].D2_bh]|max)|tostring) + "]"' "$abl_dir/tables/ablations.json")"
  abl_kbest="$(jq -r '([.[].D2_kaniadakis]|min) // "NA"' "$abl_dir/tables/ablations.json")"
fi

syn_cov68="NA"
syn_cov95="NA"
syn_covH68="NA"
syn_inv="NA"
syn_n="NA"
if [ -f "$syn_dir/tables/summary.json" ]; then
  syn_n="$(jq -r '.sbc.N // "NA"' "$syn_dir/tables/summary.json")"
  syn_cov68="$(jq -r '.sbc.coverage.logmu_68 // "NA"' "$syn_dir/tables/summary.json")"
  syn_cov95="$(jq -r '.sbc.coverage.logmu_95 // "NA"' "$syn_dir/tables/summary.json")"
  syn_covH68="$(jq -r '.sbc.coverage.H_68 // "NA"' "$syn_dir/tables/summary.json")"
  syn_inv="$(jq -r '.sbc.logprob.invalid_rate // "NA"' "$syn_dir/tables/summary.json")"
fi

{
  echo -e "stage\tstatus\tkey_metrics\tnotes\treport"
  echo -e "realdata_long_mapvar\t$real_status\tH0_p50=$(fmt "$real_h0"); H0_16_84=$(fmt "$real_h0_band"); Om_p50=$(fmt "$real_om"); acc=$(fmt "$real_acc"); ess_min=$(fmt "$real_ess")\tmapping_rms_sigma_max=$(fmt "$real_map_delta"); mapping_max_abs_dlogmu=$(fmt "$real_map_abs")\t$real_dir/report.md"
  echo -e "ablation_suite\t$abl_status\tcases=$(fmt "$abl_cases"); D2_BH_range=$(fmt "$abl_bh_range")\tbest_D2_kaniadakis=$(fmt "$abl_kbest")\t$abl_dir/report.md"
  echo -e "synthetic_closure_bh_sbc\t$syn_status\tSBC_N=$(fmt "$syn_n"); cov_logmu_68=$(fmt "$syn_cov68"); cov_logmu_95=$(fmt "$syn_cov95"); cov_H_68=$(fmt "$syn_covH68")\tinvalid_logprob_rate=$(fmt "$syn_inv")\t$syn_dir/report.md"
} > "$tsv"

{
  echo "# Entropy Submission Hardening Summary"
  echo
  echo "| Stage | Status | Key Metrics | Notes | Report |"
  echo "|---|---|---|---|---|"
  awk -F '\t' 'NR>1 {printf "| %s | %s | %s | %s | `%s` |\n", $1, $2, $3, $4, $5}' "$tsv"
} > "$md"

echo "Wrote $tsv"
echo "Wrote $md"
