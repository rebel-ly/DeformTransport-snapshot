#!/usr/bin/env bash
set -euo pipefail

root=/workspace/DeformTransport
run_root=$root/server_runs/20260804_234925_autonomous_deformtransport
chain=$run_root/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719
job=$run_root/04_smoke/OFFICIAL_SANTA_81F_TRIO_20260805_050719
final_sim=$chain/final_sim
artifact=$chain/wan_vae/vae_latent_outputs.pt
checkpoint=$root/ckpts/Realwonder-Distilled-AR-I2V-Flow/sink_size=1-attn_size=21-frame_per_block=3-denoising_steps=4/step=000800.pt
python=/workspace/tools/miniforge3/envs/realwonder-gen/bin/python
export CUDA_VISIBLE_DEVICES=2
export TORCH_HOME=$run_root/prepared_inputs/torch_cache
trap 'code=$?; echo "$code" > "$job/exit_code.txt"; date --iso-8601=seconds > "$job/结束时间.txt"' EXIT
cd "$root"

test "$(cat "$chain/exit_code.txt")" = 0
"$python" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["通过"] is True' "$final_sim/validation_with_transport_report.json"
date --iso-8601=seconds > "$job/开始时间.txt"

run_variant() {
  mode=$1
  dir=$job/$mode
  output=$dir/santa_official_${mode}_seed0.mp4
  mkdir -p "$dir"
  args=(--checkpoint_path "$checkpoint" --sim_data_path "$final_sim" --output_path "$output" --seed 0 --eval_degradation 0.5 --local_attn_size 21)
  if [ "$mode" != baseline ]; then
    args+=(--transport_latent_path "$artifact" --transport_mode "$mode")
  fi
  free_mib=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | sed -n '3p' | tr -d ' ')
  temp_c=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits | sed -n '3p' | tr -d ' ')
  mem_kib=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
  if [ "$free_mib" -lt 38000 ] || [ "$temp_c" -ge 82 ] || [ "$mem_kib" -lt 31457280 ]; then
    printf '资源门禁失败 freeMiB=%s tempC=%s MemAvailableKiB=%s\n' "$free_mib" "$temp_c" "$mem_kib" > "$dir/resource_gate_failure.txt"
    return 75
  fi
  printf '%q ' "$python" -u infer_sim.py "${args[@]}" > "$dir/inference_command.txt"
  printf '\n' >> "$dir/inference_command.txt"
  date --iso-8601=seconds > "$dir/开始时间.txt"
  nvidia-smi > "$dir/nvidia-smi_before.txt"
  sha256sum "$checkpoint" "$final_sim/config.yaml" "$final_sim/noises.npy" "$final_sim/resized_input_image.png" "$final_sim/prompt.txt" > "$dir/inputs_sha256.txt"
  if [ "$mode" != baseline ]; then sha256sum "$artifact" >> "$dir/inputs_sha256.txt"; fi
  set +e
  timeout --signal=TERM --kill-after=20s 3600s "$python" -u infer_sim.py "${args[@]}" > "$dir/stdout.log" 2> "$dir/stderr.log"
  code=$?
  set -e
  echo "$code" > "$dir/exit_code.txt"
  date --iso-8601=seconds > "$dir/结束时间.txt"
  nvidia-smi > "$dir/nvidia-smi_after.txt"
  if [ "$code" -ne 0 ] || [ ! -s "$output" ]; then return "$code"; fi
  sha256sum "$output" > "$dir/output_sha256.txt"
}

echo baseline > "$job/current_stage.txt"
run_variant baseline
echo correct > "$job/current_stage.txt"
run_variant correct
echo shuffled > "$job/current_stage.txt"
run_variant shuffled
echo 完成 > "$job/current_stage.txt"
