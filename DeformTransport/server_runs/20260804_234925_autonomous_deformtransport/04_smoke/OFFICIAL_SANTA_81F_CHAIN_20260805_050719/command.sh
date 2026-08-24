#!/usr/bin/env bash
set -euo pipefail

root=/workspace/DeformTransport
job=$root/server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719
sim=$job/simulation_source
final_sim=$job/final_sim
transport=$job/transport/transport_ready.pt
vae_out=$job/wan_vae
gen_py=/workspace/tools/miniforge3/envs/realwonder-gen/bin/python
sim_py=/workspace/tools/venvs/deformtransport-sim/bin/python

trap 'code=$?; echo "$code" > "$job/exit_code.txt"; date --iso-8601=seconds > "$job/结束时间.txt"' EXIT
cd "$root"
export CUDA_VISIBLE_DEVICES=2
export SETUPTOOLS_USE_DISTUTILS=stdlib
export LD_LIBRARY_PATH=/workspace/tools/conda-libs/deformtransport-gl/lib:${LD_LIBRARY_PATH:-}
export TORCH_HOME=/workspace/DeformTransport/server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/torch_cache
date --iso-8601=seconds > "$job/开始时间.txt"
nvidia-smi > "$job/nvidia-smi_before.txt"

echo "01_compile" > "$job/current_stage.txt"
"$gen_py" -m py_compile \
  demo_web/simulation_engine.py \
  scripts/run_realwonder_trajectory_probe.py \
  scripts/assemble_final_sim_from_trajectory.py \
  server_runs/20260804_234925_autonomous_deformtransport/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py

echo "02_simulation_81f" > "$job/current_stage.txt"
"$sim_py" -u scripts/run_realwonder_trajectory_probe.py \
  --demo-data demo_web/demo_data/santa_cloth \
  --frames 81 --direction right --strength 1 --seed 0 \
  --output-dir "$sim"

"$gen_py" -c 'import json,sys; r=json.load(open(sys.argv[1])); assert r["frames"]==81 and r["all_bindings_match_simulator"] and r["all_positions_finite"]' "$sim/report.json"

echo "03_assemble_final_sim" > "$job/current_stage.txt"
"$gen_py" -u scripts/assemble_final_sim_from_trajectory.py \
  --source-dir "$sim" \
  --demo-data demo_web/demo_data/santa_cloth \
  --output-dir "$final_sim" --seed 0

echo "04_raft_noise" > "$job/current_stage.txt"
"$gen_py" -u server_runs/20260804_234925_autonomous_deformtransport/04_smoke/final_sim_noise_reconstruction_queued/generate_noise.py \
  "$final_sim" --seed 0

echo "05_validate_final_sim" > "$job/current_stage.txt"
"$gen_py" -u server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/validate_final_sim.py \
  "$final_sim" --output "$final_sim/validation_report.json"

echo "06_export_transport_ready" > "$job/current_stage.txt"
"$gen_py" -u scripts/export_transport_ready.py \
  --source-dir "$sim" --output "$transport" --case-name santa_cloth

echo "07_wan_vae_transport" > "$job/current_stage.txt"
"$gen_py" -u scripts/run_wan_vae_transport_probe.py \
  --transport-ready "$transport" \
  --checkpoint wan_models/Wan2.1-Fun-V1.1-1.3B-InP/Wan2.1_VAE.pth \
  --output-dir "$vae_out" --seed 0

echo "08_validate_transport" > "$job/current_stage.txt"
"$gen_py" -u server_runs/20260804_234925_autonomous_deformtransport/prepared_inputs/validate_final_sim.py \
  "$final_sim" --transport-artifact "$vae_out/vae_latent_outputs.pt" \
  --output "$final_sim/validation_with_transport_report.json"

sha256sum \
  "$final_sim/config.yaml" "$final_sim/noises.npy" \
  "$final_sim/resized_input_image.png" "$final_sim/prompt.txt" \
  "$sim/point_trajectories.pt" "$transport" \
  "$vae_out/vae_latent_outputs.pt" > "$job/key_sha256.txt"
nvidia-smi > "$job/nvidia-smi_after.txt"
echo "完成" > "$job/current_stage.txt"
