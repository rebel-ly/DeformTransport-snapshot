set -euo pipefail

# 本文件是人工审查清单。默认不执行环境修改、GPU实验或交互式命令。
# 项目宿主路径：/mnt/sdbd/home/liuyu_qyh/DeformTransport
# 项目容器路径：/workspace/DeformTransport

# 1. 进入容器（人工确认后复制执行；交互命令默认注释）
# docker exec -it deformtransport-dev bash

# 2. 激活 realwonder-gen（在容器内人工执行）
# source /workspace/tools/miniforge3/etc/profile.d/conda.sh
# conda activate realwonder-gen
# cd /workspace/DeformTransport

# 3. 查看 Git 状态（兼容服务器旧版 Git）
# git symbolic-ref --short HEAD
# git rev-parse HEAD
# git status --short
# git diff --check

# 4. 查看当前活动进程；不要操作其他用户进程
# ps -p 291997,89794,86928,86942 -o pid,ppid,user,lstart,etime,stat,cmd
# tr '\0' ' ' </proc/291997/cmdline

# 5. 查看 GPU 与内存状态（只读）
# nvidia-smi
# awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo

# 6. 查看最新实验结果和 81 帧失败现场
# less server_runs/20260804_234925_autonomous_deformtransport/human_handoff/RESULT_INDEX.md
# cat server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/current_stage.txt
# cat server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/exit_code.txt
# tail -n 80 server_runs/20260804_234925_autonomous_deformtransport/04_smoke/OFFICIAL_SANTA_81F_CHAIN_20260805_050719/stderr.log

# 7. 打开五组视频路径（先核对文件；播放器命令按本机情况人工选择）
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_BASELINE_20260805_032928/santa_baseline_seed0.mp4
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_CORRECT_20260805_033343/santa_correct_seed0.mp4
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_SHUFFLED_20260805_033730/santa_shuffled_seed0.mp4
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_FLOW_20260805_035223/santa_flow_seed0.mp4
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/04_smoke/REALWONDER_SANTA_BLEND_20260805_040147/santa_blend_seed0.mp4
# ls -lh server_runs/20260804_234925_autonomous_deformtransport/05_metrics/realwonder_santa_proxy_trio_20260805/methods_contact_sheet.jpg

# 8. 检查隔离前端环境（只读 import/pip check；人工确认后执行）
# export SETUPTOOLS_USE_DISTUTILS=stdlib
# export LD_LIBRARY_PATH=/workspace/tools/conda-libs/deformtransport-gl/lib:${LD_LIBRARY_PATH:-}
# /workspace/tools/venvs/deformtransport-sim/bin/python -c 'import genesis, pytorch3d, cv2, trimesh; print("imports ok", cv2.__version__, trimesh.__version__)'
# /workspace/tools/venvs/deformtransport-sim/bin/python -m pip check
# /workspace/tools/miniforge3/envs/realwonder-gen/bin/python -m pip check

# 9. 继续最小 official case smoke：先修复/绕开 PyAV 并复用已有 simulation_source。
# 禁止直接重启原 81 帧 chain；以下仅供 code review 后手工构造新的唯一运行目录。
# sed -n '1,180p' scripts/assemble_final_sim_from_trajectory.py
# sed -n '1,180p' demo_web/simulation/utils.py
# /workspace/tools/miniforge3/envs/realwonder-gen/bin/python -u scripts/assemble_final_sim_from_trajectory.py --source-dir <既有simulation_source> --demo-data demo_web/demo_data/santa_cloth --output-dir <新的唯一final_sim目录> --seed 0

# 10. 运行 CPU 单元测试（人工确认环境后执行）
# /workspace/tools/miniforge3/envs/realwonder-gen/bin/python -m unittest discover -s tests -v
# /workspace/tools/miniforge3/envs/realwonder-gen/bin/python -m compileall -q deform_transport scripts tests infer_sim.py

# 11. 方法迭代前必须先检查的文件
# less server_runs/20260804_234925_autonomous_deformtransport/human_handoff/OPEN_ISSUES.md
# less server_runs/20260804_234925_autonomous_deformtransport/human_handoff/CODE_CHANGE_INDEX.md
# less server_runs/20260804_234925_autonomous_deformtransport/human_handoff/RESULT_INDEX.md
# sed -n '1,360p' infer_sim.py
# sed -n '1,280p' deform_transport/pipeline_integration.py
# sed -n '1,260p' deform_transport/trajectory.py
# sed -n '1,280p' wan/modules/attention.py

# 12. GPU 任务均保持注释。只有 official final_sim 契约校验、资源检查和人工审批后，
# 才可从 OFFICIAL_SANTA_81F_TRIO_20260805_050719/command.sh 或 scheduler.sh 中逐条审查执行。
# 不要直接执行 scheduler.sh；不得停止其他用户进程。
