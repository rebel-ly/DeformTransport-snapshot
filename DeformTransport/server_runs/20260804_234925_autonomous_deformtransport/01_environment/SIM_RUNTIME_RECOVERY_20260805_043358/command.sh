#!/usr/bin/env bash
set -euo pipefail

BASE_PY=/workspace/tools/miniforge3/envs/realwonder-gen/bin/python
SIM_VENV=/workspace/tools/venvs/deformtransport-sim
PROJECT=/workspace/DeformTransport

"${BASE_PY}" -m venv --system-site-packages "${SIM_VENV}"
"${SIM_VENV}/bin/python" -m pip install -r "${PROJECT}/requirements-stage1-wsl.txt"
"${SIM_VENV}/bin/python" -m pip check
