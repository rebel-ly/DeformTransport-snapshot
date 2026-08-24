# Historical Python provenance

`/workspace/DeformTransport/server_runs/wan_move_method_v4/20260810_164423__wave1_denoise_noise_hybrid_seed0/resource_watch.log` contains numerous actual historical process records for `/workspace/tools/miniforge3/envs/wan-move/bin/python generate.py`. The recorded Santa command uses `/workspace/Wan-Move/Wan-Move-14B-480P`, 81 frames, 480×832, and the Santa prompt/image lineage; these are successful operational Wan-Move generation records, so `SUCCESSFUL_RUN_EVIDENCE=True` and `PYTHON_PATH_EXPLICIT=True`.

`/workspace/DeformTransport/server_runs/wan_move_method_eval/20260810_121513__v3s_v3b_v3c_v3d_v3e_joint_eval/supervisor.sh:7` also explicitly binds `PY=/workspace/tools/miniforge3/envs/wan-move/bin/python`. These provenance records outrank default container `/usr/bin/python` and other discovered environment names.

No source, package, or environment was altered during this audit.
