# CPU 环境验收报告

- pip check：通过。
- unittest：25/25 通过。
- compileall：通过。
- git diff --check：通过。
- transport_ready.pt 结构验证：通过。
- 模型路径与容器软链接：通过。
- infer_sim.py --help：依赖已恢复，但入口导入阶段强制初始化 CUDA；无空闲 GPU 时按资源规则延后。
- CUDA 基本张量测试：延后，原因是所有 GPU 均有 A 类其他用户计算任务。

警告仅包括受信任本地测试资产使用 torch.load(weights_only=False) 的未来行为提醒，不影响当前测试结论。
