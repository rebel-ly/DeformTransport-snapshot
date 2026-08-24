# DeformTransport Autonomous Research Run

- RUN_ID: `20260804_234925_autonomous_deformtransport`
- Started: 2026-08-04 23:49:25 +08:00
- Host project: `/mnt/sdbd/home/liuyu_qyh/DeformTransport`
- Container project: `/workspace/DeformTransport`

## Dependency order

1. Audit host, container, Git, environment, resources, code, models, cases, and evaluation assets.
2. Freeze the environment and source state without modifying or deleting existing work.
3. Repair only the first proven missing dependency, under the pinned constraints.
4. Validate tests, compileability, CUDA, model paths, and the transport payload.
5. Reconstruct commands from source and existing artifacts.
6. Run one-GPU Baseline smoke only when a GPU is genuinely idle.
7. Verify baseline equivalence and transport injection.
8. Run Santa Baseline, Correct, and Shuffled sequentially on the same GPU.
9. Evaluate metrics and diagnostics, then perform evidence-driven single-factor iterations.
10. Add a fair flow baseline, official cases, and a real robot deformable-object case when assets permit.

## Concurrency policy

Read-only CPU audits may run concurrently. Environment mutation, first smoke, and each case's first Baseline/Correct/Shuffled comparison are serial. Every full generation process gets an isolated output directory; no GPU is used while occupied by another task.

## Budgets

12 wall-clock hours, 30 aggregate GPU-hours, 150 GB new disk, 24 heavy generations, four major method iterations, three retries per root cause, and five minimal missing-dependency repairs.
