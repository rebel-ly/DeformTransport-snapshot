# Phase 0D-4D-R3M-R4B final closure

## 1. Scope

Seed1 three-arm, corrected-v2 N=1257 evaluation only. No generation was launched in R4B.

## 2. Frozen evaluator contract

The authoritative evaluator is `eval_v3_corrected_v2_recovered.py` with SHA256 `e6a00e649c928fddfa569ff5c30e641c6653643a6f5a2d59bfbb78b0b2a77ef5`. The evaluation-ID SHA256 is `f94bb0a7986c693e194f750a7afd715f44506518abbb4dd37e0a791380c819b8`; N=1257. The complete frozen input contract is `SEED1_EVALUATION_CONTRACT.json`.

## 3. Blind primary execution gate

The fixed C1, C2, CS order completed with per-arm exit code 0, raw output present, and no persisted Traceback, RuntimeError, or CUDA OOM. `PRIMARY_BATCH_EXECUTION=PASS`.

## 4. Primary TC-ME results

| Arm | Seed1 TC-ME |
| --- | ---: |
| C1 | 0.39610003385941484 |
| C2 | 0.39736386332028023 |
| CS | 0.40577810212784815 |

Lower is better. Seed1 order is C1 < C2 < CS. The full-precision primary record is `SEED1_PRIMARY_TCME_RESULTS.json`.

## 5. Pre-registered mechanism calculations

`C2-C1=0.0012638294608653955`, `C2-CS=-0.00841423880756792`, and `CS-C1=0.009678068268433315`. Therefore seed1 transport direction is FAIL, identity direction is PASS, and wrong-identity harm is PASS.

## 6. Two-seed descriptive replication

Using the recovered authoritative seed0 record: transport direction consistency is FAIL; identity direction consistency is PASS; wrong-identity harm consistency is PASS; strong three-arm order replication is FAIL. Transport paired mean/sample-SD: `-0.02572617638222685` / `0.03816963231183013`. Identity paired mean/sample-SD: `-0.05029887219746387` / `0.05923381659501584`. N_SEEDS=2; no significance or equivalence test was performed.

## 7. Secondary TC-MAR

| Arm | Seed1 TC-MAR |
| --- | ---: |
| C1 | 10.819041036200092 |
| C2 | 9.338139079106316 |
| CS | 11.37400963506194 |

The secondary deltas are C2-C1=`-1.480901957093776`, C2-CS=`-2.035870555955624`, and CS-C1=`0.5549685988618478`. `TCMAR_INDEPENDENCE_FROM_PREVIEW=NOT_ESTABLISHED`; TC-MAR does not alter the primary TC-ME conclusion.

## 8. Evidence integrity

SHA256: contract `def9b5a47c551f5008af30919f978b6760e09c387f2ad900c789ab99fd1e4c02`; primary `8ef74527cdf0849ae1b6cc8da2e7c42fe42a56256105ab417e00cb6e9ce9d090`; two-seed mechanism `effd80d687a37790c5a3384a3953372c9e7b1958a26fa0822c0bcf025958a829`; secondary `dc70f6b46ba140428a2d2fd4724d1cde44b8668f981d9e0ead2d3e913cbcedac`.

## 9. Stop rule

No seed2, RW seed, cross-case, MUSIQ, VBench, subgroup, or new candidate evaluation/generation was launched. `SEED2_FULL_GENERATION_LAUNCHED=False`.
