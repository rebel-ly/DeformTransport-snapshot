# Phase0D-4C parity failure localization

## 1. Objective

Localize the preserved A-original versus B-patched-disabled mismatch using the existing canonical DT-FULL seed-0 artifact before assigning a patch regression. No C arm, epsilon artifact, enabled-path edit, new generation, or GPU3 use occurred.

## 2. Preserved evidence

| arm | video SHA-256 | decoded RGB SHA-256 | frames / shape |
| --- | --- | --- | --- |
| A original | `25e0be0caf53dfb334784ef239f9759fe4f210f43fdb3ea20c746998d75afb07` | `2fd99bf71cb031955c72457af6d3393d184002aba02a0e0b7e8afaa1a26c1030` | 81 / `[81,464,832,3]` |
| B overlay | `b8eebb17ad1753851fa95632249bd1ad971dc7b0107331a17d64407b89329f69` | `ef0180d58a635baf5c2d14dbc6cc2f39a5dfb293ff81417d347029f1c8240069` | 81 / `[81,464,832,3]` |
| canonical | `08785a3ce49d4faa98c8fe8850fed1b2912d8b8fc4fa80ee377de6dfe7bff935` | `935d9301a208abb73e437346c3297a31705563e60ce2aa2be4cf46b44ce7cbc6` | 81 / `[81,464,832,3]` |

The preserved A/B logs, exit codes, wrapper, source provenance, and the textual source diffs are retained under `parity_20260814/`; none was overwritten.

## 3. CPU three-way decoded comparison

| comparison | exact | different channel values | max abs diff | mean abs diff |
| --- | ---: | ---: | ---: | ---: |
| A vs canonical | False | 60,092,735 | 255 | 7.6334019275 |
| B vs canonical | False | 59,950,886 | 255 | 7.4308616221 |
| A vs B | False | 55,284,397 | 214 | 2.5112676238 |

Machine-readable evidence: `parity_20260814/THREE_WAY_CPU_COMPARISON.json`.

## 4. Frozen corrected-v2 evaluation

The same frozen evaluator was applied to existing videos only. The first relative-link wrapper invocation was rejected as an infrastructure failure (`frames=0`); absolute-path rebinding then completed without regenerating video.

| arm | TC-MAR mean | TC-ME mean |
| --- | ---: | ---: |
| A | 17.398205805839215 | 0.728811478690222 |
| B | 17.379338570351067 | 0.7391545451084811 |
| canonical | 17.144317299874714 | 0.7265499674289193 |

The canonical result exactly matches its frozen reference values. Evaluator outputs are retained in `20260814_095925__dt_full_5seed_drop_zero62_formal_evaluation/candidates/parity_localization_*_abs` and `parity_localization_canonical_gpu1`.

## 5. Manifest and RNG audit

The canonical frozen runner exports `DT_TRANSPORT_VARIANT=v3d`, `DT_TRACK_IDS_PATH`, and `DT_TRACK_DEPTH_PATH`. A's preserved launch record and B's preserved wrapper contain none of those three exports. Therefore neither A nor B is launch-manifest equivalent to the canonical V3D run; this directly explains why the canonical match prerequisite is absent.

A and B otherwise share base seed 0, Python runtime, exposed CUDA device 0, checkpoint, source image, prompt, tracks/visibility, 40 UniPC steps, shift 3, bf16, 81 frames, and 480x832 generation domain. Neither run recorded GPU UUID, so that field is `Unresolved` rather than inferred.

Source-level RNG order is identical: `generate.py` resolves seed; `WanMove.generate` makes a private device `torch.Generator`, seeds it with 0, creates initial noise, later calls `torch.manual_seed(0)` immediately before trajectory-feature construction. No NumPy or Python-random draw is used on the seed-0 path. The disabled overlay does not add RNG calls. Thus `RNG_CALL_ORDER_IDENTICAL=True` at source level; cross-run runtime-state equality before construction is not independently captured.

## 6. Disabled-path source audit

`original_vs_overlay_wan_move.diff` contains only: three optional signature parameters; a disabled conditional-consistency check; optional epsilon replacement; and an optional preview/start-index scheduler block. `original_vs_overlay_generate.diff` adds CLI parsing and forwards `None, None, None` in B.

For B, logs prove all three new arguments are `None`. No epsilon replacement, preview latent conversion, `add_noise`, schedule slicing, or `set_begin_index` executes. The only extra disabled-path work is boolean/`None` conditional evaluation. Therefore `DISABLED_BRANCH_OPERATION_ORDER_IDENTICAL=False` under the protocol's literal exact-operation-order criterion, while `DISABLED_BRANCH_RNG_ORDER_IDENTICAL=True` and scheduler/tensor operation order is unchanged.

## 7. First-divergence decision

`THREE_WAY_CLASSIFICATION=CASE_AB3_PARITY_RUNTIME_OR_RNG_REPRODUCTION_PROBLEM`: both A and B differ from canonical. The concrete pre-noise/configuration discrepancy is the absent V3D depth/ID environment configuration in both A and B. A bounded probe is not authorized or necessary for a patch repair because the protocol only authorizes repair after A matches canonical and B independently diverges from it.

`FIRST_DIVERGENCE_STAGE=UNRESOLVED_NOT_PROBED_AFTER_CASE_AB3_CONFIGURATION_MISMATCH`.

## 8. Repair policy result

`PATCH_DISABLED_PATH_REGRESSION_CONFIRMED=Unresolved`; `REPAIR_REQUIRED=False` for the overlay. No source was changed. The evidence instead requires a fresh, correctly manifest-equivalent baseline protocol before an adapter parity claim can be made, which this instruction explicitly does not authorize.

## 9. Stop condition

`C_AUTHORIZED=False`. C0/C1/C2 remain unlaunched; final epsilon is unfrozen; enabled-path sanity remains unresolved. The completed CPU audits specified in the protocol were not reopened. `GPU3_USED=False`, `SERVER_SIDE_CODEX_EXECUTED=False`, `SERVER_LLM_API_USED=False`, `LOCAL_CODEX_ONLY_POLICY=PRESERVED`.
