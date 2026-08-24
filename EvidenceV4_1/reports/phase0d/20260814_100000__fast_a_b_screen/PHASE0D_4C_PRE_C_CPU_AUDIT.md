# Phase0D-4C pre-C CPU audit

## 1. Objective

Run the user-authorized CPU-only lineage, normalization, scheduler, companion-metric, and enabled-interface audit while patched-disabled B ran on GPU0. No Wan model was loaded and `CUDA_VISIBLE_DEVICES=""`, `OMP_NUM_THREADS=1`, and `MKL_NUM_THREADS=1` were set for the audit process.

## 2. Scope and immutability

The audit did not modify original Wan-Move, the running overlay, B's environment, B's logs, or any GPU process. GPU0 B reached its own `COMPLETE` marker and exit code 0 without intervention. GPU1/2/3 were not used. The companion diagnostic source was frozen before any C candidate exists: `scripts/frozen_preview_companion_metrics.py`, SHA-256 `1ee9dbbc4bff4addb5c7fbe570cc7a0e60b644a411ca6b94af7dce09ca3d8f66`.

## 3. DT edited_y historical 65.74 lineage

The historical DT edited_y value is reproducible from `conditioning_alignment/20260813_081824__f2_metric_mechanism_error_localization/scripts/diagnose_rgb_condition.py`, using its persisted `dt_edited_y_decoded_frames/frame_0000.png..frame_0080.png` (81 frames; inherited corrected-v2 S0..S800 timeline), then the frozen patch/Lab aggregation. Its actual chain is `uint8 RGB -> float32 /255 -> evaluator.to_common -> /255`, producing mean/median/p95 `65.7426047584 / 67.4093818665 / 85.6565228653` over 1110 carriers.

Therefore `DT_EDITED_Y_65_74_SAME_DOUBLE_DIV255_BUG = True`.

Under the canonical single-normalization path (`uint8 RGB -> to_common -> /255` once), the same persisted frames give mean/median/p95 `37.8964013103 / 40.5233182907 / 56.7540569735`. This is a condition-representation diagnostic only, not a new formal generation endpoint.

## 4. Formal evaluator contamination audit

The historical F2 diagnostic script pre-divides before calling `to_common`; the frozen corrected-v2 evaluator's `read_video_common` passes decoded uint8 RGB directly to `to_common`, where the only normalization occurs. Thus `FORMAL_EVALUATOR_AFFECTED_BY_DOUBLE_DIV255 = False`; there is no formal-evaluator hard stop.

The historical diagnostic TC-ME values RW `0.8946881631` and DT edited_y `1.5050048806` arose from the same diagnostic script, whose RGB video also follows the pre-division path; they are affected as diagnostic representation values. They do not establish contamination of the formal endpoint TC-ME.

## 5. Independent sigma audit

The actual overlay UniPC scheduler source `wan/utils/fm_solvers_unipc.py` constructs the 40-step shift-3 schedule as `sigma = 3*u/(1+2*u)`, with `u=linspace(1,1/40,40)`, and `timestep=1000*sigma`. The complete schedule is in `pre_c_cpu_audit/pre_c_cpu_audit.json`.

Indices 14/15/16 are respectively `(847.8260869565, 0.8478260869565)`, `(833.3333333333, 0.8333333333)`, and `(818.1818181818, 0.8181818182)` for `(timestep,sigma)`. RW's `0.8333333333` independently comes from the canonical RW FlowMatch warp mapping, not an experiment hardcoded constant. `RW_WAN_SIGMA_ZERO_ERROR_INDEPENDENTLY_DERIVED = True`.

## 6. Frozen companion diagnostics

`PREVIEW_SIMILARITY_DIAGNOSTIC` is frozen as RGB L1 and PSNR between output Y and canonical preview P, separately for `FULL_FRAME` and `TRACK_OBJECT_SUPPORTED_REGION`, after only the frozen evaluation-only 480->464 mapping. `SHARPNESS_DIAGNOSTIC` is reserved as one mean spatial-gradient-energy definition. These are deterministic diagnostics, not perceptual quality or VBench. No generation artifact is resized before VAE.

## 7. Enabled-path pre-C sanity

The CPU audit verified the persisted preview artifact shape `[1,16,21,60,104]`, finite float32 values, overlay opt-in argument presence, static `set_begin_index(15)` reachability, source scheduler reset semantics, start index 15, and sigma `0.8333333333`. The intended formula is `x_start=(1-sigma)*preview_latent + sigma*epsilon`.

Final shared epsilon was deliberately not frozen before the mandatory A/B parity gate, and no scheduler runtime probe was executed. Hence `ENABLED_PATH_START_STATE_SANITY = UNRESOLVED_NO_FINAL_EPSILON_AND_NO_SCHEDULER_RUNTIME_PROBE`, not PASS.

## 8. Artifacts and provenance

Machine-readable audit: `pre_c_cpu_audit/pre_c_cpu_audit.json`. Audit script SHA-256: `9c512e10835b4c52681350993f85fc90424cc43c41b987fb99862f91cdb787ba` before the final status-only correction; current source checksum is discoverable alongside the script. Frozen evaluator SHA-256 `sha256` is recorded in JSON, as are the scheduler, overlay, preview-latent, and historical-diagnostic hashes. The complete 40-entry schedule is serialized in the JSON.

## 9. Gate decision

This CPU audit itself finds no formal-evaluator contamination hard stop. However, completed B fails the earlier authoritative disabled-path A/B decoded-RGB parity gate (`AB_PARITY_VALIDATION.json`): 81 decoded frames at 464x832 for both, but `DECODED_RGB_EXACT=False`, 55,284,397 differing channel values, maximum absolute pixel difference 214, mean 2.5112676238. Therefore the first hard failure remains `ADAPTER_DISABLED_PATH_PARITY=FAIL`. C0 reuse, epsilon freezing, C1/C2, and all C candidate metrics are prohibited and were not performed.
