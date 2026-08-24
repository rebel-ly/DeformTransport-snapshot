# Santa + Tree full-system comparison readiness audit

Scope: CPU/read-only audit only. No Wan-Move generation, GPU evaluator, RAFT inference, scientific metric, model download, package installation, or frozen-artifact modification was performed.

## Authoritative inventories

### Santa

- RealWonder: `server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260806_235302__aligned_baseline_vs_balanced_ramp4_full_generation/baseline/aligned_santa_baseline_seed0.mp4`. This is the canonical 81-frame baseline for the exact aligned Santa input. Its launch command names the same `official_santa_81f_aligned_final_sim_20260806_234410`, checkpoint `step=000800.pt`, seed 0, degradation 0.5, and local-attention size 21. Later Santa transport-region, RAFT, condition-space, and utility-selector artifacts all reference it. The earlier `REALWONDER_SANTA_BASELINE_20260805_032928` artifact is only a 21-frame smoke and is not the system-comparison baseline.
- Wan-Move Correct and Identity-Shuffled: the two frozen videos in `server_runs/wan_move_formal/20260809_195255__santa_correct_vs_identity_shuffled_seed0`. Both are 81 frames at 832x464, fps metadata 16, decode completely, and are tied to explicit `run.sh`, formal input hashes, seed 0, and the frozen Wan-Move RNG patch.
- Geometry-aligned simulation proxy: `prepared_inputs/official_santa_81f_aligned_final_sim_20260806_234410/simulation.mp4`, 81 frames at 832x480. It is a geometry-aligned simulation proxy, not future-RGB ground truth.
- Reference flows: the sibling `flows.npy`, shape `[80,2,240,416]`, float16. The noise-generation report records 832x480 RGB -> area resize to 416x240 -> torchvision RAFT-Large C_T_SKHT_V2.
- Raw physics raster flow also exists at `OFFICIAL_SANTA_81F_CHAIN_20260805_050719/simulation_source/flows.npy`, shape `[81,2,512,512]` float32. It is in the raw renderer domain, not the frozen RAFT reference; no direct generated-video comparison contract is currently frozen for it.
- Persistent tracks/visibility: `wan_move_bridge/20260809_010015__santa_correct_tracks/{santa_material_tracks_correct.npy,santa_material_visibility_correct.npy}`, shapes `[1,81,1277,2]` float32 and `[1,81,1277]` bool. Coordinates are the aligned 832x480 pixel domain.
- Historical local support exists in `quality_ramp4_full_generation.pt::transport_mask` (`[21,1,60,104]`). It belongs to the historical RealWonder latent-transport protocol; it is available as a candidate but is not automatically the correct local support for a Wan-Move-vs-RealWonder system comparison.

### Tree

- RealWonder: `server_runs/20260804_234925_autonomous_deformtransport/12_soft_transport_dev/20260807_203228__tree__realwonder_baseline_seed0/tree_realwonder_baseline_seed0.mp4`. The authoritative Tree bridge report explicitly records this path as `realwonder_baseline_record_only`, and Tree comparison audits repeatedly use it. It is 81 frames at 832x480, fps metadata 10, and decodes completely. Provenance limitation: this run directory retained stdout/stderr, times, Git head/status, and the video, but not the literal launch command. The logs prove use of the exact aligned Tree final_sim, its prompt/noises/source image, and the frozen Tree generation lineage; the precise command is less completely archived than Santa's.
- Wan-Move Correct and Identity-Shuffled: the two frozen videos in `server_runs/wan_move_formal/20260810_073902__tree_correct_vs_identity_shuffled_seed0`. Both are 81 frames at 832x464, fps metadata 16, decode completely, and have explicit run scripts and runtime/input provenance.
- Geometry-aligned simulation proxy: `prepared_inputs/tree_official_precomputed_aligned_final_sim_20260807_185055/simulation.mp4`, 81 frames at 832x480. It is a geometry-aligned simulation proxy, not future-RGB ground truth.
- Reference flows: the sibling `flows.npy`, shape `[80,2,240,416]`, float16, created under the same 832x480 -> area 416x240 -> RAFT-Large C_T_SKHT_V2 contract.
- Raw physics raster flow also exists at `20260807_175657__tree__official_precomputed__80future_81aligned/flows.npy`, shape `[80,2,512,512]` float32, with source-point raster indices and post-hoc physics validation. It is authoritative physics output but is not the 416x240 RAFT reference.
- Persistent tracks/visibility: `wan_move_bridge/20260810_072215__tree_correct_tracks/{tree_material_tracks_correct.npy,tree_material_visibility_correct.npy}`, shapes `[1,81,713,2]` float32 and `[1,81,713]` bool. Coordinates are the aligned 832x480 pixel domain; visibility is frontmost-raster visibility intersected with projection/crop validity.
- Historical local support exists in `tree_quality_ramp4_full_generation.pt::transport_mask` (`[21,1,60,104]`) and was used by the historical Tree local RGB report. It has the same limitation as the Santa historical mask.

All paths, SHA256 values, byte sizes, shapes, fps metadata, and decode results are in `inventory.json` and `input_sha256.txt`.

## Spatial alignment contract

The 480 -> 464 change is an anisotropic resize, not a 16-row crop.

1. The aligned RealWonder preprocessing first maps raw 512x512 RGB to 832x832 with PIL bilinear and center-crops rows `[176,656)` to 832x480 (`deform_transport/transport_payloads.py:32-41`). Persistent coordinates therefore start in this aligned 832x480 domain: `x=832*u/512`, `y=832*v/512-176`.
2. Formal Wan-Move uses `--size 480*832` but the non-`eval_bench` source path computes latent height with chained floating-point floor operations (`Wan-Move/wan/wan_move.py:200-214`). For this aspect ratio, `sqrt(max_area*480/832)` is represented just below 480; the chained `//8//2*2` yields latent height 58, hence decoded height `58*8=464`. Width remains 832.
3. Wan-Move then resizes the conditioning tensor to `(464,832)` with `torch.nn.functional.interpolate(..., mode='bicubic')` and scales tracks by `scale_h=464/480`, `scale_w=1` (`wan_move.py:216-220,266-274`). With current PyTorch defaults this is `align_corners=False`, `antialias=False`.
4. Therefore the common generated domain should be 832x464. Keep Wan-Move unchanged. Convert each 832x480 RealWonder/proxy RGB frame as: decode BGR -> RGB uint8 -> float32 `[0,1]` NCHW -> `F.interpolate(size=(464,832), mode='bicubic', align_corners=False, antialias=False)` -> clamp to `[0,1]` -> retain RGB for RGB/Lab metrics. Do not center-crop.
5. Coordinate mapping is exactly `x_464=x_480`, `y_464=y_480*(464/480)`. The frozen TC-MAR evaluators already use this coordinate mapping.
6. The existing reference flows are not full-resolution 480-row arrays. They are already `[80,2,240,416]` RAFT flows inferred after area-downsampling the original aligned 832x480 frames, in 416x240-grid pixel units. They must not be geometrically cropped or have their vectors rescaled merely because displayed RGB is standardized to 464. For a comparable generated-video flow, first map the generated 832x464 RGB back through a declared reconciliation route, then apply the same 416x240 RAFT input contract. The source-faithful candidate is direct area resize 464x832 -> 240x416; this produces flows in the same final grid units, but it is not pixel-identical preprocessing to reference frames (which start at 480 rows). The stricter alternative is first bicubic 464 -> 480, then historical area 480 -> 240. This unresolved choice needs pre-registration because it changes the image presented to RAFT.

## TC-MAR applicability to RealWonder

Conceptually and technically valid. TC-MAR asks whether the source material appearance at `P_i(0)` is retained at the true future material location `P_i(t)`. It does not require the evaluated video to have been conditioned on the tracks. RealWonder can therefore be evaluated using exactly the same true Correct tracks and visibility as the two Wan-Move arms.

Minimum wrapper adaptation only:

- add one immutable RealWonder video input and expected SHA;
- decode its 81 RGB frames, assert 832x480, and apply the frozen 480 -> 464 generated-domain bicubic transform above before patch sampling;
- use the same source image, 8x8 offsets `-3.5..+3.5`, exact float32 bilinear sampler, anchors 4,8,...,80, OpenCV float RGB->Lab conversion, and per-track aggregation;
- use the case's already-frozen visibility rule: Santa complete-case visibility at all anchors; Tree per-anchor true visibility with per-track mean over available anchors and at least one valid anchor;
- use exactly the same observation set for RealWonder, Correct, and Shuffled. No method-specific validity filtering.

No new scientific metric is needed. The wrapper must not compare the 480-row RealWonder frame at unscaled coordinates against 464-row Wan-Move frames.

## Appearance proxy evaluator candidates

The closest historical protocols are:

1. Global RGB MAE/L1 and MSE/PSNR against the geometry-aligned simulation proxy. Historical source: `server_runs/20260804_234925_autonomous_deformtransport/12_scaled_evaluation/evaluate_video_comparison.py` (PSNR/SSIM/temporal difference) and Tree four-way report `20260807_220851__tree__four_way_quick_metrics/report.json` (MAE/MSE/PSNR/temporal diagnostic). Preprocess all methods to RGB float in the common 832x464 domain. No mask. CPU cost is low-to-moderate (streaming is seconds to a few minutes; the current all-at-once implementation needs roughly gigabyte-scale working memory). Minimal adaptation: common-domain resizer, three-method input table, and explicit proxy wording.
2. Local RGB L1/MSE/PSNR on existing transport support. Historical Santa source: `.../transport_region_evaluation/evaluate_transport_region.py`; historical Tree record: `20260807_221342__tree__quality_transport_region_local_metrics/report.json`. Historical masks are nearest-upsampled latent `transport_mask`; future-only anchor evaluation is already used for Tree. RGB order is RGB; values are uint8/float32 0..255. CPU cost is low. Minimal adaptation: freeze one existing support per case, resize it with nearest-neighbor into 832x464, and apply the same mask to all methods. This candidate remains blocked on the support-definition PI choice below.
3. Temporal-change L1 and spatial-gradient error against the geometry-aligned simulation proxy are available as secondary diagnostics in the historical Santa local evaluator. They are appropriate only as secondary proxy-alignment diagnostics, not as standalone video quality or future fidelity. CPU cost is low; minimal adaptation is the common-domain transform and streaming accumulation.
4. Gaussian-window SSIM exists historically, but the request's nearest authoritative set is global MAE/local L1/PSNR. SSIM may be retained as secondary only; it is not needed to establish readiness.

All of the above compare with a geometry-aligned or coarse simulation proxy. None may call that proxy GT.

## RAFT motion contract candidates

Common historical facts:

- Videos/PNG frames are decoded as RGB uint8.
- Frames are converted to float NCHW and area-resized to `[240,416]`.
- Values are normalized to `[0,1]`, then passed through `Raft_Large_Weights.C_T_SKHT_V2.transforms()` and torchvision `raft_large(...)[-1]`.
- Stored reference flow is `[T-1,2,240,416]`, and predicted/reference vectors are in the same 416x240 grid-pixel units.
- EPE is `sqrt((du)^2+(dv)^2)`. Magnitude error is `abs(||pred||-||ref||)`. Angular error is `degrees(acos(clamp(dot/(||pred||||ref||+1e-6),-1,1)))`; historical direction masks additionally require reference magnitude >0.25 and predicted magnitude >0.05.
- Historical local support is a causal transition mask derived from latent transport support, nearest-upsampled to 240x416 and combined across adjacent frames. Global and local/moving-support results should be reported separately.
- Santa's early `raft_motion_queued/evaluate.py`, later `evaluate_raft_motion.py`, and authoritative SandHouse `evaluate_sandhouse_raft.py` agree on area 480x832 -> 240x416 and RAFT-Large C_T_SKHT_V2. The later evaluators add official transforms, magnitude error, angular error, and explicit support.

Candidate reconciliation protocols for the new 464-row Wan-Move videos:

- Candidate A (fewest transforms): RGB 832x464 -> area 416x240 -> official RAFT transforms. Reference flow remains unchanged. This makes both predicted and reference vectors live on the same 416x240 grid and is the simplest extension of the historical endpoint contract.
- Candidate B (strict historical input-height emulation): RGB 832x464 -> bicubic 832x480 using the declared inverse-size operation -> historical area 416x240 -> official transforms. Reference flow remains unchanged. This better matches the reference flow's two-stage image-size lineage but invents 16 rows by interpolation.

Do not resize the stored `[240,416]` flow field to 232 rows, crop it, or multiply its y component by 464/480 if the final evaluation grid remains `[240,416]`; doing so changes the vector units and breaks comparison with the archived reference. The A-vs-B preprocessing choice and the local-support definition are scientific ambiguities requiring PI freeze before a GPU run.

## Standard video-quality metric availability and limitations

Repository search found no LPIPS, FVD, DINO/DINOv2, or other no-reference video-quality implementation. No model was downloaded and no package was inspected or installed. Existing implemented metrics are RGB MAE/L1, MSE/PSNR, Gaussian-window SSIM, temporal-difference L1, spatial-gradient error, and RAFT motion diagnostics.

- PSNR, SSIM, LPIPS, DINO feature distance, and conventional full-reference fidelity require a defensible real future RGB reference. The current simulation RGB is only a geometry-aligned/coarse simulation proxy, so these cannot be primary real-future fidelity metrics.
- FVD is distributional and normally requires a sufficiently large real/reference video set; two single-case simulation proxies are not such a set.
- No-reference quality measures, if later added, could assess generic visual quality but not correctness of the intended material motion or real-future fidelity.

## Recommended frozen inputs

Freeze the 18 SHA-pinned artifacts listed in `input_sha256.txt`: three method videos, aligned simulation proxy, RAFT reference flow, raw physics raster flow, Correct tracks, visibility, and the historical support candidate for each case. For primary full-system comparisons use:

- Correct vs Identity-Shuffled for mechanism contribution.
- Correct vs RealWonder for complete-system performance.
- TC-MAR on all three methods with one shared per-case observation set.
- Appearance and RAFT only as proxy/motion diagnostics unless a real future RGB reference is supplied.

## PI decisions required before evaluation

1. RAFT 464->240 preprocessing: Candidate A direct area downsample or Candidate B bicubic restore to 480 then historical area downsample.
2. Local support: historical RealWonder `transport_mask`, a pre-existing object support, or no local metric. Reusing the historical masks is minimal and authoritative for old local protocols, but those masks are not intrinsically the same as the new Wan-Move selected-track support.
3. Frame-time interpretation: artifacts are frame-index aligned at 81 states, but container fps metadata differ (Santa proxy 8, Tree proxy/RealWonder 10, Wan-Move 16). Freeze frame-index transitions, as historical TC-MAR/RAFT do, or define an explicit physical-time resampling. Do not silently use playback timestamps.
4. Tree RealWonder provenance: accept the bridge-canonical artifact despite the missing archived literal command, or require a provenance waiver. Regeneration is outside this audit and was not performed.
