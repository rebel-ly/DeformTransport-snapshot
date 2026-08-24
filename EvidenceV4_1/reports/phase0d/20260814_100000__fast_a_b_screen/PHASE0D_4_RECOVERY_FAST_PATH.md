# Phase 0D-4 recovery — preview producer and matched-domain recovery

## B-R outcome

`infer_sim.py:load_sim_frames` is the authoritative producer: it lexically sorts `frames/frame_*.png`, converts each to RGB, resizes to 832×480, normalizes to [-1,1], and supplies the complete sequence to RealWonder SDEdit VAE. The original canonical run did not persist a separate preview output; it did persist the exact input frame sequence. Its stdout records 81 loaded frames and `sim_latent shape [1,21,16,60,104]`.

The frozen aligned build report and `build_aligned_transport_visibility_contract.py` prove `frame_0000 -> step0`, `frame_0001..0080 -> step10..800`, and explicitly discard the old step810 state. A deterministic reconstruction copied the exact 81 producer inputs with per-frame SHA256 to `preview_reconstruction_20260814/`; therefore `PREVIEW_TIMELINE_CONTRACT=PASS`.

The canonical run directory does not contain the needed raster/depth/frontmost material-index validity assets. Thus B-R5 validity coverage and hole-attribution statistics remain unresolved. The RealWonder schedule evidence is `[500,250]`, but no frozen noise/SNR mapping to Wan's 40-step shift=3.0 scheduler was recovered. No C arm manifests are prepared and `C_PREVIEW_SDEDIT_READY=False`.

## A-R outcome

The frozen formal evaluator contract resolves the common-domain transform: for 480×832 input it applies **bicubic resize to 464×832**, while x is preserved and track y is mapped as `y'=y*(464/480)`. This resolves `RW480_TO_COMMON464_TRANSFORM` for an eventual matched visual evaluator; temporal normalization would retain all 81 frames at a common fixed fps.

Official VBench remains unavailable: no valid local/shared checkout or weights were found, and the one isolated clone attempt has only an incomplete `.git` directory with no `HEAD`. Therefore `A_VBENCH_STATUS=BLOCKED_EXTERNAL_DEPENDENCY`; no substitute metric or scores were produced.

No Wan-Move generation, scheduler sweep, or video normalization was run.
