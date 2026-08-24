# FIG-Q1 final spatial/ROI contract

{
  "FIGQ1_COLUMNS": [
    0,
    20,
    40,
    60,
    80
  ],
  "FIGQ1_ROWS": [
    "PhysicalPreview",
    "C1",
    "C2",
    "CS",
    "RealWonder"
  ],
  "FIGQ1_UNIFIED_DISPLAY_DOMAIN": "832x464",
  "GENERATED_RAFT_FLOW_PANEL": false,
  "PREVIEW_RGB_GEOMETRY": "832x480",
  "PREVIEW_TO_464_RULE": "same formal evaluator/conditioning bicubic 480x832 -> 464x832 mapping; no crop",
  "ROI_A_FRAME": 60,
  "ROI_A_PADDED_INTEGER_BBOX": [
    395,
    55,
    695,
    191
  ],
  "ROI_A_RAW_BBOX": [
    420.8,
    67.0,
    669.9,
    179.6
  ],
  "ROI_A_SCORE": 1.0289578437805176,
  "ROI_A_SELECTION_RULE": "MAX_PHYSICS_ONLY_LOCAL_DEFORMATION_SCORE_AMONG_FIXED_DISPLAY_FRAMES",
  "ROI_B_AVAILABLE": true,
  "ROI_B_DISPLAY_FRAME": 20,
  "ROI_B_NUM_MATERIAL_IDS": 298,
  "ROI_B_PADDED_INTEGER_BBOX": [
    324,
    89,
    672,
    409
  ],
  "ROI_B_RAW_BBOX": [
    353.25286865234375,
    116.39028930664062,
    642.1470947265625,
    381.8840026855469
  ],
  "ROI_B_SELECTION_RULE": "cluster size desc, median duration desc, frame distance asc; visibility/trajectory only",
  "ROI_SELECTION_OUTCOME_BLIND": true,
  "RW_480_TO_464_EXACT_RULE": "torch.nn.functional.interpolate(size=(464,832), mode=bicubic, align_corners=False, antialias=False).clamp_(0,1)",
  "RW_480_TO_464_MODE": "BICUBIC_RESIZE",
  "RW_QUALITATIVE_ALIGNMENT_CONFIDENCE": "PASS",
  "source_knn_radius": 24.82009220123291
}
