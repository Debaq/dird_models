# Model Card — detection-v2.0.0 (DIRDv2r0)

TRIPOD+AI-structured **development** documentation for the DIRD+ diabetic-retinopathy
detection model. Companion to the **evaluation** record in the DIRD+ repo
(`Dird/validation/`, experiments 1–5) and the cross-repo checklist
`Dird/validation/TRIPOD-AI-mapping.md`.

> Fields marked **⚠️ TODO** are not recoverable from any local repo (the model was trained
> outside the tracked repos). Only the person who ran training holds them. Fill them in to
> reach TRIPOD+AI completeness for the development items.

---

## 1. Model summary
| | |
|---|---|
| Version | DIRDv2r0 (`detection-v2.0.0`) |
| Architecture | YOLOv26s — object detection, end-to-end NMS |
| Framework | ultralytics (JSON notes "YOLOv26n" — **⚠️ reconcile n vs s**) |
| Input | 640×640 RGB fundus |
| Output | tensor `[1, 300, 6]` = `[x1,y1,x2,y2,score,class_idx]`, NMS by model, max 300 det |
| ONNX opset | 14 (browser-compatible, via `MODEL_CONVERTER.py`) |
| Default confidence | 0.25 |
| Date trained | 2026-04-21 |
| License | GNU AGPLv3 |
| Status | REQUIRES_IMPROVEMENT |

## 2. Intended use
- **Use:** research / screening **assist** for diabetic retinopathy on color fundus images.
- **Not** a medical device; **not** for autonomous diagnosis. Output is lesion detection
  feeding a binary normal-vs-pathological screen.
- Deployed downstream as binary screen with per-class calibrated thresholds (PCT_fpr02) —
  see `Dird/validation/experiment-2-aptos`.
- **⚠️ TODO:** explicit intended population, care setting, and contraindications.

## 3. Classes (6 active of 11 defined)
| idx | class | category | severity | detected |
|---|---|---|---|---|
| 0 | optic_disc | landmark | none | ✅ |
| 1 | hard_exudate | lesion | moderate | ✅ |
| 2 | fovea | landmark | none | ✅ |
| 3 | hemorrhage | lesion | moderate–severe | ✅ |
| 4 | cotton_wool_spot | lesion | moderate–severe | ✅ |
| 5 | microhemorrhages | lesion | mild–moderate | ✅ (microaneurysm merged here) |
| 6–10 | microaneurysm, edema, neovascularization, venous_beading, irma | lesion | — | ❌ reserved |

## 4. Training data — ⚠️ TODO (not on disk)
| TRIPOD+AI item | Value |
|---|---|
| Data source(s) | **⚠️ TODO** — likely IDRiD + others (bbox derived from masks); confirm exact datasets |
| Sample size (train/val/test) | **⚠️ TODO** |
| Class distribution / imbalance | **⚠️ TODO** (hemorrhage known under-represented) |
| Split strategy (patient-level? leakage control?) | **⚠️ TODO** |
| Inclusion / exclusion / gradability | **⚠️ TODO** |
| Annotation method | masks → bbox via connected components |
| Demographics (age/sex/ethnicity/cameras) | **⚠️ TODO** |
| Preprocessing / augmentation chain | **⚠️ TODO** |

## 5. Training configuration — ⚠️ TODO
| Item | Value |
|---|---|
| Epochs | **⚠️ TODO** |
| Batch size | **⚠️ TODO** |
| Optimizer / LR / schedule | **⚠️ TODO** |
| Augmentations | **⚠️ TODO** |
| Random seed | **⚠️ TODO** |
| ultralytics version | **⚠️ TODO** (pin for reproducibility) |
| Hardware / training time | **⚠️ TODO** |

## 6. Internal performance (held-out, from metadata JSON)
Global: mAP50 **0.5776** · mAP50-95 **0.3246** · precision 0.6101 · recall 0.5762

| class | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| optic_disc | 0.980 | 1.000 | 0.995 | 0.817 |
| fovea | 0.707 | 0.833 | 0.855 | 0.472 |
| cotton_wool_spot | 0.538 | 0.621 | 0.590 | 0.288 |
| microhemorrhages | 0.604 | 0.459 | 0.502 | 0.200 |
| hard_exudate | 0.481 | 0.379 | 0.364 | 0.113 |
| hemorrhage | 0.350 | 0.165 | 0.161 | 0.058 |

## 7. External validation (cross-repo: `Dird/validation/`)
| Site | n | AUC | Notes |
|---|---|---|---|
| APTOS (India) | 3662 | 0.949 | in-domain baseline (exp-2) |
| Messidor (France) | 1057 | 0.793 | confounded by preprocessed mirror (exp-3) |
| DDR (China) | 12522 | 0.817 | clean external; ΔMCC≈0 threshold transport (exp-4) |

Threshold transportability confirmed (frozen ≈ refit τ). Bootstrap 95% CI available.

## 8. Subgroup / fairness findings (exp-5)
- **Severity:** mild-DR (grade 1) sensitivity 0.99 in-domain → **0.44 / 0.51** external. Model
  reliable for referable DR (grade ≥2), weak on early disease OOD.
- **Lesion channels:** `hemorrhage` weakest everywhere (AUC 0.56–0.64); `microhemorrhages` strongest.
- **Not assessable** (no metadata): camera/device, hospital, age, sex.

## 9. Known limitations
- microhemorrhages and microaneurysm merged into one class (index 5).
- hemorrhage recall low (0.17) — class imbalance + mask→bbox mismatch at IoU 0.5.
- Detector-level mAP modest; deployment is at image-level binary screen, not per-lesion.
- Mild-DR sensitivity drops out-of-distribution (see §8).

## 10. Recommended next version
- Retrain at 1024×1024 for small-lesion recovery.
- Balanced hemorrhage sampling.
- Separate microaneurysm vs microhemorrhage annotations.

## 11. Open science — ⚠️ TODO
Funding, conflicts of interest, protocol/registration: **⚠️ TODO**.
Code: validation scripts in `Dird/validation/`. Weights: this repo (Zenodo DOI in README).
