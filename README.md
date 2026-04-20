# DIRD Models

Computer vision models for pathological retina finding detection, focused on **Diabetic Retinopathy (DR)**.

DIRD (Diabetic & Intelligent Retinal Detection) ships pretrained YOLO models plus metadata JSONs describing classes, clinical semantics, performance, and visualization palettes. Models are exported to ONNX for browser / edge inference via `onnxruntime-web`.

---

## Contents

| File | Purpose |
|------|---------|
| `detection-v1.0.1.json` | Detection model metadata (YOLOv11, 11 classes) |
| `detection-v1.0.1.onnx` | Detection weights (opset 14, browser-ready) |
| `MODEL_CONVERTER.py` | ONNX → browser-compatible ONNX converter (opset downgrade + simplify) |

---

## Detection Model — `DIRDv1r1`

- **Architecture:** YOLOv11 (Detection)
- **Input:** `640 × 640`
- **Trained:** 2024-12-15
- **Classes:** 11 (7 currently detected, 4 reserved)

### Classes

| Idx | Technical | Category | Severity | Active |
|----:|-----------|----------|----------|:------:|
| 0 | `optic_disc` | anatomical_landmark | none | ✅ |
| 1 | `hard_exudate` | lesion | moderate | ✅ |
| 2 | `fovea` | anatomical_landmark | none | ✅ |
| 3 | `hemorrhage` | lesion | moderate_to_severe | ✅ |
| 4 | `cotton_wool_spot` | lesion | moderate_to_severe | ✅ |
| 5 | `microhemorrhages` | lesion | mild_to_moderate | ✅ |
| 6 | `edema` | lesion | severe | ✅ |
| 7 | `microaneurysm` | lesion | mild | ⏸ |
| 8 | `neovascularization` | lesion | severe | ⏸ |
| 9 | `venous_beading` | lesion | severe | ⏸ |
| 10 | `irma` | lesion | severe | ⏸ |

Each class includes bilingual display names (EN/ES), description, aliases, and color palette (hex). See `detection-v1.0.1.json` for the full schema.

### Performance

Global:
- `mAP50`: **0.826**
- `mAP50-95`: **0.450**
- Precision: **0.82** / Recall: **0.79**

Per-class `mAP50`:

| Class | mAP50 |
|-------|------:|
| optic_disc | 0.984 |
| fovea | 0.957 |
| edema | 0.995 * |
| cotton_wool_spot | 0.830 |
| hard_exudate | 0.781 |
| hemorrhage | 0.698 |
| microhemorrhages | 0.538 |

\* edema score is inflated due to low validation support (severe class imbalance).

### Status: `REQUIRES_IMPROVEMENT`

Known issues & next steps (from the metadata's `analysis_report`):
- Small-object underdetection → raise input size to `1024²` or `1280²`.
- Edema underrepresented → collect more samples or oversample.
- Loss still descending at epoch 100 → extend training to 150–200 epochs.

---

## Segmentation Model — *in development*

Segmentation model for retinal structures (blood vessels, microaneurysms, hard/soft exudates, hemorrhages, neovascularization) is under active development and **not included in this release**. Will ship in a future version once validated.

---

## ONNX Browser Converter

`MODEL_CONVERTER.py` downgrades an ONNX model to a target opset (default **14**) and runs `onnxsim` simplification so it loads under `onnxruntime-web`.

### Install

```bash
pip install onnx onnxruntime onnxsim
```

### Usage

```bash
python MODEL_CONVERTER.py input.onnx output.onnx [opset_version]
```

Example:

```bash
python MODEL_CONVERTER.py detection-v1.0.0.onnx detection-v1.0.0-web.onnx 14
```

The script validates the model, simplifies it, and prints an input/output size comparison.

---

## Metadata JSON schema (detection)

```jsonc
{
  "model_info":          { "version", "type", "date_trained", "input_size" },
  "classes":             [ { "index", "technical_name", "display_name_en", "display_name_es",
                             "category", "severity_impact", "description_en", "description_es",
                             "currently_detected", "aliases"? } ],
  "class_groups":        { "anatomical_landmarks", "early_stage_lesions",
                           "moderate_lesions", "severe_lesions" },
  "performance_metrics": { "global", "per_class_mAP50" },
  "color_palette":       { "<technical_name>": "#hex" },
  "analysis_report":     { "status", "critical_findings", "recommendations_next_steps" }
}
```

---

## Intended use

Research and assistive tooling for retinal image analysis. **Not a medical device.** Do not use outputs for diagnosis without review by a qualified ophthalmologist.
