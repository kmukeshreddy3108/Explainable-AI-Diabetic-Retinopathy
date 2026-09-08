# Explainable AI for Diabetic Retinopathy Screening

**SIH 2026 · PS 26038 · MathWorks Track**

---

## Two-Track Architecture

This project follows a **two-track** approach:

| Track | Purpose | Built With |
|---|---|---|
| **Track A — Demo Prototype** (this repo) | Fast, working, presentable demo for PPT/video and judge Q&A | Python + Streamlit + PyTorch → ONNX |
| **Track B — MathWorks Submission** (separate) | Scored deliverable: IQA, segmentation, ONNX import, Grad-CAM, Simulink district model | MATLAB (Image Processing / Deep Learning / Simulink + SimEvents Toolboxes) |

Both tracks share the **same trained model** (PyTorch → ONNX), so no work is wasted:
- Train once in PyTorch, export via `torch.onnx.export`
- Track A loads the `.onnx` file via `onnxruntime`
- Track B loads it via `importNetworkFromONNX` in MATLAB

> **Important**: The Module 5 district simulation in this repo uses **SimPy** (Python discrete-event simulation) as a lightweight demo stand-in. It is **not** a substitute for the real **Simulink/SimEvents** model required by the MathWorks problem statement.

---

## Pipeline Modules

| Module | Description | Key Techniques |
|---|---|---|
| **1 — IQA Gate** | Image quality assessment & enhancement | Laplacian variance, green-channel Otsu FOV, CLAHE, homomorphic filter |
| **2 — Segmentation** | Retinal structure & lesion segmentation | Gabor filter vessels, Hough transform OD, morphological top-hat MA/exudates |
| **3 — Grading** | ICDR severity grading (0–4) | EfficientNet-B0 (timm), temperature-calibrated confidence, ONNX export |
| **4 — Explainability** | Grad-CAM attention maps + lesion fusion | pytorch-grad-cam, spatial overlap scoring |
| **4.5 — Offline Sync** | Store-and-forward for intermittent connectivity | SQLite outbox queue, priority sync (Grade 2+ first) |
| **5 — Simulation** | District workflow demo | SimPy discrete-event model (demo stand-in for Simulink) |

---

## Dataset Roles (Hard Constraints)

| Dataset | Role | Rule |
|---|---|---|
| **APTOS 2019** | Training + internal validation | Stratified split, used for model development |
| **IDRiD** | Training + internal validation | Combined with APTOS for training |
| **Messidor-2** | External hold-out evaluation **only** | Never touched until final evaluation |
| **DRIVE** | Vessel segmentation benchmark **only** | Used only to score the vessel sub-module (Dice/Jaccard) |

---

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit demo
streamlit run streamlit_app/app.py

# Run tests
python -m pytest tests/ -v
```

---

## Project Structure

```
sih_statement/
├── src/
│   ├── iqa/              # Module 1: Image Quality Assessment
│   ├── segmentation/     # Module 2: Retinal Structure & Lesion Segmentation
│   ├── grading/          # Module 3: ICDR Severity Grading
│   ├── explainability/   # Module 4: Grad-CAM & Fusion
│   ├── simulation/       # Module 5: SimPy District Workflow (demo)
│   ├── sync/             # Phase 4.5: Offline-first Store & Forward
│   └── utils/            # Shared utilities, config, sample generator
├── models/               # Trained .pt and exported .onnx model artifacts
├── streamlit_app/        # Streamlit UI (Triage Desk, Simulation, Benchmarking)
├── data/                 # Dataset samples & DRIVE benchmark masks
├── notebooks/            # Exploratory training & evaluation notebooks
└── tests/                # pytest test suite
```

---

## Framing Notice

All lesion detection, Grad-CAM attention maps, and severity grades produced by this system are **decision-support evidence for a qualified human reviewer** — never autonomous diagnoses. The clinician always makes the final grading decision.
