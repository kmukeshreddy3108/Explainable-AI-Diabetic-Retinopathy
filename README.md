# Explainable AI for Diabetic Retinopathy Screening in Rural India

**SIH 2026 · PS 26038 · MathWorks Track**

---

## Overview

An end-to-end, offline-first, explainable AI platform designed for telemedicine diabetic retinopathy (DR) screening in rural primary healthcare centers (PHCs). The platform integrates image quality assessment, structure and lesion segmentation, hybrid ICDR severity grading, Grad-CAM attention maps, an offline store-and-forward outbox queue, specialist telemedicine feedback loops, and district-level workflow simulation.

---

## Completion Matrix across Core Problem Topics

| Topic | Description | Status | Implementation Details |
|---|---|---|---|
| **1. Image Quality Assessment & Enhancement** | Focus (Laplacian variance), illumination (exposure metrics), FOV coverage check, adaptive CLAHE & Ben Graham color normalization, recapture feedback | **100% Complete** | `src/iqa/gate.py`, `src/iqa/enhance.py`, `src/iqa/sharpness.py`, `src/iqa/exposure.py` |
| **2. Retinal Structure Segmentation** | Optic Disc & Fovea localization, Gabor vessel segmentation, Morphological lesion extraction (Microaneurysms, Hemorrhages, Exudates) | **100% Complete** | `src/segmentation/optic_disc.py`, `src/segmentation/vessels.py`, `src/segmentation/lesions.py` |
| **3. DR Severity Grading** | International Clinical DR scale (ICDR 0–4), hybrid deep learning + clinical rule fusion, referable DR sensitivity > 90%, specificity > 85% | **100% Complete** | `src/grading/model.py`, `src/grading/hybrid.py`, `src/grading/calibrate.py` |
| **4. Explainability Module** | Grad-CAM attention maps, spatial overlap scoring, temperature-calibrated confidence scores, automated diagnostic JSON reports (< 30s clinician review) | **100% Complete** | `src/explainability/gradcam.py`, `src/explainability/fusion.py`, `src/explainability/report.py` |
| **5. Offline Telemedicine & Specialist Loop** | Offline-first SQLite outbox queue, priority sync (Grade 2+ first), specialist referral response & feedback sync to PHC | **100% Complete** | `src/sync/outbox.py`, `src/sync/sync_worker.py`, `streamlit_app/pages/3_Specialist_Telemedicine_Portal.py` |
| **6. District Workflow Simulation** | Discrete-event capacity planning (100,000+ patients/year), camera & doctor bottleneck analysis, queue wait times | **100% Complete** | `src/simulation/district_sim.py`, `streamlit_app/pages/1_District_Simulation.py` |

---

## Offline Telemedicine & Specialist Feedback Architecture

```
 ┌─────────────────────────┐             ┌─────────────────────────┐
 │   RURAL PHC (OFFLINE)   │             │   DISTRICT HOSPITAL     │
 ├─────────────────────────┤             ├─────────────────────────┤
 │ 1. Local Camera Scan    │             │ 3. Ophthalmologist Portal│
 │ 2. Local AI Pipeline    │             │    - Reviews Grad-CAM   │
 │ 3. SQLite Outbox Queue  │  Batch Sync │    - Confirms Referral  │
 │    (Priority 2 for G2+) ├────────────►    - Enters Clinical Notes│
 └────────────▲────────────┘             └────────────┬────────────┘
              │                                       │
              └───────────────────────────────────────┘
                 4. Telemedicine Response Synced Back
                    to Rural PHC Database
```

1. **Local Execution at Rural PHC**: The entire pipeline (IQA $\rightarrow$ Segmentation $\rightarrow$ Hybrid Grading $\rightarrow$ Grad-CAM) runs 100% offline on a local laptop at the PHC.
2. **Prioritized Store-and-Forward Outbox**: Completed scans are stored in a local SQLite database (`data/outbox.db`). High-risk cases (Referable DR Grade 2+) get **Priority 2** (HIGH priority), while normal scans get Priority 1.
3. **Cloud/District Sync**: When cellular/internet connectivity becomes available, `SyncWorker` transmits the prioritized outbox items to the central district hospital repository.
4. **Specialist Referral & Feedback Loop**:
   - The nearby Ophthalmologist accesses the **Specialist Telemedicine Portal** (`pages/3_Specialist_Telemedicine_Portal.py`).
   - The specialist reviews the AI diagnosis, lesion breakdown, and Grad-CAM panel, then records a decision (`confirmed_referral`, `phc_rescreen`, or `recapture_requested`) along with treatment instructions.
   - The response is synchronized back to the rural PHC database.

---

## Quick Start

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run Streamlit Web Application
python -m streamlit run streamlit_app/app.py

# 3. GitHub Push Instructions
git remote add origin https://github.com/YOUR_USERNAME/Explainable-AI-for-Diabetic-Retinopathy-Screening-in-Rural-India.git
git branch -M main
git push -u origin main
```
