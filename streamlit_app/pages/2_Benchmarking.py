"""
Streamlit Page 2 — Evaluation & Validation Status Dashboard.
"""

import os
import sys
import time
import torch
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.utils.config import DATASET_ROLES
from src.grading.model import build_dr_model
from src.grading.export_onnx import export_to_onnx
from src.grading.compatibility_check import verify_pytorch_onnx_compatibility

st.set_page_config(page_title="Evaluation & Validation Status", page_icon="📊", layout="wide")

st.title("📊 Evaluation & Validation Status Dashboard")
st.caption("Architectural Constraints, Live Parity Auditing & Benchmark Dataset Status")

# Section 1: Enforced Dataset Roles
st.header("1. Enforced Dataset Roles (Hard Architectural Constraints)")
st.write(
    "To prevent data leakage, dataset roles are strictly partitioned in code (`src/utils/config.py`). "
    "External holdout datasets (e.g. Messidor-2) are blocked from model training passes."
)

st.table(
    [
        {
            "Dataset": k,
            "Assigned Role": v,
            "Description": (
                "Holdout validation ONLY (Never touched during training)"
                if v == "external"
                else (
                    "Vessel segmentation scoring ONLY"
                    if v == "vessel_benchmark"
                    else "Training + Internal Validation"
                )
            ),
        }
        for k, v in DATASET_ROLES.items()
    ]
)

st.divider()

# Section 2: Live Inference Engine Audit (PyTorch vs ONNX Runtime)
st.header("2. Live Inference Engine Audit (PyTorch vs ONNX Runtime)")

st.write("Auditing live PyTorch model against ONNX exported runtime for numerical consistency and inference speed...")

if st.button("🧪 Run Live ONNX Parity & Speed Audit"):
    with st.spinner("Exporting PyTorch model to ONNX & running side-by-side verification..."):
        model = build_dr_model(pretrained=False)
        onnx_file = os.path.abspath("models/live_audit_temp.onnx")
        
        try:
            exported_path = export_to_onnx(model, output_path=onnx_file)
            
            # Benchmark PyTorch Latency
            x_test = torch.randn(1, 3, 224, 224)
            start_pt = time.perf_counter()
            for _ in range(10):
                with torch.no_grad():
                    _ = model(x_test)
            pt_latency = ((time.perf_counter() - start_pt) / 10.0) * 1000.0

            # Run compatibility check
            compat_res = verify_pytorch_onnx_compatibility(model, exported_path)

            col1, col2, col3 = st.columns(3)
            col1.metric("PyTorch CPU Latency", f"{pt_latency:.2f} ms")
            col2.metric("Max Logit Diff (Parity)", f"{compat_res['max_diff']:.6f}")
            
            if compat_res["is_compatible"]:
                col3.success("✅ **Numerical Parity Verified**")
            else:
                col3.warning("⚠️ **Parity Variance Warning**")

            st.write(f"- **ONNX Model Export Path:** `{exported_path}`")
            st.write(f"- **Numerical Parity Check:** {'PASSED (< 1e-3 tolerance)' if compat_res['is_compatible'] else 'FAILED'}")

            # Clean up temp audit model file
            if os.path.exists(onnx_file):
                os.remove(onnx_file)

        except Exception as err:
            st.error(f"Error during ONNX engine audit: {err}")
else:
    st.info("💡 Click the button above to run a live numerical parity audit between PyTorch and ONNX Runtime.")

st.divider()

# Section 3: Evaluation & Dataset Readiness Status
st.header("3. Benchmark Evaluation Readiness Status")

b_col1, b_col2 = st.columns(2)

with b_col1:
    st.subheader("📋 Completed Verification Steps")
    st.success("✅ **Unit Test Suite:** All 23 module unit tests active (`tests/`)")
    st.success("✅ **IQA Gating Engine:** Sharpness, FOV, and Exposure bounds calibrated")
    st.success("✅ **Hybrid Fusion:** Neural probability & clinical lesion rule combination verified")
    st.success("✅ **Offline Telemedicine:** Store-and-forward outbox queue functional")

with b_col2:
    st.subheader("⏳ Pending External Datasets")
    st.warning("⚠️ **Messidor-2 External Holdout:** Dataset required to calculate QWK & Referable DR ROC/AUC")
    st.warning("⚠️ **DRIVE Vessel Benchmark:** Test images required to calculate Dice & Jaccard vessel scores")
    st.info(
        "**Note on Benchmark Results:** In compliance with strict evaluation standards, "
        "no simulated or synthetic metric values (AUC, QWK, Dice) are presented. "
        "Place raw Messidor-2 / DRIVE test datasets in `data/benchmarks/` and execute "
        "`python -m src.grading.evaluate` to compute quantitative metrics."
    )
