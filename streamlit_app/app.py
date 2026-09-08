"""
Streamlit Web Application — DR Screening Triage Desk (Track A Prototype)

SIH 2026 · PS 26038 · MathWorks Track
Explainable AI for Diabetic Retinopathy Screening
"""

import os
import sys
import json
import numpy as np
import cv2
import torch
import streamlit as st
import matplotlib.pyplot as plt

# Ensure root directory is on Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.utils.sample_generator import generate_synthetic_fundus
from src.utils.config import ICDR_LABELS, ICDR_DESCRIPTIONS
from src.iqa.gate import assess_and_gate
from src.iqa.field_of_view import extract_fov_mask
from src.segmentation.optic_disc import detect_optic_disc, estimate_fovea_location
from src.segmentation.vessels import segment_vessels
from src.segmentation.lesions import detect_all_lesions
from src.grading.model import build_dr_model
from src.grading.hybrid import fuse_hybrid_grading
from src.explainability.gradcam import GradCAM, overlay_heatmap
from src.explainability.fusion import create_composite_explanation_panel
from src.explainability.report import generate_diagnostic_report
from src.sync.outbox import OutboxQueue
from src.sync.sync_worker import SyncWorker

# Page Configuration
st.set_page_config(
    page_title="DR Triage Desk — AI Screening",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("👁️ Intelligent Diabetic Retinopathy Triage & Screening Desk")
st.caption("SIH 2026 · PS 26038 · MathWorks Track | Multi-Stage Explainable AI Pipeline")

# Initialize Session & Model
@st.cache_resource
def load_grading_model():
    return build_dr_model(pretrained=True)

model = load_grading_model()
outbox = OutboxQueue()
sync_worker = SyncWorker(outbox=outbox)

# ── Sidebar Controls ────────────────────────────────────────────────────────
st.sidebar.header("📥 Input Image Selection")
input_mode = st.sidebar.radio("Image Source", ["Synthetic Fundus Generator", "Upload Fundus PDF/Image"])

if input_mode == "Synthetic Fundus Generator":
    sample_grade = st.sidebar.select_slider(
        "Target ICDR Severity Grade",
        options=[0, 1, 2, 3, 4],
        format_func=lambda g: f"Grade {g}: {ICDR_LABELS[g]}",
        value=2,
    )
    sample_quality = st.sidebar.selectbox(
        "Quality Impairment",
        ["good", "blurry", "dark", "bright", "glare", "poor_fov"],
        index=0,
    )
    sample_seed = st.sidebar.number_input("Random Seed", value=42, step=1)

    bgr = generate_synthetic_fundus(grade=sample_grade, quality=sample_quality, seed=sample_seed)
    img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    patient_id = f"PATIENT_{sample_seed:04d}"
    scan_id = f"SCAN_{sample_grade}_{sample_quality}_{sample_seed}"
else:
    uploaded_file = st.sidebar.file_uploader("Upload Retinal Fundus (JPG, PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        # Cap resolution at 1024px max dimension for processing speed
        max_dim = max(bgr.shape[:2])
        if max_dim > 1024:
            scale = 1024 / max_dim
            bgr = cv2.resize(bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        patient_id = "PATIENT_UPLOAD"
        scan_id = f"SCAN_{uploaded_file.name}"
    else:
        st.info("👈 Please select synthetic parameters or upload a fundus image in the sidebar.")
        st.stop()

# Display Original Image in Sidebar
st.sidebar.image(img_rgb, caption=f"Scan ID: {scan_id}", width="stretch")

# ── Step 1: Module 1 — IQA Quality Assessment ──────────────────────────────
st.header("1. Image Quality Assessment (IQA Gate)")

iqa_result = assess_and_gate(img_rgb)
q_col1, q_col2, q_col3, q_col4 = st.columns(4)

with q_col1:
    if iqa_result.grade == "PASS":
        st.success(f"### Grade: {iqa_result.grade}")
    elif iqa_result.grade == "ENHANCED_PASS":
        st.warning(f"### Grade: {iqa_result.grade}")
    else:
        st.error(f"### Grade: {iqa_result.grade}")

with q_col2:
    st.metric("Sharpness Score", f"{iqa_result.sharpness_score:.1f}", delta="≥ 120 target")
with q_col3:
    st.metric("FOV Coverage", f"{iqa_result.fov_ratio*100:.1f}%", delta="≥ 55% target")
with q_col4:
    st.metric("Luminance Mean", f"{iqa_result.exposure_metrics['overall_luminance_mean']:.1f}")

if not iqa_result.passable:
    st.error("🚫 **Image Rejected by IQA Gate.** Cannot proceed to AI grading due to severe quality defects:")
    for reason in iqa_result.rejection_reasons:
        st.write(f"- {reason}")
    st.stop()

if iqa_result.grade == "ENHANCED_PASS":
    st.warning("⚠️ **Borderline Quality Detected.** Image enhanced using Ben Graham color normalization.")
    e_col1, e_col2 = st.columns(2)
    with e_col1:
        st.image(img_rgb, caption="Original Input", width="stretch")
    with e_col2:
        st.image(iqa_result.enhanced_image, caption="Enhanced (Ben Graham)", width="stretch")
    processing_img = iqa_result.enhanced_image
else:
    processing_img = img_rgb

st.divider()

# ── Step 2 & 3 & 4: Feature Extraction, Grading & Explainability ─────────────
st.header("2. Structural Feature Segmentation & AI Grading")

with st.spinner("Processing pipeline... Extracting features, running Neural Network, & generating Grad-CAM..."):
    # Segmentation
    fov_mask, _ = extract_fov_mask(processing_img)
    (cx, cy), r_od, od_mask = detect_optic_disc(processing_img, fov_mask)
    fovea_xy, fovea_mask = estimate_fovea_location(processing_img, (cx, cy), r_od, fov_mask)
    vessel_mask, vessel_density = segment_vessels(processing_img, fov_mask, od_mask)

    evidence = detect_all_lesions(
        processing_img, fov_mask, (cx, cy), r_od, od_mask, fovea_xy, fovea_mask, vessel_mask, vessel_density
    )

    # Neural Network Forward Pass
    inp_tensor = torch.from_numpy(processing_img).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    inp_tensor = torch.nn.functional.interpolate(inp_tensor, size=(224, 224), mode="bilinear")
    with torch.no_grad():
        nn_probs = model.predict_probs(inp_tensor).squeeze(0).numpy()

    # Hybrid Clinical Fusion
    grading_result = fuse_hybrid_grading(nn_probs, evidence)

    # Grad-CAM
    cam_engine = GradCAM(model)
    heatmap, _ = cam_engine.generate_heatmap(inp_tensor, target_class=grading_result.predicted_grade)

# Display Results Layout
res_col1, res_col2 = st.columns([1, 1])

with res_col1:
    st.subheader("🎯 ICDR Severity Diagnosis")
    g_color = "red" if grading_result.is_referable else "green"

    st.markdown(f"### Predicted: **{grading_result.grade_label}** (Grade {grading_result.predicted_grade})")
    st.caption(grading_result.description)

    st.metric("Confidence Score", f"{grading_result.confidence*100:.1f}%")
    st.metric("Clinical Risk Level", grading_result.risk_level)

    if grading_result.is_referable:
        st.error("🚨 **REFERABLE DR DETECTED** — Immediate Referral to Ophthalmologist Required.")
    else:
        st.success("✅ **NON-REFERABLE DR** — Standard Routine Rescreening.")

    st.subheader("🔬 Extracted Lesion Counts")
    l_counts = evidence.counts.to_dict()
    st.json(l_counts)

with res_col2:
    st.subheader("📊 Hybrid Probability Distribution")
    fig, ax = plt.subplots(figsize=(6, 3.5))
    classes = [f"G{i}: {ICDR_LABELS[i]}" for i in range(5)]
    x_pos = np.arange(5)
    width = 0.35

    ax.bar(x_pos - width/2, grading_result.nn_probabilities, width, label="Deep NN", color="#1f77b4")
    ax.bar(x_pos + width/2, grading_result.rule_probabilities, width, label="Clinical Rules", color="#ff7f0e")

    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"G{i}" for i in range(5)])
    ax.set_ylabel("Probability")
    ax.set_title("Hybrid Model Component Probabilities")
    ax.legend()
    st.pyplot(fig)

st.divider()

# ── Step 4: Explainability Panel & Diagnostic Report ────────────────────────
st.header("3. Explainable AI Visual Evidence & Report")

composite_panel = create_composite_explanation_panel(processing_img, heatmap, evidence)
st.image(composite_panel, caption="Composite Evidence Panel (Clockwise: Original, Grad-CAM Attention, Fused Map, Lesions)", width="stretch")

diagnostic_report = generate_diagnostic_report(
    patient_id, scan_id, iqa_result, grading_result, evidence, heatmap
)

st.subheader("📝 Structured Clinical Report")
st.info(f"**Clinical Recommendation:** {diagnostic_report.clinical_recommendation}")

report_json = diagnostic_report.to_json()
st.download_button(
    "📥 Download Clinical Report (JSON)",
    data=report_json,
    file_name=f"{scan_id}_report.json",
    mime="application/json",
)

st.divider()

# ── Phase 4.5: Offline Sync Action ──────────────────────────────────────────
st.header("4. Offline-First Store & Forward Queue")

sq_col1, sq_col2 = st.columns(2)
with sq_col1:
    if st.button("💾 Save Scan to Outbox Queue"):
        outbox.enqueue(
            patient_id=patient_id,
            scan_id=scan_id,
            image_path=scan_id + ".png",
            report_dict=diagnostic_report.to_dict(),
            is_referable=grading_result.is_referable,
        )
        st.success(f"Scan `{scan_id}` saved to local SQLite outbox queue!")

with sq_col2:
    outbox_stats = outbox.get_stats()
    st.write("**Local Outbox Status:**", outbox_stats)
    if st.button("⚡ Sync Outbox to Cloud Server"):
        res = sync_worker.process_outbox_batch(mock_network_available=True)
        st.success(f"Synced {res['succeeded']} pending scans to cloud repository!")
