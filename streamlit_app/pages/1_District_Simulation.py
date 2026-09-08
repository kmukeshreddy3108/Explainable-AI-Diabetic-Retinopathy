"""
Streamlit Page 1 — District Screening Workflow Discrete-Event Simulation.
"""

import os
import sys
import streamlit as st
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.simulation.district_sim import DistrictScreeningSim

st.set_page_config(page_title="District Screening Simulation", page_icon="📈", layout="wide")

st.title("📈 Rural District DR Screening Workflow Simulation")
st.caption("Discrete-Event Capacity Planning & Resource Bottleneck Analysis (SimPy)")

st.sidebar.header("⚙️ District Parameters")
annual_patients = st.sidebar.slider("Annual Patient Volume", 10000, 300000, 100000, step=10000)
num_cameras = st.sidebar.slider("Number of Primary Care Cameras", 1, 50, 10)
acq_time = st.sidebar.slider("Camera Scan Time (min)", 2.0, 15.0, 6.0, step=0.5)
reject_rate = st.sidebar.slider("Quality Reject Rate", 0.05, 0.40, 0.15, step=0.01)

st.sidebar.subheader("👨‍⚕️ Specialist Resources")
num_doctors = st.sidebar.slider("Number of Ophthalmologists", 1, 20, 2)
doc_time = st.sidebar.slider("Doctor Review Time (min)", 0.5, 10.0, 1.5, step=0.5)
referable_pct = st.sidebar.slider("Referable DR Rate (%)", 5, 50, 25, step=1) / 100.0

sim = DistrictScreeningSim(
    annual_patients=annual_patients,
    num_cameras=num_cameras,
    acquisition_time_min=acq_time,
    quality_reject_rate=reject_rate,
    num_doctors=num_doctors,
    doc_review_time_min=doc_time,
    referable_fraction=referable_pct,
    simulation_days=30,
)

results = sim.run()

m1, m2, m3, m4 = st.columns(4)
m1.metric("Annual Throughput", f"{results['annual_extrapolated_throughput']:,}")
m2.metric("Camera Wait Time", f"{results['avg_camera_wait_min']} min")
m3.metric("Doctor Wait Time", f"{results['avg_doctor_wait_min']} min")
m4.metric("30-Day Completed Scans", f"{results['completed_scans']:,}")

st.divider()

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Resource Utilization Rates")
    fig, ax = plt.subplots(figsize=(5, 3))
    resources = ["Camera Utilization", "Doctor Utilization"]
    util_rates = [results["camera_utilization_rate"] * 100, results["doctor_utilization_rate"] * 100]
    colors = ["#2ca02c" if u < 85 else "#d62728" for u in util_rates]

    ax.bar(resources, util_rates, color=colors, width=0.4)
    ax.axhline(85, color="red", linestyle="--", label="85% Overload Threshold")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Utilization (%)")
    ax.set_title("System Resource Utilization")
    ax.legend()
    st.pyplot(fig)

with col2:
    st.subheader("🔍 Bottleneck Diagnosis")
    b_type = results["bottleneck"]
    if "No Critical" in b_type:
        st.success(f"✅ **{b_type}**\n\nSystem is operating within optimal capacity parameters.")
    else:
        st.error(f"🚨 **Primary Bottleneck:** {b_type}\n\nConsider re-allocating staff or adding hardware capacity.")

    st.write(f"- **Total Patients Arrived:** {results['total_patients_arrived']:,}")
    st.write(f"- **Quality Rescans Handled:** {results['quality_rescans']:,}")
    st.write(f"- **Cases Referred to Specialist:** {results['referred_to_doctor']:,}")
