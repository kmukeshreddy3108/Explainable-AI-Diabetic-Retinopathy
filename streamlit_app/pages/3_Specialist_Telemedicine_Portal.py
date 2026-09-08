"""
Streamlit Page 3 — District Telemedicine Specialist Review Portal & PHC Feedback Loop.

Enables nearby District Hospital Ophthalmologists to review synced high-risk scans,
confirm referrals, and transmit clinical responses back to the rural Primary Health Center (PHC).
"""

import os
import sys
import json
import streamlit as st

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.sync.outbox import OutboxQueue

st.set_page_config(page_title="Specialist Telemedicine Portal", page_icon="🏥", layout="wide")

st.title("🏥 District Telemedicine Specialist Review Portal")
st.caption("Human-in-the-Loop Validation & Offline PHC Synchronized Feedback Loop")

outbox = OutboxQueue()
scans = outbox.fetch_all_scans()

if not scans:
    st.info("ℹ️ No pending scans in the outbox. Process a scan on the Screening Desk to send items to the outbox queue.")
    st.stop()

st.subheader("📥 District Referral Queue (Prioritized High-Risk Scans First)")

# Display overview metrics
total_scans = len(scans)
referable_count = sum(1 for s in scans if s["priority"] == 2)
reviewed_count = sum(1 for s in scans if s.get("specialist_status") != "pending_review")

m1, m2, m3 = st.columns(3)
m1.metric("Total Queued Scans", total_scans)
m2.metric("High-Risk Referable Cases", referable_count)
m3.metric("Ophthalmologist Reviews Completed", reviewed_count)

st.divider()

for scan in scans:
    report = json.loads(scan["report_json"]) if isinstance(scan["report_json"], str) else scan["report_json"]
    p_badge = "🔴 HIGH PRIORITY (Referable)" if scan["priority"] == 2 else "🟢 NORMAL (Routine)"

    with st.expander(f"Scan ID: {scan['scan_id']} | Patient: {scan['patient_id']} | {p_badge}"):
        col1, col2 = st.columns([1, 1])

        with col1:
            st.markdown("#### 📑 AI Screening Diagnosis & Report")
            st.write(f"- **ICDR Diagnosis:** Grade {report['grading']['predicted_grade']} — {report['grading']['grade_label']}")
            st.write(f"- **Confidence:** {report['grading']['confidence']*100:.1f}%")
            st.write(f"- **IQA Quality Status:** {report['iqa']['grade']}")
            st.write(f"- **Detected Lesion Counts:**", report['lesions']['counts'])
            st.info(f"**AI Recommendation:** {report['clinical_recommendation']}")

        with col2:
            st.markdown("#### 👨‍⚕️ Ophthalmologist Telemedicine Response")
            current_status = scan.get("specialist_status", "pending_review")
            current_notes = scan.get("specialist_notes", "")

            st.write(f"**Current Status:** `{current_status}`")

            new_status = st.selectbox(
                "Clinical Decision Response",
                ["confirmed_referral", "phc_rescreen", "recapture_requested", "pending_review"],
                index=["confirmed_referral", "phc_rescreen", "recapture_requested", "pending_review"].index(current_status),
                key=f"status_{scan['scan_id']}",
            )

            new_notes = st.text_area(
                "Clinical Feedback / Treatment Instructions for PHC Worker",
                value=current_notes if current_notes else "Patient confirmed for urgent tertiary hospital referral.",
                key=f"notes_{scan['scan_id']}",
            )

            if st.button("📤 Send Response Back to Rural PHC", key=f"btn_{scan['scan_id']}"):
                outbox.record_specialist_review(scan["scan_id"], new_status, new_notes)
                st.success(f"Response recorded for `{scan['scan_id']}`! Telemedicine feedback synced back to PHC database.")
                st.rerun()
