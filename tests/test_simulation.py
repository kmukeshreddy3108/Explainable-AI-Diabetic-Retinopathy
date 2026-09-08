"""
Unit tests for Module 5 — District Workflow Discrete-Event Simulation.
"""

import pytest
from src.simulation.district_sim import DistrictScreeningSim


def test_district_simulation_run():
    sim = DistrictScreeningSim(
        annual_patients=10000,
        num_cameras=5,
        num_doctors=2,
        simulation_days=10,
        seed=42,
    )

    metrics = sim.run()

    assert "completed_scans" in metrics
    assert "avg_camera_wait_min" in metrics
    assert "avg_doctor_wait_min" in metrics
    assert "camera_utilization_rate" in metrics
    assert "doctor_utilization_rate" in metrics
    assert "bottleneck" in metrics
    assert metrics["completed_scans"] > 0
    assert 0.0 <= metrics["camera_utilization_rate"] <= 1.0
    assert 0.0 <= metrics["doctor_utilization_rate"] <= 1.0


def test_doctor_capacity_scaling():
    # 1 Doctor
    sim1 = DistrictScreeningSim(
        annual_patients=20000, num_cameras=10, num_doctors=1, simulation_days=10, seed=42
    )
    res1 = sim1.run()

    # 5 Doctors
    sim2 = DistrictScreeningSim(
        annual_patients=20000, num_cameras=10, num_doctors=5, simulation_days=10, seed=42
    )
    res2 = sim2.run()

    assert res2["doctor_utilization_rate"] <= res1["doctor_utilization_rate"]
