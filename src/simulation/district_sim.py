"""
District-Level Discrete-Event Screening Simulation Model.

Models patient throughput, camera resource utilization, AI triage filtering,
and ophthalmologist review queues across a rural healthcare district network.
"""

from typing import Dict, Any, List
import random
import numpy as np

try:
    import simpy
    HAS_SIMPY = True
except ImportError:
    HAS_SIMPY = False

from src.utils.config import (
    SIM_DEFAULT_ANNUAL_PATIENTS,
    SIM_DEFAULT_NUM_CAMERAS,
    SIM_DEFAULT_ACQUISITION_TIME_MIN,
    SIM_DEFAULT_QUALITY_REJECT_RATE,
    SIM_DEFAULT_AI_THROUGHPUT_IMG_MIN,
    SIM_DEFAULT_NUM_DOCTORS,
    SIM_DEFAULT_DOC_REVIEW_TIME_MIN,
    SIM_DEFAULT_REFERABLE_FRACTION,
)


class DistrictScreeningSim:
    """Discrete-event simulation engine for District DR Screening Workflow."""

    def __init__(
        self,
        annual_patients: int = SIM_DEFAULT_ANNUAL_PATIENTS,
        num_cameras: int = SIM_DEFAULT_NUM_CAMERAS,
        acquisition_time_min: float = SIM_DEFAULT_ACQUISITION_TIME_MIN,
        quality_reject_rate: float = SIM_DEFAULT_QUALITY_REJECT_RATE,
        num_doctors: int = SIM_DEFAULT_NUM_DOCTORS,
        doc_review_time_min: float = SIM_DEFAULT_DOC_REVIEW_TIME_MIN,
        referable_fraction: float = SIM_DEFAULT_REFERABLE_FRACTION,
        simulation_days: int = 30,
        seed: int = 42,
    ):
        self.annual_patients = annual_patients
        self.num_cameras = num_cameras
        self.acquisition_time_min = acquisition_time_min
        self.quality_reject_rate = quality_reject_rate
        self.num_doctors = num_doctors
        self.doc_review_time_min = doc_review_time_min
        self.referable_fraction = referable_fraction
        self.simulation_days = simulation_days
        self.seed = seed

    def run_simpy_simulation(self) -> Dict[str, Any]:
        """Run simulation using SimPy framework if available."""
        random.seed(self.seed)
        np.random.seed(self.seed)

        env = simpy.Environment()
        cameras = simpy.Resource(env, capacity=self.num_cameras)
        doctors = simpy.Resource(env, capacity=self.num_doctors)

        # Working hours: 250 working days/year * 8 hours/day * 60 min/hour = 120,000 min/year
        total_working_min = self.simulation_days * 8 * 60
        daily_patients = self.annual_patients / 250.0
        patient_interarrival_min = (8 * 60) / daily_patients if daily_patients > 0 else 1.0

        camera_wait_times: List[float] = []
        doctor_wait_times: List[float] = []
        completed_scans = 0
        referred_to_doctor = 0
        quality_rescans = 0

        def patient_flow(env, patient_id):
            nonlocal completed_scans, referred_to_doctor, quality_rescans

            # 1. Camera Acquisition Queue
            arr_time = env.now
            with cameras.request() as req:
                yield req
                camera_wait_times.append(env.now - arr_time)
                # Acquisition duration
                yield env.timeout(random.normalvariate(self.acquisition_time_min, 1.0))

            # 2. Automated AI Triage Assessment (Instantaneous ~ 1 sec)
            completed_scans += 1
            if random.random() < self.quality_reject_rate:
                quality_rescans += 1
                # Minor delay for immediate rescan
                yield env.timeout(2.0)

            # 3. Decision: Is Case Referable to Doctor?
            if random.random() < self.referable_fraction:
                referred_to_doctor += 1
                doc_arr_time = env.now
                with doctors.request() as doc_req:
                    yield doc_req
                    doctor_wait_times.append(env.now - doc_arr_time)
                    yield env.timeout(random.normalvariate(self.doc_review_time_min, 0.3))

        def patient_generator(env):
            p_id = 0
            while env.now < total_working_min:
                env.process(patient_flow(env, p_id))
                p_id += 1
                yield env.timeout(random.exponential(patient_interarrival_min))

        env.process(patient_generator(env))
        env.run(until=total_working_min)

        avg_cam_wait = float(np.mean(camera_wait_times)) if camera_wait_times else 0.0
        avg_doc_wait = float(np.mean(doctor_wait_times)) if doctor_wait_times else 0.0

        # Calculate Utilization Rates
        total_cam_service_time = completed_scans * self.acquisition_time_min
        cam_utilization = min(1.0, total_cam_service_time / (self.num_cameras * total_working_min))

        total_doc_service_time = referred_to_doctor * self.doc_review_time_min
        doc_utilization = min(1.0, total_doc_service_time / (self.num_doctors * total_working_min))

        # Bottleneck identification
        if doc_utilization > 0.85 and avg_doc_wait > avg_cam_wait:
            bottleneck = "Ophthalmologist Specialist Availability"
        elif cam_utilization > 0.85:
            bottleneck = "Fundus Camera Hardware Capacity"
        else:
            bottleneck = "No Critical Bottleneck Detected"

        return {
            "simulation_days": self.simulation_days,
            "total_patients_arrived": len(camera_wait_times),
            "completed_scans": completed_scans,
            "quality_rescans": quality_rescans,
            "referred_to_doctor": referred_to_doctor,
            "avg_camera_wait_min": round(avg_cam_wait, 2),
            "avg_doctor_wait_min": round(avg_doc_wait, 2),
            "camera_utilization_rate": round(cam_utilization, 4),
            "doctor_utilization_rate": round(doc_utilization, 4),
            "bottleneck": bottleneck,
            "annual_extrapolated_throughput": int(completed_scans * (250 / self.simulation_days)),
        }

    def run_fallback_simulation(self) -> Dict[str, Any]:
        """Analytical queuing model fallback when simpy is not installed."""
        total_working_min = self.simulation_days * 8 * 60
        daily_patients = self.annual_patients / 250.0
        total_patients = int(daily_patients * self.simulation_days)

        completed_scans = total_patients
        quality_rescans = int(total_patients * self.quality_reject_rate)
        referred_to_doctor = int(total_patients * self.referable_fraction)

        avg_cam_wait = max(0.5, (total_patients * self.acquisition_time_min) / (self.num_cameras * 100))
        avg_doc_wait = max(1.0, (referred_to_doctor * self.doc_review_time_min) / (self.num_doctors * 50))

        cam_utilization = min(0.95, (total_patients * self.acquisition_time_min) / (self.num_cameras * total_working_min))
        doc_utilization = min(0.95, (referred_to_doctor * self.doc_review_time_min) / (self.num_doctors * total_working_min))

        if doc_utilization > 0.85:
            bottleneck = "Ophthalmologist Specialist Availability"
        elif cam_utilization > 0.85:
            bottleneck = "Fundus Camera Hardware Capacity"
        else:
            bottleneck = "No Critical Bottleneck Detected"

        return {
            "simulation_days": self.simulation_days,
            "total_patients_arrived": total_patients,
            "completed_scans": completed_scans,
            "quality_rescans": quality_rescans,
            "referred_to_doctor": referred_to_doctor,
            "avg_camera_wait_min": round(avg_cam_wait, 2),
            "avg_doctor_wait_min": round(avg_doc_wait, 2),
            "camera_utilization_rate": round(cam_utilization, 4),
            "doctor_utilization_rate": round(doc_utilization, 4),
            "bottleneck": bottleneck,
            "annual_extrapolated_throughput": int(completed_scans * (250 / self.simulation_days)),
        }

    def run(self) -> Dict[str, Any]:
        if HAS_SIMPY:
            return self.run_simpy_simulation()
        else:
            return self.run_fallback_simulation()
