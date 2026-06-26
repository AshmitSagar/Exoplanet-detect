import os
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from pipeline_phase2 import (
    find_transit_candidates,
    load_phase1_output,
    process_phase1_file,
    run_bls_search,
    run_phase2_pipeline,
    transit_mask,
)


TEST_TMP = Path(__file__).resolve().parent / "_tmp_phase2"


def fresh_test_dir(name):
    path = TEST_TMP / name
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True)
    return path


def make_synthetic_transit(period=2.5, t0=0.75, duration=0.16, depth=0.02, noise=0.0002):
    rng = np.random.default_rng(42)
    time = np.linspace(0.0, 20.0, 2400)
    flux = np.ones_like(time) + rng.normal(0.0, noise, size=time.size)
    in_transit = transit_mask(time, period, t0, duration, width_factor=1.0)
    flux[in_transit] -= depth
    flux_err = np.full_like(time, noise)
    return time, flux, flux_err


def save_phase1_npz(path, time, flux, flux_err, tic_id=123456789):
    metadata = {
        "tic_id": tic_id,
        "Teff": 5500.0,
        "logg": 4.4,
        "R_star": 1.0,
        "M_star": 1.0,
        "Tmag": 10.0,
        "crowding_metric": 0.02,
    }
    stats = {
        "raw_points": len(time),
        "nans_removed": 0,
        "outliers_removed": 0,
        "final_points": len(time),
    }
    np.savez_compressed(
        path,
        time=time,
        flux=flux,
        flux_err=flux_err,
        metadata=metadata,
        stats=stats,
    )


def test_load_phase1_output_reads_arrays_and_dicts():
    work_dir = fresh_test_dir("load_phase1")
    time, flux, flux_err = make_synthetic_transit()
    path = work_dir / "TIC_123456789_sector_1.npz"
    save_phase1_npz(path, time, flux, flux_err)

    loaded = load_phase1_output(path)

    assert np.allclose(loaded["time"], time)
    assert np.allclose(loaded["flux"], flux)
    assert np.allclose(loaded["flux_err"], flux_err)
    assert loaded["metadata"]["tic_id"] == 123456789
    assert loaded["stats"]["final_points"] == len(time)


def test_run_bls_search_recovers_synthetic_period_and_depth():
    period = 2.5
    depth = 0.02
    time, flux, flux_err = make_synthetic_transit(period=period, depth=depth)

    candidate, results = run_bls_search(
        time,
        flux,
        flux_err,
        minimum_period=1.0,
        maximum_period=5.0,
        durations=np.array([0.12, 0.16, 0.20]),
    )

    assert abs(candidate["period_days"] - period) < 0.03
    assert abs(candidate["depth_ppm"] - depth * 1_000_000) < 2_000
    assert candidate["duration_hours"] > 0
    assert candidate["snr"] > 7
    assert candidate["sde"] > 5
    assert candidate["n_transits"] >= 7
    assert len(results.period) > 0


def test_find_transit_candidates_applies_thresholds():
    time, flux, flux_err = make_synthetic_transit()

    candidates = find_transit_candidates(
        time,
        flux,
        flux_err,
        max_candidates=2,
        sde_threshold=5.0,
        snr_threshold=7.0,
        minimum_period=1.0,
        maximum_period=5.0,
        durations=np.array([0.12, 0.16, 0.20]),
    )

    assert len(candidates) >= 1
    assert candidates[0]["snr"] >= 7.0

    no_candidates = find_transit_candidates(
        time,
        flux,
        flux_err,
        max_candidates=2,
        sde_threshold=1_000.0,
        snr_threshold=1_000.0,
        minimum_period=1.0,
        maximum_period=5.0,
        durations=np.array([0.12, 0.16, 0.20]),
    )

    assert no_candidates == []


def test_process_phase1_file_and_pipeline_write_candidate_csv():
    work_dir = fresh_test_dir("pipeline")
    time, flux, flux_err = make_synthetic_transit()
    input_dir = work_dir / "processed"
    output_csv = work_dir / "candidates" / "phase2_candidates.csv"
    input_dir.mkdir()
    path = input_dir / "TIC_987654321_sector_1.npz"
    save_phase1_npz(path, time, flux, flux_err, tic_id=987654321)

    records = process_phase1_file(
        path,
        max_candidates=1,
        sde_threshold=5.0,
        snr_threshold=7.0,
        minimum_period=1.0,
        maximum_period=5.0,
        durations=np.array([0.12, 0.16, 0.20]),
    )

    assert len(records) == 1
    assert records[0].tic_id == 987654321
    assert records[0].candidate_number == 1

    pipeline_records = run_phase2_pipeline(
        input_dir=input_dir,
        output_csv=output_csv,
        max_candidates=1,
        sde_threshold=5.0,
        snr_threshold=7.0,
        minimum_period=1.0,
        maximum_period=5.0,
        durations=np.array([0.12, 0.16, 0.20]),
    )

    assert len(pipeline_records) == 1
    assert output_csv.exists()
    csv_text = output_csv.read_text(encoding="utf-8")
    assert "tic_id" in csv_text
    assert "987654321" in csv_text
