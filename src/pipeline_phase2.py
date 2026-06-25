import csv
import glob
import os
import re
from dataclasses import asdict, dataclass

import numpy as np
from astropy.timeseries import BoxLeastSquares


@dataclass
class TransitCandidate:
    """Single threshold-crossing event from the Stage 2 transit search."""

    source_file: str
    tic_id: int | None
    candidate_number: int
    period_days: float
    t0: float
    depth_ppm: float
    duration_hours: float
    snr: float
    sde: float
    power: float
    n_transits: int


def load_phase1_output(path):
    """
    Read the compressed .npz artifact produced by pipeline_phase1.py.

    Phase 1 stores Python dictionaries for metadata/stats, so allow_pickle=True
    and .item() are required for those two fields.
    """
    data = np.load(path, allow_pickle=True)
    required = {"time", "flux", "flux_err", "metadata", "stats"}
    missing = required.difference(data.files)
    if missing:
        missing_list = ", ".join(sorted(missing))
        raise ValueError(f"Phase 1 file {path!r} is missing: {missing_list}")

    return {
        "time": np.asarray(data["time"], dtype=float),
        "flux": np.asarray(data["flux"], dtype=float),
        "flux_err": np.asarray(data["flux_err"], dtype=float),
        "metadata": data["metadata"].item(),
        "stats": data["stats"].item(),
    }


def _tic_id_from_path(path, metadata=None):
    if metadata and "tic_id" in metadata:
        try:
            return int(metadata["tic_id"])
        except (TypeError, ValueError):
            pass

    match = re.search(r"TIC_(\d+)", os.path.basename(path))
    if match:
        return int(match.group(1))
    return None


def _finite_lightcurve(time, flux, flux_err=None):
    time = np.asarray(time, dtype=float)
    flux = np.asarray(flux, dtype=float)

    if flux_err is None:
        flux_err = np.full_like(flux, np.nanstd(flux - np.nanmedian(flux)))
    else:
        flux_err = np.asarray(flux_err, dtype=float)

    mask = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if np.count_nonzero(mask) < 10:
        raise ValueError("Need at least 10 finite light-curve points for BLS search")

    order = np.argsort(time[mask])
    return time[mask][order], flux[mask][order], flux_err[mask][order]


def compute_sde(power):
    """Signal Detection Efficiency: peak power above the BLS power background."""
    power = np.asarray(power, dtype=float)
    finite_power = power[np.isfinite(power)]
    if finite_power.size == 0:
        return 0.0

    scatter = np.nanstd(finite_power)
    if scatter == 0 or not np.isfinite(scatter):
        return 0.0

    return float((np.nanmax(finite_power) - np.nanmedian(finite_power)) / scatter)


def count_transits(time, period, t0, duration):
    """Count unique predicted transit centers covered by the light curve."""
    if period <= 0:
        return 0

    first_epoch = int(np.floor((np.nanmin(time) - t0) / period))
    last_epoch = int(np.ceil((np.nanmax(time) - t0) / period))
    centers = t0 + np.arange(first_epoch, last_epoch + 1) * period
    in_baseline = (centers >= np.nanmin(time) - duration) & (centers <= np.nanmax(time) + duration)
    return int(np.count_nonzero(in_baseline))


def transit_mask(time, period, t0, duration, width_factor=1.5):
    """Return True for points close to predicted transit centers."""
    phase = ((time - t0 + 0.5 * period) % period) - 0.5 * period
    return np.abs(phase) <= 0.5 * duration * width_factor


def run_bls_search(
    time,
    flux,
    flux_err=None,
    minimum_period=0.5,
    maximum_period=27.0,
    durations=None,
    frequency_factor=2.0,
):
    """
    Run a Box Least Squares search and return the strongest candidate plus raw results.

    Flux is expected to be Phase-1 normalized around 1.0. BLS is run on
    relative flux residuals, so the returned depth is converted to ppm.
    """
    time, flux, flux_err = _finite_lightcurve(time, flux, flux_err)

    baseline = np.nanmax(time) - np.nanmin(time)
    if baseline <= minimum_period:
        raise ValueError("Light-curve baseline is shorter than the minimum search period")

    maximum_period = min(maximum_period, baseline)
    if maximum_period <= minimum_period:
        raise ValueError("maximum_period must be greater than minimum_period")

    if durations is None:
        durations = np.linspace(0.04, 0.30, 12)
    durations = np.asarray(durations, dtype=float)

    residual_flux = flux - 1.0
    model = BoxLeastSquares(time, residual_flux, dy=flux_err)
    results = model.autopower(
        durations,
        minimum_period=minimum_period,
        maximum_period=maximum_period,
        frequency_factor=frequency_factor,
    )

    best = int(np.nanargmax(results.power))
    period = float(results.period[best])
    t0 = float(results.transit_time[best])
    duration = float(results.duration[best])
    depth = float(results.depth[best])
    depth_ppm = abs(depth) * 1_000_000.0
    power = float(results.power[best])
    snr = float(results.depth_snr[best]) if hasattr(results, "depth_snr") else 0.0
    sde = compute_sde(results.power)
    n_transits = count_transits(time, period, t0, duration)

    candidate = {
        "period_days": period,
        "t0": t0,
        "depth_ppm": depth_ppm,
        "duration_hours": duration * 24.0,
        "snr": abs(snr),
        "sde": sde,
        "power": power,
        "n_transits": n_transits,
    }
    return candidate, results


def find_transit_candidates(
    time,
    flux,
    flux_err=None,
    max_candidates=3,
    sde_threshold=9.0,
    snr_threshold=7.0,
    **search_kwargs,
):
    """
    Iteratively find transit candidates, masking each detection before rerunning BLS.
    """
    time, working_flux, flux_err = _finite_lightcurve(time, flux, flux_err)
    candidates = []

    for _ in range(max_candidates):
        candidate, _ = run_bls_search(time, working_flux, flux_err, **search_kwargs)
        if candidate["sde"] < sde_threshold or candidate["snr"] < snr_threshold:
            break

        candidates.append(candidate)

        duration_days = candidate["duration_hours"] / 24.0
        mask = transit_mask(time, candidate["period_days"], candidate["t0"], duration_days)
        if np.count_nonzero(mask) == 0:
            break

        working_flux = working_flux.copy()
        working_flux[mask] = 1.0

    return candidates


def process_phase1_file(
    path,
    max_candidates=3,
    sde_threshold=9.0,
    snr_threshold=7.0,
    **search_kwargs,
):
    """Load one Phase 1 .npz file and return Stage 2 candidate records."""
    target = load_phase1_output(path)
    tic_id = _tic_id_from_path(path, target["metadata"])
    candidates = find_transit_candidates(
        target["time"],
        target["flux"],
        target["flux_err"],
        max_candidates=max_candidates,
        sde_threshold=sde_threshold,
        snr_threshold=snr_threshold,
        **search_kwargs,
    )

    records = []
    for index, candidate in enumerate(candidates, start=1):
        records.append(
            TransitCandidate(
                source_file=os.path.basename(path),
                tic_id=tic_id,
                candidate_number=index,
                **candidate,
            )
        )
    return records


def write_candidates_csv(candidates, output_path):
    """Write Stage 2 candidates to a CSV file."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fieldnames = list(asdict(TransitCandidate("", None, 0, 0, 0, 0, 0, 0, 0, 0, 0)).keys())

    with open(output_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for candidate in candidates:
            writer.writerow(asdict(candidate))


def run_phase2_pipeline(
    input_dir="data/processed",
    output_csv="data/candidates/phase2_candidates.csv",
    pattern="*.npz",
    **search_kwargs,
):
    """Run Stage 2 transit search over all Phase 1 .npz files in input_dir."""
    paths = sorted(glob.glob(os.path.join(input_dir, pattern)))
    all_candidates = []

    for path in paths:
        all_candidates.extend(process_phase1_file(path, **search_kwargs))

    write_candidates_csv(all_candidates, output_csv)
    print(f"Phase 2 finished. Found {len(all_candidates)} candidates across {len(paths)} files.")
    print(f"Saved candidate table to: {output_csv}")
    return all_candidates


if __name__ == "__main__":
    run_phase2_pipeline()
