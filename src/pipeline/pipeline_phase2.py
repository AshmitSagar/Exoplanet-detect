"""
pipeline_phase2.py — Stage 2: Transit Search → AI-Ready Feature Vectors

Flow:
    Phase 1 .npz  →  TLS search  →  candidate detection  →  phase-fold
                  →  global_view (2001 pts) + local_view (201 pts)
                  →  .npz per candidate  →  Phase 3 CNN / classifier

Design decisions:
    - TLS only (no BLS fallback bloat; install transitleastsquares)
    - No joblib parallelism — simple loop, easy to debug in 30 hrs
    - global_view + local_view match Astronet / AstroNet-Kepler conventions
      so any 1D-CNN or Transformer from literature plugs straight in
    - label = -1 (unknown) until Phase 3 labelling stage
    - One CSV summary + one .npz per candidate — nothing else written to disk
"""

import csv
import glob
import os
import re
import traceback
from dataclasses import asdict, dataclass, fields
from typing import Optional

import numpy as np
from tqdm import tqdm
from transitleastsquares import transitleastsquares


# ---------------------------------------------------------------------------
# Data contract — every field here becomes a feature / metadata column
# ---------------------------------------------------------------------------

@dataclass
class TransitCandidate:
    source_file:      str
    tic_id:           int          # -1 if unknown
    candidate_number: int          # 1-indexed per star (for multi-planet)
    period_days:      float
    t0:               float        # time of first transit [BTJD]
    depth_ppm:        float        # transit depth in parts per million
    duration_hours:   float
    snr:              float        # TLS signal-to-noise
    sde:              float        # Signal Detection Efficiency (TLS native)
    n_transits:       int          # observed transits inside time baseline


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------

def load_phase1(path: str) -> dict:
    """Load a Phase 1 .npz artifact. Raises clearly if keys are missing."""
    data = np.load(path, allow_pickle=True)
    for key in ("time", "flux", "flux_err", "metadata"):
        if key not in data.files:
            raise KeyError(f"Phase 1 file missing key '{key}': {path}")
    return {
        "time":     np.asarray(data["time"],     dtype=float),
        "flux":     np.asarray(data["flux"],     dtype=float),
        "flux_err": np.asarray(data["flux_err"], dtype=float),
        "metadata": data["metadata"].item(),
    }


def tic_id_from(path: str, metadata: dict) -> int:
    """Extract TIC ID from metadata dict or filename. Returns -1 if absent."""
    try:
        return int(metadata.get("tic_id", -1))
    except (TypeError, ValueError):
        pass
    m = re.search(r"TIC_?(\d+)", os.path.basename(path), re.IGNORECASE)
    return int(m.group(1)) if m else -1


# ---------------------------------------------------------------------------
# Light-curve sanitiser
# ---------------------------------------------------------------------------

def sanitise(time, flux, flux_err=None):
    """
    Return time-sorted, finite-only (time, flux, flux_err).
    Synthesises flux_err from scatter if not provided.
    Raises ValueError if fewer than 10 usable cadences remain.
    """
    time, flux = np.asarray(time, float), np.asarray(flux, float)
    flux_err = (np.asarray(flux_err, float) if flux_err is not None
                else np.full_like(flux, np.nanstd(flux - np.nanmedian(flux))))

    ok = np.isfinite(time) & np.isfinite(flux) & np.isfinite(flux_err) & (flux_err > 0)
    if ok.sum() < 10:
        raise ValueError(f"Only {ok.sum()} finite cadences — need ≥10")

    idx = np.argsort(time[ok])
    return time[ok][idx], flux[ok][idx], flux_err[ok][idx]


# ---------------------------------------------------------------------------
# Transit search (TLS)
# ---------------------------------------------------------------------------

def tls_search(time, flux, flux_err,
               period_min=0.5, period_max=27.0) -> dict:
    """
    Run TLS on a single sanitised light curve.
    Returns a flat dict of transit parameters — same keys as TransitCandidate
    (minus source_file / tic_id / candidate_number).
    """
    period_max = min(period_max, float(np.ptp(time)))

    results = transitleastsquares(time, flux, flux_err).power(
        period_min          = period_min,
        period_max          = period_max,
        oversampling_factor = 3,
        duration_grid_step  = 1.05,
        show_progress_bar   = False,
    )

    period        = float(results.period)
    t0            = float(results.T0)
    duration_days = float(results.duration)
    depth_frac    = float(results.depth)           # e.g. 0.99 means 1% dip
    sde           = float(results.SDE)
    snr           = float(getattr(results, "snr", sde))

    # Count how many transits actually fall inside the observed baseline
    baseline_start, baseline_end = np.nanmin(time), np.nanmax(time)
    first_epoch = int(np.floor((baseline_start - t0) / period))
    last_epoch  = int(np.ceil( (baseline_end   - t0) / period))
    centers     = t0 + np.arange(first_epoch, last_epoch + 1) * period
    n_transits  = int(np.sum(
        (centers >= baseline_start - duration_days) &
        (centers <= baseline_end   + duration_days)
    ))

    return dict(
        period_days    = period,
        t0             = t0,
        depth_ppm      = abs(1.0 - depth_frac) * 1_000_000.0,
        duration_hours = duration_days * 24.0,
        snr            = snr,
        sde            = sde,
        n_transits     = n_transits,
    )


# ---------------------------------------------------------------------------
# Iterative multi-planet search
# ---------------------------------------------------------------------------

def find_candidates(time, flux, flux_err,
                    max_planets=3,
                    sde_threshold=9.0,
                    snr_threshold=7.0,
                    **tls_kwargs) -> list[dict]:
    """
    Mask each detected transit and re-run TLS to find up to max_planets signals.
    Stops early if SDE or SNR drops below threshold.
    """
    time, working_flux, flux_err = sanitise(time, flux, flux_err)
    found = []

    for _ in range(max_planets):
        try:
            hit = tls_search(time, working_flux, flux_err, **tls_kwargs)
        except Exception:
            break

        if hit["sde"] < sde_threshold or hit["snr"] < snr_threshold:
            break

        found.append(hit)

        # Mask detected transit windows (1.5× duration) so next iteration
        # searches the residuals for additional planets
        dur   = hit["duration_hours"] / 24.0
        phase = ((time - hit["t0"] + 0.5 * hit["period_days"])
                 % hit["period_days"]) - 0.5 * hit["period_days"]
        in_transit = np.abs(phase) <= 0.75 * dur      # 1.5× half-duration
        if in_transit.sum() == 0:
            break
        working_flux = working_flux.copy()
        working_flux[in_transit] = 1.0

    return found


# ---------------------------------------------------------------------------
# Phase-folding & AI view generation
# ---------------------------------------------------------------------------

def phase_fold(time, flux, period, t0):
    """Phase in [-0.5, 0.5), sorted."""
    phase = ((time - t0) / period + 0.5) % 1.0 - 0.5
    idx   = np.argsort(phase)
    return phase[idx], flux[idx]


def median_bin(phase, flux, bins):
    """Median-bin flux into `bins` edges. Unfilled bins → 1.0 (baseline)."""
    view = np.ones(len(bins) - 1, dtype=float)
    for i in range(len(bins) - 1):
        mask = (phase >= bins[i]) & (phase < bins[i + 1])
        if mask.any():
            view[i] = np.median(flux[mask])
    return view


def _normalise(v):
    """Zero-median, unit-std normalisation. Returns v unchanged if std==0."""
    std = np.std(v)
    return (v - np.median(v)) / std if std > 0 else v


def global_view(phase, flux, n_bins=2001):
    """
    2001-bin representation of the full phase-folded light curve.
    Matches the AstroNet / Astronet-Kepler global-view convention.
    Shape: (2001,)  — direct input to 1D-CNN branch 1.
    """
    bins = np.linspace(-0.5, 0.5, n_bins + 1)
    return _normalise(median_bin(phase, flux, bins))


def local_view(phase, flux, duration_days, period_days, n_bins=201):
    """
    201-bin zoom on the ±2× transit-duration window around phase=0.
    Matches the AstroNet / Astronet-Kepler local-view convention.
    Shape: (201,)  — direct input to 1D-CNN branch 2.
    """
    half_w = min(2.0 * duration_days / period_days, 0.45)
    bins   = np.linspace(-half_w, half_w, n_bins + 1)
    return _normalise(median_bin(phase, flux, bins))


def save_views(candidate: TransitCandidate, time, flux,
               metadata: dict, out_dir: str) -> str:
    """
    Phase-fold and save .npz with everything Phase 3 needs.

    Keys in the .npz
    ─────────────────────────────────────────────────────────────
    global_view     float32  (2001,)   full phase-folded curve
    local_view      float32  (201,)    zoom on transit
    period          float64  scalar
    t0              float64  scalar    [BTJD]
    depth_ppm       float64  scalar
    duration_hours  float64  scalar
    snr             float64  scalar
    sde             float64  scalar
    n_transits      int32    scalar
    tic_id          int64    scalar    (-1 if unknown)
    label           int32    scalar    (-1 = unlabelled)
                                       0 = non-transit
                                       1 = transit
                                       2 = EB / eclipse
                                       3 = blend / other
    metadata        object             raw Phase 1 dict
    ─────────────────────────────────────────────────────────────
    """
    os.makedirs(out_dir, exist_ok=True)

    phase, folded = phase_fold(time, flux,
                               candidate.period_days, candidate.t0)
    gv = global_view(phase, folded).astype(np.float32)
    lv = local_view(phase, folded,
                    duration_days  = candidate.duration_hours / 24.0,
                    period_days    = candidate.period_days).astype(np.float32)

    fname = (f"TIC{candidate.tic_id}"
             f"_c{candidate.candidate_number}"
             f"_P{candidate.period_days:.4f}.npz")
    fpath = os.path.join(out_dir, fname)

    np.savez_compressed(
        fpath,
        global_view    = gv,
        local_view     = lv,
        period         = np.float64(candidate.period_days),
        t0             = np.float64(candidate.t0),
        depth_ppm      = np.float64(candidate.depth_ppm),
        duration_hours = np.float64(candidate.duration_hours),
        snr            = np.float64(candidate.snr),
        sde            = np.float64(candidate.sde),
        n_transits     = np.int32(candidate.n_transits),
        tic_id         = np.int64(candidate.tic_id),
        label          = np.int32(-1),
        metadata       = np.array(metadata, dtype=object),
    )
    return fpath


# ---------------------------------------------------------------------------
# CSV summary
# ---------------------------------------------------------------------------

def write_csv(candidates: list[TransitCandidate], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    col_names = [f.name for f in fields(TransitCandidate)]
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=col_names)
        writer.writeheader()
        for c in candidates:
            writer.writerow(asdict(c))


# ---------------------------------------------------------------------------
# Per-file processor
# ---------------------------------------------------------------------------

def process_file(path: str, views_dir: str, **tls_kwargs) -> list[TransitCandidate]:
    """Load one Phase 1 file, search for transits, save views. Returns candidates."""
    try:
        target   = load_phase1(path)
        tic_id   = tic_id_from(path, target["metadata"])
        time     = target["time"]
        flux     = target["flux"]
        flux_err = target["flux_err"]
        metadata = target["metadata"]

        hits = find_candidates(time, flux, flux_err, **tls_kwargs)

        records = []
        for i, hit in enumerate(hits, start=1):
            record = TransitCandidate(
                source_file      = os.path.basename(path),
                tic_id           = tic_id,
                candidate_number = i,
                **hit,
            )
            save_views(record, time, flux, metadata, out_dir=views_dir)
            records.append(record)

        return records

    except Exception:
        print(f"\n[WARN] Skipped {os.path.basename(path)}:\n{traceback.format_exc()}")
        return []


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    input_dir    = "data/processed",
    output_csv   = "data/candidates/phase2_candidates.csv",
    views_dir    = "data/candidates/views",
    # TLS thresholds
    sde_threshold  = 9.0,
    snr_threshold  = 7.0,
    max_planets    = 3,
    period_min     = 0.5,
    period_max     = 27.0,
):
    paths = sorted(glob.glob(os.path.join(input_dir, "*.npz")))
    if not paths:
        print(f"[Phase 2] No .npz files in '{input_dir}'. Run Phase 1 first.")
        return []

    print(f"[Phase 2] Searching {len(paths)} star(s) with TLS …")

    all_candidates = []
    for path in tqdm(paths, unit="star"):
        results = process_file(
            path,
            views_dir     = views_dir,
            max_planets   = max_planets,
            sde_threshold = sde_threshold,
            snr_threshold = snr_threshold,
            period_min    = period_min,
            period_max    = period_max,
        )
        all_candidates.extend(results)

    write_csv(all_candidates, output_csv)
    print(f"\n[Phase 2] Done.  {len(all_candidates)} candidate(s) from {len(paths)} star(s)")
    print(f"  CSV   → {output_csv}")
    print(f"  Views → {views_dir}/")
    return all_candidates


if __name__ == "__main__":
    run()