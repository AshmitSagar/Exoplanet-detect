import os
import lightkurve as lk
from astroquery.mast import Catalogs
import numpy as np


def download_lightcurves(tic_id, sector=None, download_dir="data/raw",
                         stitch=True, max_sectors=None):
    """
    Download TESS-SPOC 2-min PDCSAP flux for a TIC ID.

    Args:
        tic_id      : TIC ID (int)
        sector      : specific sector (int), or None for all available
        download_dir: where raw files go
        stitch      : if True, stitch + normalise multi-sector into one LC
        max_sectors : cap how many sectors to download (None = no cap)
                      e.g. max_sectors=3 keeps it fast for hackathon runs
    """
    os.makedirs(download_dir, exist_ok=True)

    search_result = lk.search_lightcurve(
        f"TIC {tic_id}", mission="TESS",
        sector=sector, author="SPOC", exptime=120
    )

    if len(search_result) == 0:
        print(f"[TIC {tic_id}] No SPOC 2-min data found (sector={sector})")
        return None

    # ── Cap sectors to avoid bulk downloads ──────────────────────────────────
    if max_sectors is not None and len(search_result) > max_sectors:
        print(f"[TIC {tic_id}] {len(search_result)} sectors found — "
              f"capping at {max_sectors}")
        search_result = search_result[:max_sectors]
    else:
        print(f"[TIC {tic_id}] Downloading {len(search_result)} sector(s)")

    lc_collection = search_result.download_all(download_dir=download_dir)

    if lc_collection is None or len(lc_collection) == 0:
        return None

    # ── Stitch or return single sector ───────────────────────────────────────
    if len(lc_collection) == 1:
        return lc_collection[0]

    if stitch:
        # stitch() normalises each sector to the same median before joining
        return lc_collection.stitch()

    # stitch=False → return collection as-is (caller handles it)
    return lc_collection


def fetch_tic_metadata(tic_id):
    """
    Fetch stellar parameters from the TIC catalog.
    Teff, logg, R_star used by TLS for more accurate transit models.
    """
    try:
        catalog_data = Catalogs.query_object(
            f"TIC {tic_id}", radius=0.001, catalog="TIC"
        )
        if len(catalog_data) == 0:
            print(f"[TIC {tic_id}] No TIC metadata found")
            return {"tic_id": tic_id}   # return minimal dict, not None

        star = catalog_data[0]
        return {
            "tic_id"          : int(tic_id),
            "Teff"            : float(star["Teff"]      if "Teff"      in star.colnames else np.nan),
            "logg"            : float(star["logg"]      if "logg"      in star.colnames else np.nan),
            "R_star"          : float(star["rad"]       if "rad"       in star.colnames else np.nan),
            "M_star"          : float(star["mass"]      if "mass"      in star.colnames else np.nan),
            "Tmag"            : float(star["Tmag"]      if "Tmag"      in star.colnames else np.nan),
            "crowding_metric" : float(star["contratio"] if "contratio" in star.colnames else np.nan),
        }

    except Exception as e:
        print(f"[TIC {tic_id}] Metadata fetch failed: {e}")
        return {"tic_id": int(tic_id)}   # never return None — Phase 1 expects a dict