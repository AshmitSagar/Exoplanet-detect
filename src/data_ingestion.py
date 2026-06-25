import os
import lightkurve as lk
from astroquery.mast import Catalogs
import numpy as np

def download_lightcurves(tic_id, sector=None, download_dir='data/raw'):
    """
    Download TESS-SPOC 2-minute PDCSAP flux for a given TIC ID and sector.
    """
    os.makedirs(download_dir, exist_ok=True)
    search_result = lk.search_lightcurve(f"TIC {tic_id}", mission="TESS", sector=sector, author="SPOC", exptime=120)
    
    if len(search_result) == 0:
        print(f"No SPOC 2-min data found for TIC {tic_id} in Sector {sector}.")
        return None
        
    # We download all matches (might be multi-sector if sector is not specified)
    lc_collection = search_result.download_all(download_dir=download_dir)
    
    if lc_collection is None or len(lc_collection) == 0:
        return None
        
    # Stitch if there are multiple sectors, otherwise return the single lightcurve
    if len(lc_collection) > 1:
        lc = lc_collection.stitch()
    else:
        lc = lc_collection[0]
        
    return lc

def fetch_tic_metadata(tic_id):
    """
    Fetch stellar parameters from the TIC catalog.
    Returns a dictionary of parameters like Teff, logg, R_star, etc.
    """
    try:
        catalog_data = Catalogs.query_object(f"TIC {tic_id}", radius=0.001, catalog="TIC")
        if len(catalog_data) > 0:
            # Get the first match
            star = catalog_data[0]
            metadata = {
                'tic_id': tic_id,
                'Teff': float(star.get('Teff', np.nan)),
                'logg': float(star.get('logg', np.nan)),
                'R_star': float(star.get('rad', np.nan)),
                'M_star': float(star.get('mass', np.nan)),
                'Tmag': float(star.get('Tmag', np.nan)),
                'crowding_metric': float(star.get('contratio', np.nan)), # Approximation for blending/crowding
            }
            return metadata
        else:
            print(f"No TIC metadata found for TIC {tic_id}")
            return None
    except Exception as e:
        print(f"Error fetching metadata for TIC {tic_id}: {e}")
        return None
