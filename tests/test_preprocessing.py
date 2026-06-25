import sys
import os
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from data_ingestion import download_lightcurves
from preprocessing import preprocess_lightcurve

def test_preprocessing_preserves_data():
    """
    Test that the preprocessing pipeline runs and returns valid numpy arrays.
    """
    # TOI-270 is TIC 259377017. Sector 3 is a known sector for it.
    tic_id = 259377017
    print(f"Downloading lightcurve for TIC {tic_id}...")
    lc = download_lightcurves(tic_id, sector=3)
    
    if lc is not None:
        print("Preprocessing lightcurve...")
        time, flux, flux_err, stats = preprocess_lightcurve(lc)
        
        # Check arrays are returned
        assert time is not None
        assert flux is not None
        assert flux_err is not None
        
        # Check arrays are same length
        assert len(time) == len(flux) == len(flux_err)
        
        # Check no NaNs
        assert not np.isnan(flux).any()
        assert not np.isnan(time).any()
        
        # Check normalized
        assert np.isclose(np.median(flux), 1.0, rtol=1e-2)
        
        # Check stats
        assert 'raw_points' in stats
        assert 'nans_removed' in stats
        assert 'outliers_removed' in stats
        assert 'final_points' in stats
        assert stats['raw_points'] > 0
        assert stats['final_points'] <= stats['raw_points']
        print("Test passed successfully.")
    else:
        print("Test skipped: No lightcurve found. (Check internet connection or MAST availability)")

if __name__ == "__main__":
    test_preprocessing_preserves_data()
