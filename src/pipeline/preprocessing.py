import numpy as np
import wotan
from astropy.stats import sigma_clip

def preprocess_lightcurve(lc, window_length=0.3, sigma_upper=3.0, sigma_lower=3.0, iters=5, detrend_method='biweight'):
    """
    Cleans and detrends a TESS light curve.
    
    Steps:
    1. Remove NaNs
    2. Iterative sigma clipping to remove outliers
    3. Detrending using wotan (preserves transits)
    4. Median normalization (done inherently by wotan.flatten)
    
    Returns: 
    time, flux, flux_err (as numpy arrays), stats (dict)
    """
    # Track raw points before cleaning
    raw_points = len(lc.flux)
    
    # 1. Remove NaNs
    lc_nonan = lc.remove_nans()
    nans_removed = raw_points - len(lc_nonan.flux)
    
    # Convert to standard numpy arrays to avoid astropy masked array issues with sigma_clip
    time = np.asarray(lc_nonan.time.value, dtype=float)
    flux = np.asarray(lc_nonan.flux.value, dtype=float)
    flux_err = np.asarray(lc_nonan.flux_err.value, dtype=float)
    
    # 2. Sigma Clipping (remove outliers)
    clipped_flux = sigma_clip(flux, sigma_lower=sigma_lower, sigma_upper=sigma_upper, maxiters=iters)
    
    # Create mask of valid data points
    mask = ~clipped_flux.mask
    outliers_removed = int(np.sum(clipped_flux.mask))
    final_points = int(np.sum(mask))
    
    stats = {
        'raw_points': raw_points,
        'nans_removed': nans_removed,
        'outliers_removed': outliers_removed,
        'final_points': final_points
    }
    
    print("\n--- Preprocessing Stats ---")
    print(f"Raw points       : {stats['raw_points']:,}")
    print(f"NaNs removed     : {stats['nans_removed']:,}")
    print(f"Outliers removed : {stats['outliers_removed']:,}")
    print(f"Final points     : {stats['final_points']:,}")
    print("----------------------------\n")
    
    time_clean = time[mask]
    flux_clean = flux[mask]
    err_clean = flux_err[mask]
    
    if len(time_clean) == 0:
        return None, None, None, stats
        
    # 3 & 4. Detrending and Normalization using wotan
    try:
        # Wotan flatten automatically normalizes the flux (median ≈ 1.0)
        flattened_flux, trend_flux = wotan.flatten(
            time_clean, 
            flux_clean, 
            method=detrend_method, 
            window_length=window_length, 
            return_trend=True
        )
        
        # Normalize the error array by the trend to maintain SNR
        normalized_err = err_clean / trend_flux
        
        # Ensure median is exactly 1.0 just to be safe
        median_flux = np.median(flattened_flux)
        flattened_flux = flattened_flux / median_flux
        normalized_err = normalized_err / median_flux
        
        return time_clean, flattened_flux, normalized_err, stats
        
    except Exception as e:
        print(f"Error during detrending: {e}")
        # Fallback: just return median normalized raw flux
        median_flux = np.median(flux_clean)
        return time_clean, flux_clean / median_flux, err_clean / median_flux, stats

