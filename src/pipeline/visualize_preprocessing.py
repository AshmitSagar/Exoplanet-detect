import os
import matplotlib.pyplot as plt
import numpy as np
from data_ingestion import download_lightcurves
from preprocessing import preprocess_lightcurve

def plot_comparison(tic_id, sector=None, output_dir='plots'):
    """
    Downloads a light curve, preprocesses it, and plots the raw vs preprocessed flux.
    Saves the plot as a PNG in the specified output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Downloading lightcurve for TIC {tic_id}...")
    lc = download_lightcurves(tic_id, sector=sector)
    
    if lc is None:
        print(f"Could not retrieve lightcurve for TIC {tic_id}")
        return
        
    print("Preprocessing lightcurve...")
    # Raw data for comparison
    raw_time = lc.time.value
    raw_flux = lc.flux.value
    # Remove NaNs from raw for plotting
    clean_mask = ~np.isnan(raw_flux) & ~np.isnan(raw_time)
    raw_time = raw_time[clean_mask]
    raw_flux = raw_flux[clean_mask]
    # Median normalize raw flux for direct comparison
    raw_flux = raw_flux / np.median(raw_flux)
    
    # Preprocessed data
    prep_time, prep_flux, prep_err, stats = preprocess_lightcurve(lc)
    
    if prep_time is None:
        print("Preprocessing returned empty array.")
        return
        
    # Plotting
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Top Plot: Raw Light Curve
    ax1.plot(raw_time, raw_flux, 'k.', alpha=0.3, label='Raw PDCSAP Flux')
    ax1.set_ylabel('Normalized Flux')
    ax1.set_title(f'TIC {tic_id} - Raw vs Preprocessed Light Curve')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # Add preprocessing stats text box
    stats_text = (
        f"Raw points:       {stats['raw_points']:,}\n"
        f"NaNs removed:     {stats['nans_removed']:,}\n"
        f"Outliers removed: {stats['outliers_removed']:,}\n"
        f"Final points:     {stats['final_points']:,}"
    )
    ax1.text(0.02, 0.95, stats_text, transform=ax1.transAxes, verticalalignment='top',
             fontfamily='monospace', fontsize=10,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='#f5f5f5', alpha=0.8, edgecolor='gray'))
    
    # Bottom Plot: Preprocessed (Cleaned & Detrended) Light Curve
    ax2.plot(prep_time, prep_flux, 'b.', alpha=0.3, label='Cleaned & Detrended Flux (Wotan)')
    ax2.set_xlabel('Time (TBJD)')
    ax2.set_ylabel('Normalized Detrended Flux')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    sector_str = str(sector) if sector is not None else "all"
    filename = os.path.join(output_dir, f"TIC_{tic_id}_sector_{sector_str}_preprocess.png")
    plt.savefig(filename, dpi=150)
    plt.close()
    
    print(f"Saved visualization plot to: {filename}")

if __name__ == "__main__":
    # Test with a known TOI: TOI-270 (TIC 259377017) in Sector 3
    plot_comparison(259377017, sector=3)
