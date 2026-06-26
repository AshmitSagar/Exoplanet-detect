import os
import numpy as np
from joblib import Parallel, delayed
from data_ingestion import download_lightcurves, fetch_tic_metadata
from preprocessing import preprocess_lightcurve

def process_target(tic_id, sector=None, output_dir='data/processed', download_dir='data/raw'):
    """
    End-to-end Phase 1 processing for a single target.
    """
    try:
        # 1. Fetch Metadata
        metadata = fetch_tic_metadata(tic_id)
        if metadata is None:
            metadata = {'tic_id': tic_id}
            
        # 2. Download Lightcurve
        lc = download_lightcurves(tic_id, sector=sector, download_dir=download_dir)
        if lc is None:
            return False
            
        # 3. Preprocess
        time, flux, flux_err, stats = preprocess_lightcurve(lc)
        if time is None:
            return False
            
        # 4. Save to .npz
        sector_str = str(sector) if sector is not None else "all"
        output_path = os.path.join(output_dir, f"TIC_{tic_id}_sector_{sector_str}.npz")
        np.savez_compressed(
            output_path, 
            time=time, 
            flux=flux, 
            flux_err=flux_err, 
            metadata=metadata,
            stats=stats
        )
        print(f"Successfully processed and saved TIC {tic_id}")
        return True
        
    except Exception as e:
        print(f"Error processing TIC {tic_id}: {e}")
        return False

def run_phase1_pipeline(tic_ids, sector=1, n_jobs=-1):
    """
    Run the preprocessing pipeline in parallel.
    """
    os.makedirs('data/processed', exist_ok=True)
    
    sector_str = str(sector) if sector is not None else "all"
    print(f"Starting Phase 1 pipeline for {len(tic_ids)} targets in Sector {sector_str}...")
    
    results = Parallel(n_jobs=n_jobs)(
        delayed(process_target)(tic, sector) for tic in tic_ids
    )
    
    success_count = sum(1 for r in results if r)
    print(f"Pipeline finished. Successfully processed {success_count}/{len(tic_ids)} targets.")
    
if __name__ == "__main__":
    # Example test run with known TOIs
    # TOI-270 (TIC 259377017), TOI-700 (TIC 150428135)
    test_targets = [259377017, 150428135]
    run_phase1_pipeline(test_targets, sector=None, n_jobs=2)
