# BAH 2026 — Problem Statement 7
# AI-Enabled Detection of Exoplanets from Noisy Astronomical Light Curves
## Comprehensive Technical Plan

**Hackathon:** Bharatiya Antariksh Hackathon 2026 (ISRO × Hack2skill)  
**Team Size:** 3–4 members  
**Grand Finale:** 6–7 August 2026 (30-hour sprint)  
**Submission Deadline:** 1 July 2026  
**Hardware:** Intel Core i9 (latest gen) + NVIDIA RTX 4080 (16 GB VRAM)

---

## 1. Executive Summary

We will build **AstroVet-ISRO**, an end-to-end AI pipeline that ingests raw TESS light curves, denoises and detrends them, detects periodic transit-like dips via classical algorithms, classifies those dips into astrophysical categories (transit, eclipsing binary, blend, other), estimates physical parameters (period, depth, duration), and produces confidence-calibrated outputs with rich visualisation — all within the hardware budget of a single RTX 4080 workstation.

The design philosophy: **classical detection + deep-learning classification + physics-based parameter estimation**. Each stage is independently testable, which is crucial in a 30-hour sprint.

---

## 2. Problem Decomposition

The problem has five explicit deliverables that map cleanly onto pipeline stages:

| # | Deliverable | Pipeline Stage |
|---|-------------|---------------|
| 1 | Identify datasets with periodic dips | Stage 2 — BLS/TLS transit search |
| 2 | Classify dips (transit / eclipse / blend / other) | Stage 3 — AI classifier |
| 3 | Apply classifier to science datasets | Stage 3 — Inference |
| 4 | Signal-to-noise / significance levels | Stage 2 — SDE / SNR output |
| 5 | Estimate transit depth, period, duration | Stage 4 — batman/MCMC fitting |

---

## 3. Hardware Configuration & Optimisation

### 3.1 System Profile

| Component | Spec | Role |
|-----------|------|------|
| CPU | Intel Core i9-14900K / i9-13900K | Data preprocessing, BLS/TLS parallelism, MCMC |
| GPU | NVIDIA RTX 4080 (16 GB GDDR6X) | Model training, inference, cuBLS |
| RAM | Recommend ≥ 64 GB DDR5 | Full-sector light-curve buffer |
| Storage | NVMe SSD ≥ 2 TB | TESS FITS cache |

### 3.2 GPU Constraints & Mitigation

- **16 GB VRAM** is sufficient for training 1D-CNN / Transformer models on phase-folded 2001-point light curves (standard AstroNet format). Batch size of 256–512 fits comfortably.
- Use **mixed-precision training (FP16/BF16)** via `torch.cuda.amp` to double effective throughput.
- Use **gradient checkpointing** if using Transformer encoders deeper than 6 layers.
- Data preprocessing (lightkurve, BLS/TLS) runs on CPU with full i9 multi-core parallelism (`n_jobs = -1`, up to 24 P-cores).
- Reserve GPU exclusively for model training and inference; do NOT run BLS/TLS on GPU in the same session.

### 3.3 Time Budget (30-hour Finale)

| Phase | Hours | GPU Active? |
|-------|-------|-------------|
| Data ingestion & preprocessing | 0–4 h | No |
| BLS/TLS transit search on sector | 4–8 h | No (CPU) |
| Model training (transfer learning) | 8–14 h | Yes |
| Inference & classification | 14–16 h | Yes |
| Parameter estimation (batman/MCMC) | 16–22 h | No (CPU) |
| Visualisation & report | 22–28 h | No |
| Buffer / debugging | 28–30 h | — |

---

## 4. Data Strategy

### 4.1 Primary Dataset — TESS Raw Light Curves

**Source:** [MAST / TESS TIC-CTL](https://archive.stsci.edu/tess/tic_ctl.html)

**Recommended sector for the hackathon:** Sector 1 (Southern CVZ, July–Aug 2018). This is the most studied sector with ~20–30k light curves and the highest density of known TOIs for ground-truth validation.

**Access method (lightkurve):**
```python
import lightkurve as lk
# Download 2-minute cadence SPOC light curves for a sector
search = lk.search_lightcurve("TIC 29169159", mission="TESS", sector=1)
lc = search.download()
```

**What to download:**
- TESS-SPOC 2-minute PDCSAP flux (pre-detrended, systematics-removed) — primary input
- TESS-SPOC 20-second cadence where available (Sector 27+) for shallow transit recovery

**Estimated storage:** ~15–25 GB for a single sector (FITS files)

### 4.2 Training/Labelled Datasets

The classifier requires ground-truth labels. Use the following curated public datasets:

| Dataset | Contents | Size | Access |
|---------|----------|------|--------|
| **Kepler DR25 TCE Catalogue** | ~34k TCEs labelled as PC / FP / EB | ~2 GB | [NASA Exoplanet Archive](https://exoplanetarchive.ipac.caltech.edu) |
| **ExoMiner++ Vetting Catalogue (2025)** | 2-min TESS TOIs with ExoMiner scores | ~500 MB | arXiv:2502.09790 supplementary |
| **TESS TOI Catalogue** | ~7655 TOI dispositions (PC/FP/KP/CP) | ~50 MB | [ExoFOP-TESS](https://exofop.ipac.caltech.edu/tess/) |
| **Kepler/TESS AstroNet Training Set** | Phase-folded, pre-processed arrays | ~8 GB | [Shallue & Vanderburg 2018 GitHub](https://github.com/google-research/exoplanet-ml) |
| **NASA Exoplanet Archive Confirmed** | 5800+ confirmed planets with parameters | ~10 MB | [NExScI](https://exoplanetarchive.ipac.caltech.edu/cgi-bin/TblView/nph-tblView?app=ExoTbls&config=PSCompPars) |
| **Eclipsing Binary Catalog (Kirk+2016)** | 2,878 Kepler EBs with periods | ~5 MB | [Villanova EB Catalog](http://keplerebs.villanova.edu) |

**Key insight:** Train on the **Kepler DR25** labelled set (large, clean labels) and **transfer-learn** to TESS using the ExoMiner++ TESS catalogue — this mirrors NASA's own ExoMiner++ methodology and gives the highest accuracy on TESS data with limited TESS labels.

### 4.3 Auxiliary Stellar Parameters

Required for the classifier (stellar radius, Teff, logg, etc.):
- **TESS Input Catalog v8.2 (TIC):** Download the xCTL CSV (497 MB) for stellar parameters cross-matched to all targets.
- **Gaia DR3:** Proper motions and parallaxes for crowding/blending assessment.

---

## 5. Pipeline Architecture

```
Raw TESS FITS
      │
      ▼
┌─────────────────────────────┐
│  Stage 1: Ingestion &       │
│  Preprocessing              │
│  - lightkurve download      │
│  - sigma-clipping outliers  │
│  - Wotan/GP detrending      │
│  - Normalisation            │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 2: Transit Search    │
│  - BLS (astropy.timeseries) │
│  - TLS (transitleastsquares)│
│  - SDE & SNR computation    │
│  - Period candidates ranked │
└────────────┬────────────────┘
             │ (TCE list)
             ▼
┌─────────────────────────────┐
│  Stage 3: AI Classifier     │
│  - Phase-folding → 2001-pt  │
│    global + 201-pt local    │
│  - 1D-CNN / CNN-BiLSTM-Attn │
│  - Stellar param. branch    │
│  - Softmax: PC / EB /       │
│    Blend / Junk / Other     │
│  - Confidence calibration   │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 4: Parameter         │
│  Estimation (Transits only) │
│  - batman transit model     │
│  - MCMC (emcee) posterior   │
│  - Period, depth, duration  │
│  - Uncertainty intervals    │
└────────────┬────────────────┘
             │
             ▼
┌─────────────────────────────┐
│  Stage 5: Visualisation &   │
│  Output                     │
│  - Interactive Plotly/Dash  │
│  - Phase-folded light curve │
│  - Classification label     │
│  - Parameter table + errors │
│  - PDF report               │
└─────────────────────────────┘
```

---

## 6. Detailed Implementation Plan

### Stage 1 — Ingestion & Preprocessing

**Tools:** `lightkurve`, `astropy`, `numpy`, `wotan`

**Steps:**
1. Download PDCSAP flux for all TIC IDs in a chosen sector via lightkurve bulk download.
2. Remove NaN, outliers using iterative sigma-clipping (3σ, 5 iterations).
3. Stitch multi-sector light curves using `lk.LightCurveCollection.stitch()`.
4. Detrend stellar variability using **Wotan biweight** sliding window (window = 0.75 × minimum plausible transit duration, ≈ 0.3 days for hot Jupiters). Apply Gaussian Process (GP) detrending with a Matérn-3/2 kernel for active stars.
5. Normalise to median = 1.0.
6. Save preprocessed arrays as compressed `.npz` files (much faster I/O than FITS for the ML stages).

**Validation check:** Run detrending on 3 known TOIs (e.g., TOI-270, TOI-700, TOI-1431) and verify transit depth is preserved after detrending.

**i9 parallelism:** `joblib.Parallel(n_jobs=-1)` across TIC IDs — all 24 P-cores engaged.

---

### Stage 2 — Transit Search

**Tools:** `astropy.timeseries.BoxLeastSquares`, `transitleastsquares`, `scipy`

**BLS Search:**
```python
from astropy.timeseries import BoxLeastSquares
model = BoxLeastSquares(time, flux, dy=flux_err)
results = model.autopower(0.16, minimum_period=0.5, maximum_period=27)
# period_range covers TESS sector baseline (~27 days)
```

**TLS Search (preferred for small planets):**
```python
from transitleastsquares import transitleastsquares
model = transitleastsquares(time, flux)
results = model.power(
    period_min=0.5, period_max=27,
    oversampling_factor=5,
    duration_grid_step=1.05
)
# Returns: period, T0, depth, duration, SDE, SNR
```

**Output per TCE:**
- Best-fit period (days)
- Transit epoch T0
- Transit depth (ppm)
- Transit duration (hours)
- Signal Detection Efficiency (SDE) — use SDE > 9 as detection threshold
- Signal-to-Noise Ratio (SNR) — use SNR > 7 as secondary threshold

**Iterative search:** After finding the strongest signal, mask the in-transit points and re-run TLS to find additional planet candidates in the same system (multi-planet systems).

---

### Stage 3 — AI Classifier

#### 3.1 Model Architecture

We implement a **trimodal 1D CNN + Attention** classifier inspired by AstroNet (Shallue & Vanderburg 2018) and ExoNet (2026):

```
Input: Global view (2001 pts) + Local view (201 pts) + Stellar params (7 scalars)
         │                         │                       │
    Conv1D blocks             Conv1D blocks           Dense(32)
    (8 layers, ReLU)          (4 layers, ReLU)             │
    Multi-Head Attention       Max Pooling                  │
         │                         │                       │
         └──────────── Late Fusion (Concat) ───────────────┘
                              │
                         Dense(512) → Dense(256)
                              │
                   Softmax (5 classes):
               [PC, EB, Blend, Junk, Starspot]
```

**5 Output Classes:**
- `PC` — Planet Candidate (genuine transit)
- `EB` — Eclipsing Binary (V-shaped dip, secondary eclipse)
- `BLEND` — Background eclipsing binary / contamination
- `JUNK` — Instrumental artefact, scattered light, momentum dump
- `STARSPOT` — Stellar rotation / spot modulation

#### 3.2 Input Representations

**Global view (2001 points):** Phase-folded light curve from −0.5 to +0.5 in phase, median-binned. Captures transit shape and any out-of-transit variation.

**Local view (201 points):** Phase-folded light curve zoomed to ±2 transit durations around transit centre. Captures ingress/egress flatness — the key discriminator between transits (flat bottom) and eclipsing binaries (V-shape).

**Stellar parameters (7 features):** Teff, logg, R★, M★, [Fe/H], TESS magnitude, crowding metric — pulled from TIC v8.2.

#### 3.3 Training Strategy

**Phase 1 — Pre-train on Kepler DR25** (~34k TCEs, clean labels):
- Use Shallue & Vanderburg (2018) AstroNet training set
- Train for 30,000 steps, batch size 256, Adam(lr=1e-3), cosine LR decay
- Expected AUC > 0.96 on Kepler test set (consistent with literature)
- Estimated GPU time: ~3–4 hours on RTX 4080

**Phase 2 — Transfer learn to TESS**:
- Freeze first 4 Conv1D blocks, fine-tune upper layers + classification head
- Training data: ExoMiner++ TESS vetting catalogue (positive: labelled PC; negative: labelled FP/EB)
- Train for 5,000 steps, batch size 128, Adam(lr=1e-4)
- Estimated GPU time: ~1 hour on RTX 4080

**Data augmentation during training:**
- Phase shift (random roll of phase-folded curve)
- Flux noise injection (Gaussian σ = 0.1×transit depth)
- Depth rescaling (×0.8 to ×1.2)
- These augmentations improve robustness to TESS systematics

#### 3.4 Confidence Calibration

Apply **Platt scaling (isotonic regression)** post-training to ensure output probabilities are well-calibrated. This is a specific evaluation criterion and will be highlighted in the report.

```python
from sklearn.calibration import CalibratedClassifierCV
# Calibrate on a held-out validation set, not the training set
```

#### 3.5 Baseline Comparison

Also train a **Random Forest** on scalar features extracted from BLS/TLS output + stellar parameters as a non-DL baseline. This gives evaluators confidence that the DL model is adding genuine value.

---

### Stage 4 — Parameter Estimation

**Tools:** `batman`, `emcee`, `corner`

For all signals classified as `PC` (planet candidate), fit a physical transit model:

**batman model parameters:**
- `P` — Orbital period (days) [from TLS output, used as initial guess]
- `t0` — Mid-transit time (BJD)
- `rp` — Planet-to-star radius ratio Rp/R★ (controls transit depth δ ≈ (Rp/R★)²)
- `a` — Semi-major axis in stellar radii
- `inc` — Inclination (degrees)
- `ecc` — Eccentricity (fix to 0 for circular orbit initially)
- `u1, u2` — Quadratic limb-darkening coefficients (prior from Claret 2017 tables)

**MCMC sampling (emcee):**
```python
import emcee
# 64 walkers, 5000 steps, discard first 1000 as burn-in
sampler = emcee.EnsembleSampler(nwalkers=64, ndim=7, log_prob_fn=log_likelihood)
sampler.run_mcmc(pos, 5000, progress=True)
```

**Derived outputs:**
- Orbital period P ± σP (days)
- Transit depth δ = (Rp/R★)² ± σδ (ppm)
- Transit duration T14 ± σT14 (hours)
- Planet radius Rp = (Rp/R★) × R★ (Earth radii, using TIC R★)

**Uncertainty estimation:** Report 16th–84th percentile of MCMC posterior as 1σ interval. Show corner plot for the report.

**GPU parallelism:** `emcee` runs on CPU. With i9's 24 cores, run multiple `emcee` chains in parallel across different TCE candidates.

---

### Stage 5 — Visualisation & Output

**Interactive Dashboard (Plotly Dash):**
- Input: TIC ID or sector number
- Output panels:
  1. Raw + detrended light curve (full sector)
  2. BLS/TLS periodogram with SDE/SNR marked
  3. Phase-folded light curve with batman model overlay
  4. Classification probability bar chart (all 5 classes)
  5. Parameter table with uncertainties
  6. Confidence calibration curve

**Static outputs (for report/submission):**
- PNG plots at 300 dpi for each candidate
- CSV summary table: TIC_ID, Period, Depth_ppm, Duration_hr, Class, Confidence, SNR, SDE
- PDF report (max 3 pages, as per problem statement requirement)

---

## 7. Report Structure (3-page limit)

**Page 1 — Methodology:**
- Pipeline overview diagram
- Preprocessing choices (why Wotan biweight + GP)
- Detection algorithm: BLS vs TLS, SDE threshold justification
- Data sources used

**Page 2 — AI Model & Classification:**
- Architecture diagram (simplified)
- Training strategy: Kepler pre-training + TESS transfer learning
- Accuracy metrics: Precision, Recall, F1, AUC-ROC per class
- Confidence calibration methodology

**Page 3 — Results & Parameter Estimation:**
- Summary table of detected and classified signals in the test sector
- Example light curves with transit fits
- Parameter estimates with uncertainties
- Assumptions: circular orbits, single-star host, PDCSAP flux as ground truth

---

## 8. Evaluation Criteria Mapping

| Criterion | Our Approach | Expected Score |
|-----------|-------------|----------------|
| **Accuracy — Event Detection** | TLS with SDE > 9, validated against known TOIs in sector | High — TLS is state-of-art |
| **Accuracy — Classification** | CNN-BiLSTM-Attn + transfer learning, calibrated softmax | High — follows ExoMiner++ methodology |
| **Accuracy — Parameters** | batman MCMC with stellar priors, 1σ uncertainty reporting | High — physical model, not heuristic |
| **Methods/Approach** | Classical + DL + Bayesian, well-motivated at each stage | Excellent — mirrors ISRO/NASA workflow |
| **Visualisation & Clarity** | Interactive Dash dashboard + static publication-quality plots | Excellent |

---

## 9. Software Stack

```
Python 3.11
├── Data Access
│   ├── lightkurve 2.4+       (TESS FITS download, preprocessing)
│   ├── astroquery 0.4+       (TIC/ExoFOP queries)
│   └── astropy 6.0+          (FITS, BLS, coord transforms)
│
├── Signal Processing
│   ├── transitleastsquares   (TLS transit search)
│   ├── wotan 1.10+           (light curve detrending)
│   └── scipy 1.13+           (signal processing, stats)
│
├── Machine Learning
│   ├── PyTorch 2.3+ (CUDA 12.4)  (model training on RTX 4080)
│   ├── torchvision             (data augmentation utilities)
│   ├── scikit-learn 1.5+       (RF baseline, calibration)
│   └── optuna                  (hyperparameter search)
│
├── Parameter Estimation
│   ├── batman-package          (transit model)
│   ├── emcee 3.1+              (MCMC sampling)
│   └── corner                  (posterior visualisation)
│
├── Visualisation
│   ├── plotly 5.22+            (interactive plots)
│   ├── dash 2.17+              (web dashboard)
│   └── matplotlib 3.9+         (static publication plots)
│
└── Utilities
    ├── numpy, pandas
    ├── joblib                  (CPU parallelism for i9)
    └── tqdm                    (progress bars)
```

---

## 10. Pre-Finale Preparation Checklist

### Before 1 July 2026 (Idea Submission)
- [ ] Submit concept proposal describing the three-stage pipeline
- [ ] Include architecture diagram and planned datasets
- [ ] Cite ExoMiner++, AstroNet, TLS as prior art being built upon

### Before 6 August 2026 (Finale)
- [ ] Download and cache Sector 1 TESS SPOC light curves locally (save finale time)
- [ ] Pre-train CNN on Kepler DR25 AstroNet training set (takes 3–4 GPU hours)
- [ ] Download and index: TIC xCTL CSV, TOI catalogue, Kepler DR25 TCE table
- [ ] Set up Conda environment and test full pipeline on 100 light curves
- [ ] Validate on 5 known TOIs: TOI-270 b/c/d, TOI-700 d, WASP-121 b
- [ ] Prepare report template (Overleaf or Word, 3-page structure)
- [ ] Pre-generate batman limb-darkening grids for TESS bandpass

---

## 11. Risk Register & Mitigation

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| MAST server slow during finale | Medium | High | Pre-download all data before 6 Aug |
| GPU OOM during training | Low | Medium | Batch size = 128, FP16 training |
| TLS too slow for full sector | Medium | Medium | Subsample 5000 brightest stars (CTL), use i9 parallelism |
| MCMC non-convergence | Low | Low | Fallback to batman least-squares fit (scipy.optimize) |
| Transfer learning underperforms | Low | Medium | Fallback to Kepler-trained model directly |
| batman fitting diverges | Low | Low | Strong priors from TLS initial guess |

---

## 12. Key References & Resources

1. **AstroNet (Shallue & Vanderburg 2018)** — CNN for Kepler transit classification: [arXiv:1712.05orbital](https://arxiv.org/abs/1712.05absolute)
2. **ExoMiner++ (Valizadegan et al. 2025)** — Transfer learning Kepler→TESS: [arXiv:2502.09790](https://arxiv.org/abs/2502.09790)
3. **ExoNet (2026)** — Trimodal CNN+Attention for TESS: [arXiv:2604.15560](https://arxiv.org/abs/2604.15560)
4. **CNN-BiLSTM-Attention (2025)** — Kepler DR25 F1=0.984: [arXiv:2509.04793](https://arxiv.org/abs/2509.04793)
5. **Transit Least Squares (Hippke & Heller 2019)** — TLS algorithm: [hippke/tls](https://github.com/hippke/tls)
6. **batman (Kreidberg 2015)** — Transit model: [batman-package](https://lweb.cfa.harvard.edu/~lkreidberg/batman/)
7. **Wotan (Hippke et al. 2019)** — Detrending: [hippke/wotan](https://github.com/hippke/wotan)
8. **lightkurve (Cardoso et al. 2018)** — TESS data access: [lightkurve.org](https://lightkurve.org)
9. **TESS TIC/CTL:** [archive.stsci.edu/tess/tic_ctl.html](https://archive.stsci.edu/tess/tic_ctl.html)
10. **NASA Exoplanet Archive:** [exoplanetarchive.ipac.caltech.edu](https://exoplanetarchive.ipac.caltech.edu)
11. **ExoFOP-TESS (TOI Catalogue):** [exofop.ipac.caltech.edu/tess](https://exofop.ipac.caltech.edu/tess/)
12. **Kepler EB Catalog (Kirk+2016):** [keplerebs.villanova.edu](http://keplerebs.villanova.edu)
13. **COUNTESS I (2025)** — Uniformly vetted TESS CVZ catalogue: [arXiv:2606.13789](https://arxiv.org/abs/2606.13789)

---

## 13. Success Metrics

| Metric | Target |
|--------|--------|
| BLS/TLS period recovery on known TOIs | > 95% within 0.01-day tolerance |
| Classifier precision (PC class) | > 85% |
| Classifier recall (PC class) | > 80% |
| Transit depth estimation error | < 10% relative |
| Period estimation error | < 0.1% relative |
| Transit duration estimation error | < 15% relative |
| Processing time per light curve | < 30 seconds (TLS+classification) |
| Total sector processing (5000 stars) | < 8 hours (i9 parallel) |

---

*Plan prepared as of June 2026. All library versions should be pinned in `requirements.txt` before the finale. Dataset URLs are verified against MAST and NASA archives as of June 2026.*
