# Compare README and PDF

> Optimize all of these to create the best plan. What are the differences between the readme and the given pdf. Tell the improvements that can be done in the README

The best combined plan should use the **README as the execution backbone** and borrow selected strengths from the **PDF**: more physics rationale, noise taxonomy, focal loss, synthetic transit injection, XGBoost/LightGBM ensemble option, Grad-CAM explainability, and explicit team roles.

**Best Optimized Plan**
Use a hybrid pipeline:

1. **Data**
   Use TESS PDCSAP light curves via `lightkurve`, plus TIC/CTL stellar metadata, ExoFOP-TESS, NASA Exoplanet Archive, Kepler DR25, and curated false-positive catalogues.

2. **Preprocessing**
   Apply NaN removal, asymmetric sigma clipping, normalization, and transit-preserving detrending using Wotan biweight or B-splines. Add centroid/crowding metadata where possible to catch blends.

3. **Detection**
   Run BLS as the fast baseline and TLS for high-quality final candidates. Report period, epoch, depth, duration, SDE, and SNR. Use iterative masking to find multi-planet systems.

4. **Classification**
   Use a multi-input model:
   `global phase view + local transit view + stellar parameters`.
   Best architecture: 1D CNN + attention, trained with focal loss for imbalance. Then either:
   - calibrated softmax for simpler implementation, or
   - XGBoost/LightGBM on learned embeddings for stronger tabular fusion.

5. **Parameter Estimation**
   For high-confidence transit candidates, fit `batman` transit models and estimate uncertainty with `emcee` MCMC. Report period, depth, duration, radius ratio, confidence interval, SNR, and residual quality.

6. **Outputs**
   Produce CSV results, candidate plots, phase-folded curves, model overlays, residual plots, probability bars, and optional Grad-CAM/attention maps. Dashboard is useful, but the 3-page report must stay the priority.

**README vs PDF Differences**

| Area | README | PDF |
|---|---|---|
| Style | Practical, structured, implementation-ready | Long-form, theoretical, narrative |
| Strength | Better roadmap, time budget, software stack, metrics, checklist, risks | Better physics explanation and problem justification |
| Pipeline | Clear 5-stage pipeline | Similar pipeline but less compact |
| Detection | Uses BLS and TLS, TLS preferred | Focuses mostly on BLS |
| Preprocessing | Wotan/GP detrending, sigma clipping, `.npz` storage | B-splines/Savitzky-Golay, HDF5 storage |
| Classification | CNN/attention, transfer learning, calibration, RF baseline | CNN/attention plus XGBoost/LightGBM ensemble and focal loss |
| Classes | PC, EB, Blend, Junk, Starspot | Transit, Eclipse, Blend, Other |
| Parameter fitting | `batman` + `emcee` MCMC, concrete settings | Bayesian/MCMC explanation, mentions NUTS |
| Visualization | Dash dashboard + static report plots | Adds residual plots and Grad-CAM explainability |
| Execution | More realistic 30-hour breakdown and pre-finale checklist | Adds team role division |
| References | More directly relevant but some links/IDs look malformed | More citations, but several are weak or unrelated |

**README Improvements**

1. Add a short **physics/noise background** from the PDF: transit depth, duration, limb darkening, instrumental noise, shot noise, stellar variability, blends.

2. Replace the 5-class output with a cleaner hackathon-aligned mapping:
   `Transit / Eclipsing Binary / Blend / Other`, with internal sublabels like `Starspot` and `Instrumental`.

3. Add **focal loss** for class imbalance. This is a strong missing piece.

4. Add **synthetic injection tests**: inject fake transits into real noisy TESS light curves to measure recovery rate by depth, period, and SNR.

5. Add **centroid/crowding checks** for blend detection using TIC/Gaia metadata.

6. Add **residual plots and Grad-CAM/attention maps** to make the AI explainable.

7. Add team roles for the finale:
   Data Engineer, ML Lead, Astrophysics/Validation Lead, Visualization/Report Lead.

8. Fix the references. Some README arXiv links are visibly malformed, such as `1712.05orbital` and `1712.05absolute`.

9. Make the plan more realistic by marking components as:
   `must-have`, `should-have`, and `stretch`.
   For a 30-hour sprint, dashboard, Optuna, deep transformers, and full MCMC for many candidates should be stretch items.

10. Add a final **minimum viable submission path**:
   BLS/TLS detection → CNN classifier → batman least-squares fit → CSV + plots + 3-page report. This protects the team if training or MCMC runs late.

Bottom line: the README is already the stronger document. The PDF adds useful scientific depth, but it is too verbose and less execution-focused. The best version is a tighter README that adds the PDF’s physics, focal loss, ensemble option, synthetic injection validation, centroid checks, explainability plots, and role-based finale execution.
