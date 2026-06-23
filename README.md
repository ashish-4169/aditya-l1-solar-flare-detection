# SoLEXS Solar Flare Detection Pipeline

> **Bharatiya Antariksh Hackathon 2026 — Problem Statement 15**  
> Solar Flare Detection & Forecasting using Aditya-L1 / SoLEXS Level-1 data

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![Astropy](https://img.shields.io/badge/Astropy-5.0%2B-orange)](https://www.astropy.org/)
[![Status](https://img.shields.io/badge/Milestone%201-Complete-brightgreen)]()

---

## Project Overview

This repository implements a robust, scientifically grounded solar flare
detection pipeline for data from the **Solar Low Energy X-ray Spectrometer
(SoLEXS)** aboard India's **Aditya-L1** spacecraft.

### Relation to Bharatiya Antariksh Hackathon 2026

Problem Statement 15 requires teams to:
1. Detect solar flares from SoLEXS Level-1 light curve data
2. Perform spectral analysis to characterise flares
3. Build a master multi-day flare catalog
4. Construct a forecasting feature dataset
5. Train and validate an ML-based flare prediction model

### Milestone 1 Objectives (Current)

- [x] Inspect and understand all FITS data products
- [x] Apply authoritative GTI masking to identify valid observation windows
- [x] Implement adaptive rolling median background estimation
- [x] Implement dynamic rolling MAD noise floor estimation
- [x] Detect flares using a 5σ consecutive-point trigger algorithm
- [x] Extract flare start, peak, end times, duration, fluence, and classification
- [x] Export flare catalog, intermediate products, and visualisation

### Current Status

**Milestone 1: COMPLETE**  
2 flare events detected and validated on the 2026-06-18 observation.
Full documentation, architecture diagrams, and code walkthrough included.

---

## Data Description

All data is from the Aditya-L1 SoLEXS Level-1 pipeline (version 1.4),
observation date **2026-06-18**, Observation ID `N00_0000_000934`.

### File Inventory

| File | Detector | Size | Scientific Role |
|:---|:---:|:---:|:---|
| `AL1_SOLEXS_20260618_SDD1_L1.gti(.gz)` | SDD1 | 0 rows | GTI file — **empty** (SDD1 inactive this day) |
| `AL1_SOLEXS_20260618_SDD2_L1.gti(.gz)` | SDD2 | 4 rows | Good Time Intervals — authoritative validity mask |
| `AL1_SOLEXS_20260618_SDD2_L1.lc(.gz)` | SDD2 | 86 400 rows | 1-second X-ray light curve — primary flare detection input |
| `AL1_SOLEXS_20260618_SDD2_L1.pi.gz` | SDD2 | 86 400 rows × 340 channels | Time-resolved energy spectra — Milestone 2 input |

### SDD1 GTI (`AL1_SOLEXS_20260618_SDD1_L1.gti`)

- **HDUs:** PRIMARY (metadata), GTI (BinTableHDU)
- **Columns:** `START` (int16), `STOP` (int16)
- **Rows:** 0
- **Scientific meaning:** Silicon Drift Detector 1 had no valid observation
  windows on this date. The file exists but contains no data. It is not used
  by the pipeline.

### SDD2 GTI (`AL1_SOLEXS_20260618_SDD2_L1.gti`)

- **HDUs:** PRIMARY (metadata), GTI (BinTableHDU)
- **Columns:** `START` (float64, Unix epoch s), `STOP` (float64, Unix epoch s)
- **Rows:** 4
- **Scientific meaning:** Defines the four intervals during which SDD2 was
  observing the Sun stably. Data outside these intervals is contaminated by
  Earth occultation, passage through the South Atlantic Anomaly (SAA), or
  detector calibration sequences.
- **Usage in pipeline:** Stage 2 — builds the authoritative `valid` boolean mask

| Interval | Start (UTC) | End (UTC) | Duration |
|:---:|:---:|:---:|:---:|
| 0 | 02:10:39 | 03:35:28 | 2 689 s |
| 1 | 03:35:30 | 13:51:55 | 39 385 s |
| 2 | 13:51:57 | 13:52:00 | 3 s |
| 3 | 13:52:02 | 23:59:59 | 36 477 s |

Total valid: **78 558 s** (90.9% duty cycle)

### SDD2 Light Curve (`AL1_SOLEXS_20260618_SDD2_L1.lc`)

- **HDUs:** PRIMARY (metadata), RATE (BinTableHDU)
- **Columns:** `TIME` (float64, Unix epoch s), `COUNTS` (float64, cts/s)
- **Rows:** 86 400 (one per second for the full 24-hour day)
- **Scientific meaning:** Total integrated X-ray count rate from the Sun,
  summed across all 340 energy channels. This is the primary observable for
  identifying solar flares as impulsive enhancements above the quiescent
  solar X-ray background.
- **Data quality:** 7 700 NaN values (8.9%) from GTI gaps
- **Count range:** 0–97 cts/s (mean 6.2 cts/s during valid intervals)
- **Usage in pipeline:** Stages 1–11 — all pipeline steps operate on this array

### SDD2 Spectral Data (`AL1_SOLEXS_20260618_SDD2_L1.pi.gz`)

- **HDUs:** PRIMARY (metadata), SPECTRUM (BinTableHDU)
- **Columns:**
  - `TSTART` (float64) — bin start time in Unix epoch seconds
  - `TELAPSE` (float64) — elapsed time per bin (1.0 s)
  - `SPEC_NUM` (int32) — sequential spectrum index 1–86 400
  - `CHANNEL` (340-element int64 array) — energy channel indices 0–339
  - `COUNTS` (340-element float64 array) — counts per energy channel per second
  - `EXPOSURE` (float64) — effective exposure per bin (1.0 s)
- **Rows:** 86 400 (one 340-channel spectrum per second)
- **Scientific meaning:** Time-resolved pulse height spectra enabling
  energy-resolved analysis — separation of soft and hard X-ray emission,
  temperature diagnostics, and hardness ratio computation.
- **Usage in pipeline:** Not used in Milestone 1. Reserved for Milestone 2.

---

## Pipeline Architecture

```
 AL1_SOLEXS_SDD2_L1.lc   AL1_SOLEXS_SDD2_L1.gti
         │                         │
         └──────────┬──────────────┘
                    ▼
            [1] FITS Loading
            Extract TIME, COUNTS arrays
            Read GTI START, STOP intervals
                    │
                    ▼
            [2] GTI Masking
            counts[outside GTI] = NaN
            Duty cycle: 90.9%
                    │
                    ▼
            [3] Adaptive Background Estimation
            rolling_median(3600 s) → quiet background
            rolling_median(7200 s) → active background
            Switch at local_std > 4 cts/s
                    │
                    ▼
            [4] Rolling MAD Noise Estimation
            sigma(t) = 1.4826 × rolling_MAD(|residuals|, 3600 s)
            Floor: sigma >= 0.5 cts/s
                    │
                    ▼
            [5] 5σ Threshold Generation
            threshold(t) = background(t) + 5 × sigma(t)
                    │
                    ▼
            [6] Consecutive-Point Detection
            Require >= 5 consecutive seconds above threshold
            Rejects: cosmic rays (1s), noise bursts (2-4s)
                    │
                    ▼
            [7] Gap Bridging
            Merge events separated by <= 10 s
                    │
                    ▼
            [8] Minimum Duration Filter
            Discard events with duration < 10 s
                    │
                    ▼
            [9] Parameter Extraction
            start, peak, end, fluence, duration
                    │
                    ▼
            [10] Flare Classification
            Weak / Moderate / Strong
            (by background-subtracted peak rate)
                    │
                    ▼
            [11] Catalog & Visualisation
            flare_catalog.csv  summary.json
            cleaned_lightcurve.csv  background.csv
            flare_plot.png
```

---

## Detection Parameters

| Parameter | Value | Scientific Justification |
|:---|:---:|:---|
| **Default background window** | 3 600 s | 1-hour window captures slow instrumental drift and solar quiescent variation without absorbing short C-class flares (typical duration 5–30 min) |
| **Adaptive background window** | 7 200 s | Activated during locally noisy periods; 2-hour window averages over more quiet data, providing a conservative lower background estimate |
| **Adaptive switch threshold** | local_std > 4 cts/s | Empirically found to separate quiet periods from activity periods in this dataset |
| **Sigma estimator** | MAD × 1.4826 | MAD is robust to flare-peak outliers; the 1.4826 scale converts MAD to equivalent Gaussian sigma (= 1/Φ⁻¹(0.75)) |
| **Sigma floor** | 0.5 cts/s | Prevents zero-sigma collapse during extremely quiet intervals |
| **Trigger threshold** | Background + 5σ | 5σ gives ~1 false positive per 1.7 million samples under Gaussian noise; chosen to eliminate spurious detections across a 24-hour observation |
| **Minimum consecutive points** | 5 s | Diagnostic analysis confirmed: cosmic rays last 1 s, Poisson bursts 2–4 s, smallest confirmed real flare was 19 s |
| **Gap bridge** | 10 s | Bridges counting-statistics dips within genuine flare events; shorter than the rise-to-peak timescale of even the weakest C-class flares |
| **Minimum event duration** | 10 s | Eliminates residual short-duration triggers that survive gap bridging |

---

## Results

### Summary Statistics

| Metric | Value |
|:---|:---|
| Observation date | 2026-06-18 |
| Detector | SDD2 |
| Total observation time | 86 400 s |
| Valid (in-GTI) time | 78 558 s (90.9%) |
| Average background rate | 4.59 cts/s |
| Average total count rate | 6.21 cts/s |
| **Flares detected** | **2** |

### Detected Flares

| ID | Class | Start (UTC) | Peak (UTC) | End (UTC) | Duration | Peak Raw | Peak Net | Fluence |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **1** | Strong | 2026-06-18 11:55:18 | 2026-06-18 11:56:49 | 2026-06-18 11:57:29 | **132 s** | 95.0 cts/s | **84.0 cts/s** | 8 034 cts |
| **2** | Strong | 2026-06-18 22:03:20 | 2026-06-18 22:03:27 | 2026-06-18 22:03:38 | **19 s** | 58.0 cts/s | **53.0 cts/s** | 770 cts |

### Why Some Peaks Were Rejected

Two visually prominent peaks at t≈27 916 s and t≈53 744 s were investigated
and confirmed correctly rejected:

- **t≈27 916 s** (07:45 UTC): 7.87σ amplitude — above threshold, but maximum
  consecutive run = **4 seconds** (< 5 required). Poisson-fluctuating interval.
- **t≈53 744 s** (14:55 UTC): 9.78σ amplitude — above threshold, but maximum
  consecutive run = **2 seconds**. Single-second spike consistent with cosmic
  ray / energetic particle hit.

---

## Output Files

All files are written to `outputs/` by the pipeline and regenerated on each run.

### `outputs/flare_catalog.csv`

One row per detected flare. 13 columns:

| Column | Type | Description |
|:---|:---:|:---|
| `START_IDX` | int | Index into time array |
| `END_IDX` | int | Index into time array |
| `START_TIME_EPOCH` | float | Unix epoch seconds |
| `START_TIME_UTC` | str | ISO-8601 UTC |
| `PEAK_TIME_EPOCH` | float | Unix epoch seconds |
| `PEAK_TIME_UTC` | str | ISO-8601 UTC |
| `END_TIME_EPOCH` | float | Unix epoch seconds |
| `END_TIME_UTC` | str | ISO-8601 UTC |
| `PEAK_RATE_RAW` | float | Maximum raw count rate (cts/s) |
| `PEAK_RATE_SUB` | float | Max background-subtracted rate (cts/s) |
| `FLUENCE` | float | Integrated net counts over event |
| `DURATION_SECONDS` | float | Event duration |
| `CLASSIFICATION` | str | Weak / Moderate / Strong |

### `outputs/summary.json`

JSON document with four top-level keys: `metadata`, `parameters`, `statistics`,
`flares`. The `flares` array contains the full event properties including
`"spectrum": null` (Milestone 2 hook). See
[`outputs/summary.json`](./outputs/summary.json).

### `outputs/cleaned_lightcurve.csv`

86 400-row time series with columns `TIME`, `COUNTS_RAW`, `COUNTS_CLEANED`.
`COUNTS_CLEANED` is empty (not "nan") for out-of-GTI samples.

### `outputs/background.csv`

86 400-row time series with columns `TIME`, `BACKGROUND`, `THRESHOLD_5SIGMA`.
Used to reproduce the detection logic without rerunning the pipeline.

### `outputs/flare_plot.png`

Annotated 15×7 inch light curve (150 dpi) showing:
- Light gray: raw counts
- Royal blue: GTI-cleaned counts
- Dark orange: adaptive background
- Forest green dashed: 5σ threshold
- Crimson shading: flare event windows
- Star markers: peak times
- Text labels: classification

---

## Quick Start

```bash
# 1. Clone / copy the project
cd solexs_2026Jun21T094306019

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run pipeline
python flare_detection.py

# 4. Check outputs
ls outputs/
```

---

## Future Work

### Milestone 2 — Spectral Analysis
- Load `AL1_SOLEXS_20260618_SDD2_L1.pi.gz` (340-channel time-resolved spectra)
- Define soft-band and hard-band energy channel ranges using SoLEXS response matrix
- Compute per-second hardness ratio: `HR = Hard / (Soft + Hard)`
- Attach spectral arrays to `FlareEvent.spectrum` for each detected flare
- Generate per-flare integrated spectra and export as CSV/FITS

### Milestone 3 — Master Catalog
- Generalise pipeline to process multiple observation dates in a loop
- Accumulate all detections into a multi-day master flare catalog
- Add cross-referencing with NOAA/GOES flare lists for validation

### Milestone 4 — Forecasting Dataset
- Extract pre-flare features: background trend, recent variability, hardness ratio evolution
- Label samples by flare onset using the master catalog
- Export as training-ready feature matrix

### Milestone 5 — ML Prediction
- Train binary flare/no-flare classifier (e.g. LightGBM, LSTM, Transformer)
- Evaluate on held-out dates
- Optimise for recall (missed flares more costly than false positives)

---

## Documentation Index

| Document | Purpose |
|:---|:---|
| [`README.md`](./README.md) | This file — project overview and quick start |
| [`architecture_diagram.md`](./architecture_diagram.md) | ASCII pipeline flow chart with stage I/O tables |
| [`TECHNICAL_DOCUMENTATION.md`](./TECHNICAL_DOCUMENTATION.md) | Functions, maths, complexity, assumptions |
| [`CODE_WALKTHROUGH.md`](./CODE_WALKTHROUGH.md) | Line-by-line explanation for new developers |
| [`TEAM_HANDOVER.md`](./TEAM_HANDOVER.md) | Project status, folder structure, run instructions |

---

## References

- ISRO Aditya-L1 Mission: https://www.isro.gov.in/Aditya_L1.html
- SoLEXS instrument: Solar Low Energy X-ray Spectrometer aboard Aditya-L1
- OGIP FITS Standards: https://heasarc.gsfc.nasa.gov/docs/heasarc/ofwg/
- Bharatiya Antariksh Hackathon 2026, Problem Statement 15

---

*Pipeline created for Bharatiya Antariksh Hackathon 2026.*
