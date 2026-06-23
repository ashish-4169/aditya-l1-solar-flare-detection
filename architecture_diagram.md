# SoLEXS Flare Detection — Architecture Diagram

## System Overview

This document shows every processing stage of the Milestone 1 pipeline,
its inputs, its outputs, and the design rationale for each block.

```
╔══════════════════════════════════════════════════════════════════════════════╗
║            ADITYA-L1 / SoLEXS  —  Milestone 1 Pipeline Architecture        ║
╚══════════════════════════════════════════════════════════════════════════════╝

 ┌─────────────────────────────────────────────────────────────────────────┐
 │                         RAW INPUT FILES (FITS)                          │
 │                                                                         │
 │  AL1_SOLEXS_20260618_SDD2_L1.lc   ──► RATE HDU: 86 400 rows × 2 cols  │
 │  AL1_SOLEXS_20260618_SDD2_L1.gti  ──► GTI  HDU:      4 rows × 2 cols  │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 1 — FITS LOADER  (load_solexs_data)                             │
 │                                                                         │
 │  Read .lc FITS: TIME[86400], COUNTS[86400]  — 1 row = 1 second        │
 │  Read .gti FITS: START[4], STOP[4]          — valid observation gaps   │
 │  Convert all arrays to native float64 (endianness fix for Pandas)      │
 │  Extract metadata: MISSION, INSTRUME, OBS_DATE, OBS_ID                 │
 │                                                                         │
 │  OUT: time[86400], counts_raw[86400], gti_starts[4], gti_stops[4]     │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 2 — GTI AUTHORITATIVE MASKING  (load_solexs_data)               │
 │                                                                         │
 │  Build valid[86400] = False everywhere                                  │
 │  For each (START, STOP) in GTI:                                         │
 │      valid[time >= START and time <= STOP] = True                      │
 │  counts_cleaned = counts_raw.copy()                                     │
 │  counts_cleaned[~valid] = NaN          ← masks ~7 842 s out of 86 400 │
 │                                                                         │
 │  GTI Intervals (SDD2, 2026-06-18):                                     │
 │    02:10:39–03:35:28  (2 689 s)                                        │
 │    03:35:30–13:51:55  (39 385 s)                                       │
 │    13:51:57–13:52:00  (3 s)                                            │
 │    13:52:02–23:59:59  (36 477 s)                                       │
 │    Total valid: 78 558 s  (90.9% duty cycle)                           │
 │                                                                         │
 │  OUT: counts_cleaned[86400]  (NaN outside GTI)                         │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 3 — ADAPTIVE BACKGROUND ESTIMATION                              │
 │            (estimate_background_adaptive)                               │
 │                                                                         │
 │  bg_3600 = rolling_median(counts_cleaned, window=3600 s)               │
 │  bg_7200 = rolling_median(counts_cleaned, window=7200 s)               │
 │                                                                         │
 │  residuals_3600 = counts_cleaned - bg_3600                             │
 │  local_std      = rolling_std(residuals_3600, window=3600 s)           │
 │                                                                         │
 │  Adaptive selector:                                                     │
 │    where local_std > 4.0:  background = bg_7200  (active / noisy)     │
 │    elsewhere:              background = bg_3600  (quiet)               │
 │                                                                         │
 │  Rationale: 3 600 s captures slowly varying instrumental drift;        │
 │  7 200 s prevents C-class flares (10–30 min) from biasing background  │
 │                                                                         │
 │  OUT: background[86400]  (gap-interpolated, NaN-free)                  │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 4 — DYNAMIC NOISE ESTIMATION  (estimate_dynamic_sigma)          │
 │                                                                         │
 │  residuals     = counts_cleaned - background                            │
 │  abs_residuals = |residuals|                                            │
 │  rolling_MAD   = rolling_median(abs_residuals, window=3600 s)          │
 │  sigma         = 1.4826 × rolling_MAD                                  │
 │  sigma         = max(sigma, 0.5)     ← floor prevents hypersensitivity │
 │                                                                         │
 │  Scale factor 1.4826: converts MAD to equivalent Gaussian sigma        │
 │  for a normal distribution (sigma = MAD / 0.6745)                     │
 │                                                                         │
 │  OUT: sigma[86400]  (time-varying noise estimate)                       │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 5 — 5-SIGMA THRESHOLD GENERATION                                │
 │                                                                         │
 │  threshold[t] = background[t] + 5.0 × sigma[t]                        │
 │                                                                         │
 │  This is a time-varying upper limit, not a fixed value.                │
 │  Typical range: 18–30 cts/s during quiet periods of this observation   │
 │                                                                         │
 │  OUT: threshold[86400]                                                  │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 6 — CONSECUTIVE-POINT TRIGGER DETECTION                         │
 │            (detect_flares_consecutive)                                  │
 │                                                                         │
 │  above[t] = (counts[t] - background[t] > 5 × sigma[t])                │
 │           AND NOT NaN(counts[t])                                        │
 │                                                                         │
 │  Run-length algorithm:                                                  │
 │    run_length[t] = consecutive seconds of above=True ending at t       │
 │    valid_trigger[t] = True  iff  run_length[t] >= 5                    │
 │                                                                         │
 │  Rationale: requires 5 s persistence to reject:                        │
 │    • Poisson noise spikes  (typically 1–2 s above threshold)           │
 │    • Cosmic ray / proton hits  (single-second, very high amplitude)    │
 │    • Detector read-out transients                                       │
 │                                                                         │
 │  OUT: valid_trigger[86400]  (boolean)                                   │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 7 — GAP BRIDGING & EVENT MERGING                                │
 │            (merge_and_filter_events)                                    │
 │                                                                         │
 │  Find contiguous trigger blocks from valid_trigger                      │
 │  For adjacent blocks (A, B):                                            │
 │    if gap(A.end, B.start) <= 10 s:  merge into single event            │
 │                                                                         │
 │  Rationale: flares occasionally dip below threshold briefly due to     │
 │  counting statistics; 10 s bridges these sub-threshold dropouts        │
 │                                                                         │
 │  OUT: merged_blocks  (list of index pairs)                              │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 8 — MINIMUM DURATION FILTER                                     │
 │            (merge_and_filter_events, continued)                         │
 │                                                                         │
 │  For each merged event:                                                 │
 │    duration = time[end_idx] - time[start_idx] + 1                      │
 │    if duration < 10 s:  DISCARD                                         │
 │                                                                         │
 │  Removes residual noise transients that survive gap bridging            │
 │                                                                         │
 │  OUT: filtered_blocks                                                   │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 9 — FLARE PARAMETER EXTRACTION & CLASSIFICATION                 │
 │            (merge_and_filter_events, continued)                         │
 │                                                                         │
 │  For each valid event, compute:                                         │
 │                                                                         │
 │  peak_rate_raw  = max(counts[start:end])                               │
 │  peak_rate_sub  = max(counts[start:end] - background[start:end])       │
 │  peak_time      = time[argmax(counts[start:end])]                      │
 │  fluence        = sum(counts[start:end] - background[start:end])       │
 │  duration       = time[end] - time[start] + 1                          │
 │                                                                         │
 │  Classification (background-subtracted peak rate):                     │
 │    peak_rate_sub <  10 cts/s  →  "Weak"                               │
 │    peak_rate_sub <  50 cts/s  →  "Moderate"                           │
 │    peak_rate_sub >= 50 cts/s  →  "Strong"                             │
 │                                                                         │
 │  spectrum = None  (hook for Milestone 2 energy-resolved analysis)      │
 │                                                                         │
 │  OUT: List of FlareEvent objects                                        │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 10 — CATALOG & INTERMEDIATE FILE GENERATION                     │
 │             (save_intermediate_outputs)                                 │
 │                                                                         │
 │  outputs/flare_catalog.csv      13-column event table                  │
 │  outputs/summary.json           metadata + params + statistics          │
 │  outputs/cleaned_lightcurve.csv TIME, COUNTS_RAW, COUNTS_CLEANED       │
 │  outputs/background.csv         TIME, BACKGROUND, THRESHOLD_5SIGMA     │
 │  outputs/flare_plot.png         annotated light curve (150 dpi)        │
 │                                                                         │
 └──────────────────────────┬──────────────────────────────────────────────┘
                            │
                            ▼
 ┌─────────────────────────────────────────────────────────────────────────┐
 │  STAGE 11 — VISUALISATION  (generate_plot)                             │
 │                                                                         │
 │  Overlaid on a single 15×7 inch figure:                                │
 │    ·  Light gray  — raw counts (including NaN gaps)                    │
 │    ·  Royal blue  — GTI-cleaned counts                                  │
 │    ·  Dark orange — adaptive background                                 │
 │    ·  Forest green dashed — 5-sigma trigger threshold                  │
 │    ·  Crimson shading — detected flare windows                         │
 │    ·  Star marker + label — flare peak + classification                │
 │                                                                         │
 └─────────────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
.lc  file   ──┐
              ├──► [GTI Mask] ──► [Adaptive BG] ──► [Rolling MAD] ──► [5σ Thresh]
.gti file   ──┘                                                             │
                                                                            ▼
                                                                   [Consecutive 5s]
                                                                            │
                                                                            ▼
                                                                     [Gap Bridge]
                                                                            │
                                                                            ▼
                                                                   [Min Duration]
                                                                            │
                                                                            ▼
                                                               [Extract Properties]
                                                                            │
                                                               ┌────────────┴──────────────┐
                                                               ▼                           ▼
                                                       flare_catalog.csv          flare_plot.png
                                                       summary.json
                                                       cleaned_lightcurve.csv
                                                       background.csv
```

## Stage Input/Output Quick Reference

| Stage | Function | Key Input | Key Output |
|:---:|:---|:---|:---|
| 1 | `load_solexs_data` | `.lc`, `.gti` FITS paths | `time`, `counts_raw`, GTI arrays |
| 2 | `load_solexs_data` | GTI arrays, `time` | `counts_cleaned` (NaN masked) |
| 3 | `estimate_background_adaptive` | `counts_cleaned` | `background` (adaptive rolling median) |
| 4 | `estimate_dynamic_sigma` | `counts_cleaned`, `background` | `sigma` (rolling MAD × 1.4826) |
| 5 | inline | `background`, `sigma` | `threshold = bg + 5σ` |
| 6 | `detect_flares_consecutive` | `counts_cleaned`, `background`, `sigma` | `valid_trigger` boolean mask |
| 7 | `merge_and_filter_events` | `valid_trigger`, `time` | merged event index pairs |
| 8 | `merge_and_filter_events` | merged pairs, `time` | duration-filtered pairs |
| 9 | `merge_and_filter_events` | filtered pairs, `counts`, `background` | `List[FlareEvent]` |
| 10 | `save_intermediate_outputs` | `FlareEvent` list, arrays | 4 CSV/JSON files |
| 11 | `generate_plot` | all arrays + events | `flare_plot.png` |
