# SoLEXS Flare Detection — Code Walkthrough

This document walks through [`flare_detection.py`](./flare_detection.py)
section by section, explaining *what* each block does and *why* it does it.

---

## 0. Imports (Lines 1–8)

```python
import os, json, csv, datetime
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt
```

| Package | Role |
|:---|:---|
| `os` | Path construction for multi-OS compatibility |
| `json`, `csv` | Writing `summary.json` and the CSV output files |
| `datetime` | Converting Unix epoch times to ISO-8601 UTC strings |
| `numpy` | Array arithmetic — masking, argmax, nansum, nanmean |
| `pandas` | Rolling window operations (median, std) with NaN-aware logic |
| `astropy.io.fits` | Reading FITS files, including gzip-compressed `.gz` variants |
| `matplotlib` | Generating `flare_plot.png` |

> `scipy` is intentionally **not imported**. All required statistics (MAD, median)
> are implemented via NumPy/Pandas to keep the dependency footprint minimal.

---

## 1. Data Models & Utilities (Lines 10–68)

### 1.1 `FlareEvent` class

```python
class FlareEvent:
    def __init__(self, start_idx, end_idx, start_time, end_time, peak_time,
                 peak_rate_raw, peak_rate_sub, fluence, duration, classification):
        ...
        self.spectrum = None   # Milestone 2 hook
```

This class holds **all properties of a single detected event**.
The `spectrum = None` field is deliberately present now so Milestone 2 code
can attach a 340-channel spectral array without changing the class signature.

`to_dict()` is the serialisation method used for both the JSON summary and CSV
catalog. It calls `epoch_to_utc()` to convert the three epoch timestamps.

### 1.2 `epoch_to_utc()`

```python
return datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc).isoformat()
```

`datetime.timezone.utc` ensures the output is always `+00:00` regardless of
the local machine timezone. This is important for reproducibility across systems.

### 1.3 `find_contiguous_blocks(mask)`

```python
for idx, val in enumerate(mask):
    if val and not in_block:      # rising edge
        start_idx = idx
    elif not val and in_block:    # falling edge
        blocks.append((start_idx, idx - 1))
```

Classic O(N) run-detection. Used to extract trigger windows from the
`valid_trigger` boolean array. Returns `[(start0, end0), (start1, end1), ...]`.

---

## 2. FITS Loader & GTI Masking (Lines 74–112)

### 2.1 Opening FITS files

```python
with fits.open(lc_path) as hdul:
    lc_data    = hdul[1].data          # HDU index 1 = RATE extension
    time       = np.array(lc_data['TIME']).astype(float)
    counts_raw = np.array(lc_data['COUNTS']).astype(float)
```

**Why `.astype(float)`?**
FITS stores data in big-endian byte order by FITS standard. On x86 (little-endian)
machines, passing a big-endian `np.array` directly to `pandas.Series.rolling()`
causes:
```
ValueError: Big-endian buffer not supported on little-endian compiler
```
Calling `.astype(float)` triggers a memory copy into native byte order.

### 2.2 Metadata extraction

```python
meta = {
    "mission": lc_header.get("MISSION", "ADITYA-L1"),
    ...
}
```

`header.get(key, default)` is used rather than `header[key]` to avoid
`KeyError` on headers with missing optional keywords.

### 2.3 GTI masking

```python
valid = np.zeros(len(time), dtype=bool)
for start, stop in zip(gti_starts, gti_stops):
    valid |= (time >= start) & (time <= stop)

counts_cleaned = counts_raw.copy()
counts_cleaned[~valid] = np.nan
```

The `|=` (OR-assignment) across multiple GTI intervals correctly handles the
case where GTI intervals are non-overlapping — each iteration OR-masks its
own interval into the global validity array.

Any pre-existing NaNs in `counts_raw` (data gaps embedded in the FITS file)
are preserved since `counts_raw.copy()` keeps them and `~valid` only *adds*
more NaNs; it does not remove existing ones.

---

## 3. Background Estimation (Lines 114–138)

```python
bg_3600 = pd.Series(counts).rolling(window=3600, center=True, min_periods=1).median().values
bg_7200 = pd.Series(counts).rolling(window=7200, center=True, min_periods=1).median().values
```

`center=True` places each window symmetrically around the current sample
(using ±1800 s / ±3600 s of context), avoiding systematic lag.

`min_periods=1` ensures the first and last ~1800 samples (where less than a
full window of data is available) still get a result by using whatever data
is available, rather than producing NaN.

```python
local_std = pd.Series(residuals_3600).rolling(window=3600, center=True, min_periods=1).std().values
local_std = np.nan_to_num(local_std, nan=0.0)
is_noisy  = local_std > 4.0
background = np.where(is_noisy, bg_7200, bg_3600)
```

`np.where(condition, x, y)` is the element-wise ternary: where the local
residual standard deviation exceeds 4 cts/s, the more conservative 2-hour
background is selected.

```python
background = pd.Series(background).interpolate(limit_direction='both').bfill().ffill().values
```

`interpolate(limit_direction='both')` uses linear interpolation to fill NaN
values between valid estimates. `.bfill().ffill()` catches any remaining NaNs
at the very start or end of the array that interpolation cannot reach.

---

## 4. MAD Sigma Calculation (Lines 140–160)

```python
residuals    = counts - background
abs_residuals = np.abs(residuals)
rolling_mad  = pd.Series(abs_residuals).rolling(window=3600, center=True, min_periods=1).median().values
rolling_sigma = 1.4826 * rolling_mad
rolling_sigma = np.maximum(rolling_sigma, 0.5)
```

`np.abs(residuals)` converts signed deviations to unsigned magnitudes.
Taking the median of absolute deviations gives the Median Absolute Deviation (MAD).

Multiplying by 1.4826 converts MAD to an equivalent Gaussian sigma.
This is exact for a Gaussian: `sigma ≈ 1.4826 × MAD`.

`np.maximum(sigma, 0.5)` applies an element-wise floor. Without this,
during completely quiet periods where all residuals are 0–1 cts/s, the MAD
approaches zero and the threshold collapses to `background + 0 = background`,
triggering on every above-background sample.

---

## 5. Trigger Generation (Lines 162–191)

```python
above_threshold = (residuals > n_sigma * sigma) & ~np.isnan(counts)
```

Two conditions must both be True:
1. The residual exceeds 5 sigma (the flux condition)
2. The count is not NaN (the validity condition — rejects GTI-masked samples)

**Pass 1 — build run-length:**
```python
run_length = np.zeros(len(above_threshold), dtype=int)
count = 0
for i in range(len(above_threshold)):
    if above_threshold[i]:
        count += 1
    else:
        count = 0
    run_length[i] = count
```
`run_length[i]` = the number of consecutive True values *ending at* index `i`.
E.g. `[F, T, T, T, F]` → `[0, 1, 2, 3, 0]`.

**Pass 2 — backward scan to mark valid blocks:**
```python
i = len(above_threshold) - 1
while i >= 0:
    if run_length[i] >= min_trigger_duration:
        length = run_length[i]
        valid_trigger[i - length + 1 : i + 1] = True
        i -= length          # jump past the entire qualifying run
    else:
        i -= 1
```
The backward scan with `i -= length` is critical: it ensures overlapping runs
are not double-counted and that we jump cleanly to the sample *before* each
qualifying run. This correctly handles runs longer than `min_trigger_duration`.

---

## 6. Event Merging (Lines 193–213)

```python
for start, end in initial_blocks[1:]:
    gap_sec = time[start] - time[current_end] - 1
    if gap_sec <= max_gap:
        current_end = end
    else:
        merged_blocks.append((current_start, current_end))
        current_start, current_end = start, end
```

`time[start] - time[current_end] - 1` calculates the gap *between* two
blocks in seconds. The `-1` corrects for the fact that `time[current_end]`
and `time[start]` are the timestamps of the last and first samples; the
gap between them is `time[start] - time[current_end]`, but since each sample
covers 1 second, consecutive seconds with no gap yield a difference of 1.
Subtracting 1 gives 0 for truly adjacent samples, as expected.

---

## 7. Event Classification (Lines 243–249)

```python
if peak_rate_sub < 10.0:
    classification = "Weak"
elif peak_rate_sub < 50.0:
    classification = "Moderate"
else:
    classification = "Strong"
```

Classification uses the **background-subtracted** peak, not the raw peak.
This means a 15 cts/s peak against a 12 cts/s background (net 3 cts/s) is
correctly classified as "Weak", while a 15 cts/s peak against a 2 cts/s
background (net 13 cts/s) is "Moderate".

---

## 8. CSV & JSON Output (Lines 271–337)

### `cleaned_lightcurve.csv`

```python
writer.writerow([t, r, c if not np.isnan(c) else ""])
```

`np.isnan()` converts NaN to empty string in the CSV, which is more portable
than the literal string "nan" for downstream tools.

### `background.csv`

Writes the two most important derived time series — the rolling background
estimate and the dynamic threshold — needed for downstream quality control
and Milestone 3 multi-day analysis.

### `flare_catalog.csv`

13-column table, one row per detected flare. Dual timestamps (epoch + UTC)
are included to avoid timezone ambiguity when the CSV is opened in Excel.

### `summary.json`

Nested JSON with four top-level keys: `metadata`, `parameters`, `statistics`,
`flares`. The `flares` key contains the full `FlareEvent.to_dict()` output for
each event, including `"spectrum": null` for Milestone 2 readiness.

---

## 9. Plot Generation (Lines 345–383)

```python
plt.figure(figsize=(15, 7))
plt.plot(time - time[0], counts_raw, color='lightgray', ...)
plt.plot(time - time[0], counts_cleaned, color='royalblue', ...)
plt.plot(time - time[0], background, color='darkorange', ...)
plt.plot(time - time[0], threshold, color='forestgreen', linestyle='--', ...)
```

`time - time[0]` shifts the x-axis to seconds since midnight (start of day),
making the plot readable without interpreting Unix epoch values.

```python
plt.axvspan(ev.start_time - time[0], ev.end_time - time[0], color='crimson', alpha=0.25)
plt.plot(ev.peak_time - time[0], ev.peak_rate_raw, '*', markersize=8)
plt.text(ev.peak_time - time[0], ev.peak_rate_raw + 2.0, ev.classification, ...)
```

`axvspan` draws a vertical shaded band for the full flare duration.
The star marker (`'*'`) marks the exact peak sample.
The text label is placed 2 cts/s above the peak to avoid overlap.

---

## 10. Main Orchestrator (Lines 389–451)

```python
# Path resolution with fallback to .gz
lc_path = os.path.join(sdd2_dir, lc_file, lc_file)  # uncompressed nested
if not os.path.exists(lc_path):
    lc_path = os.path.join(sdd2_dir, "AL1_SOLEXS_20260618_SDD2_L1.lc.gz")
```

The ISRO L1 data ships with a peculiar structure where uncompressed files are
stored inside a subdirectory with the same name as the file itself (e.g.
`SDD2/AL1_SOLEXS_20260618_SDD2_L1.lc/AL1_SOLEXS_20260618_SDD2_L1.lc`).
The fallback to `.gz` handles archives that have not been decompressed.

```python
threshold = background + 5.0 * sigma
trigger_mask = detect_flares_consecutive(counts_cleaned, background, sigma, n_sigma=5.0, min_trigger_duration=5)
events = merge_and_filter_events(trigger_mask, time, counts_cleaned, background, max_gap=10, min_duration=10)
```

The main pipeline is a clean linear sequence of pure functions — each
takes arrays as input and returns new arrays or object lists. No global
mutable state is used, making the pipeline trivially testable.

---

## Quick Modification Guide

| Change | Where to edit |
|:---|:---|
| Change threshold from 5σ to 3σ | `main()`: `n_sigma=3.0` in `detect_flares_consecutive(...)` |
| Use a different background window | `main()`: call `estimate_background_adaptive(..., default_window=1800)` |
| Add a new flare property (e.g. rise time) | `merge_and_filter_events()`: compute and store in `FlareEvent`, add to `to_dict()` |
| Process a different date | `main()`: update the `sdd2_dir` path and fallback filenames |
| Use SDD1 data | Update paths; handle the empty GTI case (SDD1 returns 0 valid intervals) |
| Export to FITS instead of CSV | Replace the CSV writer with `astropy.table.Table.write(..., format='fits')` |
