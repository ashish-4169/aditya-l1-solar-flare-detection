# SoLEXS Solar Flare Detection — Technical Documentation

## 1. Software Architecture

The pipeline is implemented as a **single-script, function-oriented** Python module
([`flare_detection.py`](./flare_detection.py)) with four logical sections:

```
Section 1  — Data Models & Utilities      (lines   1–68)
Section 2  — Core Pipeline Functions      (lines  70–265)
Section 3  — Output & Visualisation       (lines 267–383)
Section 4  — Main Orchestrator            (lines 385–452)
```

There are **no external ML frameworks** and **no scipy dependency** — only
`numpy`, `pandas`, `astropy`, and `matplotlib`.

---

## 2. Core Classes

### `FlareEvent`

Represents a single detected solar flare event. All numeric fields are Python
native types to allow direct JSON serialisation.

```python
class FlareEvent:
    start_idx     : int    # index into time[] array where flare begins
    end_idx       : int    # index into time[] array where flare ends
    start_time    : float  # Unix epoch seconds of flare start
    end_time      : float  # Unix epoch seconds of flare end
    peak_time     : float  # Unix epoch seconds of maximum count rate
    peak_rate_raw : float  # maximum raw count rate (cts/s)
    peak_rate_sub : float  # maximum background-subtracted rate (cts/s)
    fluence       : float  # integrated (counts - background) over event [counts]
    duration      : float  # end_time - start_time + 1 [seconds]
    classification: str    # "Weak" | "Moderate" | "Strong"
    spectrum      : None   # placeholder — Milestone 2 will attach spectral data
```

**Method: `to_dict()`** — serialises the event to a plain Python dictionary,
converting epoch times to ISO-8601 UTC strings via `epoch_to_utc()`.

---

## 3. Core Functions

### `epoch_to_utc(epoch_time: float) -> str`

```
Input:  Unix epoch time in seconds (float)
Output: ISO-8601 UTC string, e.g. "2026-06-18T11:55:18+00:00"
```

Uses `datetime.datetime.fromtimestamp(t, datetime.timezone.utc)`.
Returns `"N/A"` on any exception (protects against NaN inputs).

---

### `find_contiguous_blocks(mask: np.ndarray[bool]) -> List[Tuple[int,int]]`

```
Input:  boolean array of length N
Output: list of (start_idx, end_idx) pairs for every run of True values
```

Single linear pass: O(N). Used twice — once to extract trigger windows,
once implicitly inside the gap-bridging loop.

---

### `load_solexs_data(lc_path, gti_path)`

```
Input:  paths to .lc and .gti FITS files (compressed .gz supported)
Output: time, counts_raw, counts_cleaned, gti_starts, gti_stops, meta
```

**Key implementation note:** FITS arrays from `astropy.io.fits` are stored in
big-endian byte order. Pandas rolling functions require native (little-endian)
arrays on x86 machines. All arrays are cast with `.astype(float)` immediately
after `np.array()` to force native endianness. Omitting this causes:

```
ValueError: Big-endian buffer not supported on little-endian compiler
```

**GTI masking logic:**
```python
valid = np.zeros(len(time), dtype=bool)
for start, stop in zip(gti_starts, gti_stops):
    valid |= (time >= start) & (time <= stop)
counts_cleaned[~valid] = np.nan
```

This is the **authoritative** validity mask. Pre-existing NaNs from the FITS
file are supplementary — the GTI takes precedence.

---

### `estimate_background_adaptive(time, counts, default_window=3600, active_window=7200, noise_threshold=4.0)`

```
Input:  cleaned counts array, window sizes in seconds
Output: background[N]  (float64, no NaNs)
```

**Algorithm:**

1. `bg_3600 = rolling_median(counts, 3600)` — 1-hour window
2. `bg_7200 = rolling_median(counts, 7200)` — 2-hour window
3. `local_std = rolling_std(counts - bg_3600, 3600)` — local variability
4. Where `local_std > 4.0`: select `bg_7200`, else `bg_3600`

**Why rolling median, not mean?**
The median is resistant to outliers (flare peaks). A 40 cts/s flare peak
spanning 2 minutes in a 3600-second window shifts the mean by ~2 cts/s but
does not affect the median if it represents less than 50% of the window.

**Why two windows?**
A 1-hour window is precise for quiet times. During active periods (many small
enhancements), the local standard deviation rises above 4 cts/s, signalling
that the 3600 s estimate may be biased upward; the 2-hour window averages over
more quiet data to give a more conservative lower floor.

**NaN handling:** `pd.Series.interpolate(limit_direction='both').bfill().ffill()`
fills any remaining NaN gaps (e.g. at array edges or total data outages) after
rolling computation.

---

### `estimate_dynamic_sigma(counts, background, window=3600)`

```
Input:  counts_cleaned, background, window size
Output: sigma[N]  (float64, >= 0.5 everywhere)
```

**Formula:**

```
residuals[t]     = counts[t] - background[t]
rolling_MAD[t]   = median(|residuals[t-w/2 : t+w/2]|)
sigma[t]         = 1.4826 × rolling_MAD[t]
sigma[t]         = max(sigma[t], 0.5)
```

**Derivation of scale factor 1.4826:**
For a Gaussian distribution:
```
sigma = MAD / Phi^-1(3/4)  =  MAD / 0.6745  =  MAD × 1.4826
```
This makes the MAD-based sigma directly comparable to the standard deviation
under Gaussian assumptions.

**Why MAD over standard deviation?**
Standard deviation is inflated by flare peaks (outliers). A 97 cts/s peak in
an 86 400-sample distribution would artificially raise the global sigma,
suppressing sensitivity during quiet periods. MAD is bounded at 50th percentile
so isolated extreme events cannot dominate the noise estimate.

**Minimum floor of 0.5 cts/s:**
During extremely quiet intervals, all residuals may be near zero, giving
`MAD = 0` and `sigma = 0`. A threshold of `background + 5 × 0 = background`
would trigger on every point above background. The floor of 0.5 prevents this.

---

### `detect_flares_consecutive(counts, background, sigma, n_sigma=5.0, min_trigger_duration=5)`

```
Input:  counts_cleaned, background, sigma
Output: valid_trigger[N]  (bool)
```

**Algorithm (two-pass):**

Pass 1 — build run-length array:
```python
run_length[i] = (run_length[i-1] + 1)  if above[i]  else  0
```
`run_length[i]` = number of consecutive True values ending at index `i`.

Pass 2 — backward scan to set valid_trigger:
```python
if run_length[i] >= 5:
    valid_trigger[i - run_length[i] + 1 : i + 1] = True
    i -= run_length[i]   # skip past the whole run
```
This correctly marks the *entire* qualifying run as triggered, not just its
last sample.

**Complexity:** O(N) time, O(N) space.

**Why 5 consecutive seconds?**
Diagnostic analysis showed that:
- Cosmic ray hits: 1 sample, ~10σ amplitude → rejected
- Poisson noise excursions: 1–4 samples → rejected
- Real C-class flares: sustained 10–100+ s → pass
- Smallest confirmed flare (Flare 2): 19 s → pass with margin

---

### `merge_and_filter_events(trigger_mask, time, counts, background, max_gap=10, min_duration=10)`

```
Input:  valid_trigger mask, time/counts/background arrays
Output: List[FlareEvent]
```

**Gap bridging:**
```python
gap_sec = time[next_start] - time[current_end] - 1
if gap_sec <= max_gap:
    current_end = next_end   # extend current event
```

**Why 10-second gap tolerance?**
In 1 cts/s Poisson noise at 5 cts/s mean, there is a ~0.7% chance per second
of drawing ≥0 counts. Consecutive 1-second dropouts of 10+ s are rare in
genuine flares; 10 s bridges nearly all counting-statistics dips while not
merging genuinely separate events.

**Parameter extraction (per event):**

| Parameter | Formula |
|:---|:---|
| `peak_rate_raw` | `max(counts[start:end])` |
| `peak_rate_sub` | `max(counts[start:end] - background[start:end])` |
| `peak_time` | `time[argmax(counts[start:end])]` |
| `fluence` | `sum(counts[start:end] - background[start:end])` (ignores NaN) |
| `duration` | `time[end] - time[start] + 1` |

**Classification thresholds:**

| Class | Condition | Rationale |
|:---|:---|:---|
| Weak | `peak_rate_sub < 10 cts/s` | Sub-threshold enhancement, possibly B-class |
| Moderate | `10 <= peak_rate_sub < 50 cts/s` | Clear C-class range |
| Strong | `peak_rate_sub >= 50 cts/s` | M-class equivalent at SoLEXS sensitivity |

> **Note:** These thresholds map to SoLEXS count rates, not GOES W m⁻² classes.
> A proper GOES-equivalent mapping requires spectral calibration (Milestone 2).

---

## 4. Data Structures

### Time Series Arrays (all length 86 400)

| Array | dtype | Description |
|:---|:---|:---|
| `time` | float64 | Unix epoch seconds |
| `counts_raw` | float64 | As-read FITS counts (includes off-GTI data) |
| `counts_cleaned` | float64 | NaN where outside GTI |
| `background` | float64 | Adaptive rolling median |
| `sigma` | float64 | Rolling MAD × 1.4826, >= 0.5 |
| `threshold` | float64 | `background + 5 × sigma` |
| `above` | bool | `residual > 5σ AND NOT NaN` |
| `run_length` | int | Consecutive True count ending at each sample |
| `valid_trigger` | bool | Passes the ≥5 consecutive seconds test |

### GTI Arrays (length 4 for this observation)

| Array | dtype | Description |
|:---|:---|:---|
| `gti_starts` | float64 | Unix epoch start of each good interval |
| `gti_stops` | float64 | Unix epoch end of each good interval |

---

## 5. Mathematical Summary

```
Background:   B(t) = median{ C(s) : |s-t| <= W/2 }
              W = 3600 s (quiet)  or  7200 s (active)

Noise:        MAD(t) = median{ |C(s)-B(s)| : |s-t| <= 1800 s }
              sigma(t) = 1.4826 × MAD(t)  or  0.5  (whichever is larger)

Threshold:    T(t) = B(t) + 5 × sigma(t)

Trigger:      A(t) = [C(t) - B(t) > 5*sigma(t)]  AND  [C(t) != NaN]
              ValidRun(t) = [max consecutive A ending at t] >= 5

Fluence:      F_event = SUM_{t=start}^{end} [ C(t) - B(t) ]  (NaN excluded)
```

---

## 6. Computational Complexity

| Stage | Time | Space |
|:---|:---:|:---:|
| GTI masking | O(N × G) ≈ O(N) | O(N) |
| Rolling median (pandas) | O(N × W) | O(W) |
| Rolling std | O(N × W) | O(W) |
| Run-length detection | O(N) | O(N) |
| Gap bridging | O(E) | O(E) |
| Output writing | O(N) | O(N) |
| **Total** | **O(N × W)** | **O(N)** |

For N=86 400, W=7 200: approximately 622 million operations.
Actual runtime: ~6 seconds on a modern CPU (dominated by Pandas rolling median).

---

## 7. Known Assumptions and Limitations

| Assumption | Implication |
|:---|:---|
| 1-second uniform time bins | Pipeline assumes `TIMEDEL=1`. Irregularly sampled data would break rolling window logic. |
| Single detector (SDD2) | SDD1 had no valid GTI intervals on this date. Multi-detector coincidence logic is not implemented. |
| Poisson count statistics | The MAD-based sigma implicitly assumes near-Poisson noise. Systematic detector artefacts (temperature drifts, calibration pulses) are not modelled. |
| No energy resolution | The `.lc` file sums all energy channels. Spectral contamination (particle background vs X-ray) is not separated until Milestone 2. |
| Classification thresholds | Based on empirical count rates, not calibrated to GOES W m⁻² flux. |
| UTC epoch reference | Time is stored as Unix epoch. Astropy MJDREFI=40587 is noted but not used; epoch_to_utc uses Python's datetime. |
