import os
import json
import csv
import datetime
import numpy as np
import pandas as pd
from astropy.io import fits
import matplotlib.pyplot as plt

# ==============================================================================
# 1. DATA MODELS & UTILS
# ==============================================================================

class FlareEvent:
    def __init__(self, start_idx, end_idx, start_time, end_time, peak_time, 
                 peak_rate_raw, peak_rate_sub, fluence, duration, classification):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.start_time = start_time
        self.end_time = end_time
        self.peak_time = peak_time
        self.peak_rate_raw = peak_rate_raw
        self.peak_rate_sub = peak_rate_sub
        self.fluence = fluence
        self.duration = duration
        self.classification = classification
        self.spectrum = None  # Hook for Milestone 2 (energy-resolved analysis)

    def to_dict(self):
        return {
            "start_idx": int(self.start_idx),
            "end_idx": int(self.end_idx),
            "start_time_epoch": float(self.start_time),
            "end_time_epoch": float(self.end_time),
            "peak_time_epoch": float(self.peak_time),
            "start_time_utc": epoch_to_utc(self.start_time),
            "end_time_utc": epoch_to_utc(self.end_time),
            "peak_time_utc": epoch_to_utc(self.peak_time),
            "peak_rate_raw": float(self.peak_rate_raw),
            "peak_rate_sub": float(self.peak_rate_sub),
            "fluence": float(self.fluence),
            "duration_seconds": float(self.duration),
            "classification": self.classification,
            "spectrum": self.spectrum
        }

def epoch_to_utc(epoch_time):
    """Convert Unix epoch seconds to UTC ISO 8601 string."""
    try:
        return datetime.datetime.fromtimestamp(epoch_time, datetime.timezone.utc).isoformat()
    except Exception:
        return "N/A"

def find_contiguous_blocks(mask):
    """Find start and end indices of contiguous True blocks in a boolean mask."""
    blocks = []
    in_block = False
    start_idx = None
    for idx, val in enumerate(mask):
        if val and not in_block:
            in_block = True
            start_idx = idx
        elif not val and in_block:
            in_block = False
            blocks.append((start_idx, idx - 1))
    if in_block:
        blocks.append((start_idx, len(mask) - 1))
    return blocks

# ==============================================================================
# 2. CORE PIPELINE FUNCTIONS
# ==============================================================================

def load_solexs_data(lc_path, gti_path):
    """
    Loads SoLEXS light curve and GTI files, applying authoritative GTI masking.
    """
    print(f"Loading light curve from: {lc_path}")
    with fits.open(lc_path) as hdul:
        lc_data = hdul[1].data
        time = np.array(lc_data['TIME']).astype(float)
        counts_raw = np.array(lc_data['COUNTS']).astype(float)
        
        # Read header metadata
        lc_header = hdul[0].header
        meta = {
            "mission": lc_header.get("MISSION", "ADITYA-L1"),
            "telescope": lc_header.get("TELESCOP", "AL1"),
            "instrument": lc_header.get("INSTRUME", "SoLEXS"),
            "observation_date": lc_header.get("OBS_DATE", "Unknown"),
            "observation_id": lc_header.get("OBS_ID", "Unknown")
        }

    print(f"Loading GTI from: {gti_path}")
    with fits.open(gti_path) as hdul:
        gti_data = hdul[1].data
        gti_starts = np.array(gti_data['START']).astype(float)
        gti_stops = np.array(gti_data['STOP']).astype(float)

    # Authoritative GTI Masking
    print("Applying authoritative GTI masking...")
    valid = np.zeros(len(time), dtype=bool)
    for start, stop in zip(gti_starts, gti_stops):
        valid |= (time >= start) & (time <= stop)
    
    counts_cleaned = counts_raw.copy()
    counts_cleaned[~valid] = np.nan
    
    meta["exposure_seconds"] = float(np.sum(valid))
    meta["duty_cycle_percent"] = float((np.sum(valid) / len(time)) * 100.0)
    
    return time, counts_raw, counts_cleaned, gti_starts, gti_stops, meta

def estimate_background_adaptive(time, counts, default_window=3600, active_window=7200, noise_threshold=4.0):
    """
    Estimates the background count rate using an adaptive rolling median window.
    Uses 3600s default window, and 7200s window if the local residuals become noisy.
    """
    print("Estimating background with adaptive rolling median window...")
    # Calculate rolling medians using pandas (handles NaNs gracefully)
    bg_3600 = pd.Series(counts).rolling(window=default_window, center=True, min_periods=1).median().values
    bg_7200 = pd.Series(counts).rolling(window=active_window, center=True, min_periods=1).median().values
    
    # Calculate residuals from the baseline 3600s median
    residuals_3600 = counts - bg_3600
    
    # Compute rolling standard deviation of residuals to detect local activity/noise
    local_std = pd.Series(residuals_3600).rolling(window=default_window, center=True, min_periods=1).std().values
    local_std = np.nan_to_num(local_std, nan=0.0)
    
    # Select background window based on local activity
    is_noisy = local_std > noise_threshold
    background = np.where(is_noisy, bg_7200, bg_3600)
    
    # Clean up any remaining NaNs in background (e.g. at boundaries or full gap periods)
    background = pd.Series(background).interpolate(limit_direction='both').bfill().ffill().values
    
    return background

def estimate_dynamic_sigma(counts, background, window=3600):
    """
    Calculates the dynamic noise level (sigma) using rolling Median Absolute Deviation (MAD).
    """
    print("Estimating dynamic noise sigma using rolling MAD...")
    residuals = counts - background
    abs_residuals = np.abs(residuals)
    
    # Rolling median of absolute residuals
    rolling_mad = pd.Series(abs_residuals).rolling(window=window, center=True, min_periods=1).median().values
    
    # Scaling factor for normal distribution equivalence (1.4826)
    rolling_sigma = 1.4826 * rolling_mad
    
    # Enforce minimum sigma floor to prevent division by zero or hypersensitivity in quiet times
    rolling_sigma = np.maximum(rolling_sigma, 0.5)
    
    # Clean up NaNs in sigma series
    rolling_sigma = pd.Series(rolling_sigma).interpolate(limit_direction='both').bfill().ffill().values
    
    return rolling_sigma

def detect_flares_consecutive(counts, background, sigma, n_sigma=5.0, min_trigger_duration=5):
    """
    Identifies triggers where the count rate exceeds the threshold for at least
    min_trigger_duration consecutive seconds.
    """
    print(f"Detecting triggers exceeding background + {n_sigma}*sigma for >= {min_trigger_duration}s...")
    residuals = counts - background
    above_threshold = (residuals > n_sigma * sigma) & ~np.isnan(counts)
    
    # Find runs of at least min_trigger_duration consecutive seconds
    run_length = np.zeros(len(above_threshold), dtype=int)
    count = 0
    for i in range(len(above_threshold)):
        if above_threshold[i]:
            count += 1
        else:
            count = 0
        run_length[i] = count
        
    valid_trigger = np.zeros(len(above_threshold), dtype=bool)
    i = len(above_threshold) - 1
    while i >= 0:
        if run_length[i] >= min_trigger_duration:
            length = run_length[i]
            valid_trigger[i - length + 1 : i + 1] = True
            i -= length
        else:
            i -= 1
            
    return valid_trigger

def merge_and_filter_events(trigger_mask, time, counts, background, max_gap=10, min_duration=10):
    """
    Merges adjacent trigger blocks separated by <= max_gap seconds,
    and filters out final events with duration < min_duration seconds.
    """
    print(f"Merging events (gap <= {max_gap}s) and filtering (duration >= {min_duration}s)...")
    initial_blocks = find_contiguous_blocks(trigger_mask)
    if not initial_blocks:
        return []
        
    # Merge blocks
    merged_blocks = []
    current_start, current_end = initial_blocks[0]
    for start, end in initial_blocks[1:]:
        gap_sec = time[start] - time[current_end] - 1
        if gap_sec <= max_gap:
            current_end = end
        else:
            merged_blocks.append((current_start, current_end))
            current_start, current_end = start, end
    merged_blocks.append((current_start, current_end))
    
    # Filter and construct FlareEvent objects
    final_events = []
    for start_idx, end_idx in merged_blocks:
        start_time = time[start_idx]
        end_time = time[end_idx]
        duration = end_time - start_time + 1
        
        if duration < min_duration:
            continue
            
        # Segment data for properties
        flare_counts = counts[start_idx : end_idx + 1]
        flare_bg = background[start_idx : end_idx + 1]
        flare_times = time[start_idx : end_idx + 1]
        
        # Calculate peak rate (handling potential NaNs)
        net_counts = flare_counts - flare_bg
        if np.all(np.isnan(flare_counts)):
            continue
            
        peak_idx_local = np.nanargmax(flare_counts)
        peak_time = flare_times[peak_idx_local]
        peak_rate_raw = flare_counts[peak_idx_local]
        peak_rate_sub = net_counts[peak_idx_local]
        
        # Fluence (integrated counts above background, ignoring NaNs)
        fluence = np.nansum(net_counts)
        
        # Classification based on background-subtracted peak rate
        if peak_rate_sub < 10.0:
            classification = "Weak"
        elif peak_rate_sub < 50.0:
            classification = "Moderate"
        else:
            classification = "Strong"
            
        event = FlareEvent(
            start_idx=start_idx,
            end_idx=end_idx,
            start_time=start_time,
            end_time=end_time,
            peak_time=peak_time,
            peak_rate_raw=peak_rate_raw,
            peak_rate_sub=peak_rate_sub,
            fluence=fluence,
            duration=duration,
            classification=classification
        )
        final_events.append(event)
        
    return final_events

# ==============================================================================
# 3. SAVING OUTPUTS & VISUALIZATION
# ==============================================================================

def save_intermediate_outputs(time, counts_raw, counts_cleaned, background, threshold, events, meta, output_dir):
    """
    Saves catalog, cleaned light curve, background, summary, and plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving intermediate files to: {output_dir}")
    
    # 1. Cleaned Light Curve CSV
    lc_csv_path = os.path.join(output_dir, "cleaned_lightcurve.csv")
    with open(lc_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TIME", "COUNTS_RAW", "COUNTS_CLEANED"])
        for t, r, c in zip(time, counts_raw, counts_cleaned):
            writer.writerow([t, r, c if not np.isnan(c) else ""])
            
    # 2. Background and Threshold CSV
    bg_csv_path = os.path.join(output_dir, "background.csv")
    with open(bg_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["TIME", "BACKGROUND", "THRESHOLD_5SIGMA"])
        for t, b, th in zip(time, background, threshold):
            writer.writerow([t, b, th])
            
    # 3. Flare Catalog CSV
    cat_csv_path = os.path.join(output_dir, "flare_catalog.csv")
    with open(cat_csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "START_IDX", "END_IDX", "START_TIME_EPOCH", "START_TIME_UTC",
            "PEAK_TIME_EPOCH", "PEAK_TIME_UTC", "END_TIME_EPOCH", "END_TIME_UTC",
            "PEAK_RATE_RAW", "PEAK_RATE_SUB", "FLUENCE", "DURATION_SECONDS", "CLASSIFICATION"
        ])
        for ev in events:
            d = ev.to_dict()
            writer.writerow([
                d["start_idx"], d["end_idx"], d["start_time_epoch"], d["start_time_utc"],
                d["peak_time_epoch"], d["peak_time_utc"], d["end_time_epoch"], d["end_time_utc"],
                d["peak_rate_raw"], d["peak_rate_sub"], d["fluence"], d["duration_seconds"], d["classification"]
            ])
            
    # 4. Summary JSON
    class_counts = {"Weak": 0, "Moderate": 0, "Strong": 0}
    for ev in events:
        class_counts[ev.classification] += 1
        
    summary = {
        "metadata": meta,
        "parameters": {
            "default_bg_window_seconds": 3600,
            "adaptive_bg_window_seconds": 7200,
            "trigger_threshold_sigma": 5.0,
            "min_consecutive_trigger_seconds": 5,
            "max_gap_bridge_seconds": 10,
            "min_event_duration_seconds": 10
        },
        "statistics": {
            "total_flares_detected": len(events),
            "flare_classifications": class_counts,
            "average_background_rate": float(np.nanmean(background)),
            "average_counts_rate": float(np.nanmean(counts_cleaned))
        },
        "flares": [ev.to_dict() for ev in events]
    }
    
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, mode='w') as f:
        json.dump(summary, f, indent=4)
        
    # 5. Flare Plot Image
    plot_path = os.path.join(output_dir, "flare_plot.png")
    generate_plot(time, counts_raw, counts_cleaned, background, threshold, events, meta, plot_path)
    
    print("All files saved successfully.")

def generate_plot(time, counts_raw, counts_cleaned, background, threshold, events, meta, output_path):
    """
    Generates a high-quality visualization plot of the light curve and detections.
    """
    plt.figure(figsize=(15, 7))
    
    # Plot raw and cleaned count rate
    plt.plot(time - time[0], counts_raw, color='lightgray', label='Raw Counts (Offline/Gaps)', alpha=0.7, linewidth=0.8)
    plt.plot(time - time[0], counts_cleaned, color='royalblue', label='Cleaned Counts (Valid Obs)', alpha=0.9, linewidth=1.0)
    
    # Plot background and threshold
    plt.plot(time - time[0], background, color='darkorange', label='Adaptive Background', linewidth=1.5)
    plt.plot(time - time[0], threshold, color='forestgreen', linestyle='--', label='Trigger Threshold (Bg + 5σ)', linewidth=1.2)
    
    # Shade flare regions
    shaded_label = False
    for ev in events:
        label = 'Detected Flare' if not shaded_label else None
        plt.axvspan(ev.start_time - time[0], ev.end_time - time[0], color='crimson', alpha=0.25, label=label)
        shaded_label = True
        
        # Mark flare peaks
        plt.plot(ev.peak_time - time[0], ev.peak_rate_raw, '*', color='crimson', markersize=8)
        # Add classification label text above peak
        plt.text(ev.peak_time - time[0], ev.peak_rate_raw + 2.0, ev.classification, 
                 color='darkred', fontsize=8, weight='bold', ha='center')

    # Formatting
    plt.title(f"Solar Flare Detection Pipeline (SoLEXS SDD2) - {meta['observation_date']}\n(Obs ID: {meta['observation_id']}, Mission: {meta['mission']})", fontsize=14, weight='bold', pad=10)
    plt.xlabel(f"Time (seconds since start of day: {epoch_to_utc(time[0])})", fontsize=12)
    plt.ylabel("Counts per Second (cts/s)", fontsize=12)
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend(loc='upper right', frameon=True, facecolor='white', framealpha=0.9)
    plt.ylim(0, np.nanmax(counts_raw) * 1.15 if len(counts_raw) > 0 else 100)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Generated visualization plot: {output_path}")

# ==============================================================================
# 4. MAIN PIPELINE RUNNER
# ==============================================================================

def main():
    print("======================================================================")
    print("             Milestone 1: Solar Flare Detection Pipeline             ")
    print("======================================================================")
    
    # Automatically locate files in the workspace
    workspace = os.path.dirname(os.path.abspath(__file__))
    
    # Look for SDD2 files
    sdd2_dir = os.path.join(workspace, "AL1_SLX_L1_20260618_v1.0", "AL1_SLX_L1_20260618_v1.0", "SDD2")
    
    lc_file = "AL1_SOLEXS_20260618_SDD2_L1.lc"
    gti_file = "AL1_SOLEXS_20260618_SDD2_L1.gti"
    
    lc_path = os.path.join(sdd2_dir, lc_file, lc_file)
    gti_path = os.path.join(sdd2_dir, gti_file, gti_file)
    
    # If uncompressed folder structure is not found, fall back to gzip files directly
    if not os.path.exists(lc_path):
        lc_path = os.path.join(sdd2_dir, "AL1_SOLEXS_20260618_SDD2_L1.lc.gz")
    if not os.path.exists(gti_path):
        gti_path = os.path.join(sdd2_dir, "AL1_SOLEXS_20260618_SDD2_L1.gti.gz")
        
    if not os.path.exists(lc_path) or not os.path.exists(gti_path):
        print(f"ERROR: Could not find required input files in {sdd2_dir}")
        print(f"Checked path: {lc_path}")
        print(f"Checked path: {gti_path}")
        return
        
    output_dir = os.path.join(workspace, "outputs")
    
    # Step 1: Load and Mask Data
    time, counts_raw, counts_cleaned, gti_starts, gti_stops, meta = load_solexs_data(lc_path, gti_path)
    
    # Step 2: Background Estimation
    background = estimate_background_adaptive(time, counts_cleaned)
    
    # Step 3: Noise Level Estimation (Dynamic Sigma)
    sigma = estimate_dynamic_sigma(counts_cleaned, background)
    
    threshold = background + 5.0 * sigma
    
    # Step 4: Consecutive-Point Triggering (at least 5s)
    trigger_mask = detect_flares_consecutive(counts_cleaned, background, sigma, n_sigma=5.0, min_trigger_duration=5)
    
    # Step 5: Merge and filter events (bridge <= 10s gaps, filter duration < 10s)
    events = merge_and_filter_events(trigger_mask, time, counts_cleaned, background, max_gap=10, min_duration=10)
    
    print(f"\nPipeline execution completed. Detected {len(events)} valid solar flare events.")
    for idx, ev in enumerate(events):
        print(f"  Flare {idx+1} [{ev.classification}]: "
              f"Start: {epoch_to_utc(ev.start_time)} | "
              f"Peak: {epoch_to_utc(ev.peak_time)} | "
              f"End: {epoch_to_utc(ev.end_time)} | "
              f"Duration: {ev.duration}s | "
              f"Fluence: {ev.fluence:.2f} counts")
        
    # Step 6: Save results and plots
    save_intermediate_outputs(time, counts_raw, counts_cleaned, background, threshold, events, meta, output_dir)
    print("======================================================================")

if __name__ == "__main__":
    main()
