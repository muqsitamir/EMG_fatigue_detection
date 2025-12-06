import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from scipy.signal import butter, filtfilt, iirnotch, welch, find_peaks


def butter_bandpass(lowcut, highcut, fs, order=4):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a

def bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

def notch_filter(data, fs, notch_freq=50.0, quality=30.0):
    b, a = iirnotch(notch_freq, quality, fs)
    return filtfilt(b, a, data)

def rectify_and_envelope(emg, fs, lp_cut=5.0):
    rect = np.abs(emg)
    b, a = butter(4, lp_cut / (0.5 * fs), btype='low')
    env = filtfilt(b, a, rect)
    return rect, env

def segment_reps_by_envelope(env, fs, distance_seconds=0.5, prominence=0.1):
    distance = int(distance_seconds * fs)
    peaks, props = find_peaks(env, distance=distance, prominence=prominence*np.max(env))
    return peaks, props

def rms(signal_segment):
    return np.sqrt(np.mean(signal_segment**2))

def median_frequency(signal_segment, fs):
    f, Pxx = welch(signal_segment, fs=fs, nperseg=min(1024, len(signal_segment)))
    cumsum = np.cumsum(Pxx)
    total = cumsum[-1]
    if total == 0:
        return 0.0
    median_idx = np.searchsorted(cumsum, total / 2.0)
    return f[median_idx]

def process_emg(time, emg, fs=None, lowcut=20, highcut=450, notch_freq=50.0):
    if fs is None:
        dt = np.median(np.diff(time))
        fs = 1.0 / dt
    # print(f'Estimated sampling rate: {fs:.1f} Hz')
    emg_bp = bandpass_filter(emg, lowcut, highcut, fs)
    emg_notch = notch_filter(emg_bp, fs, notch_freq=notch_freq)
    rect, env = rectify_and_envelope(emg_notch, fs, lp_cut=5.0)
    return {'fs': fs, 'raw': emg, 'bp': emg_bp, 'notch': emg_notch, 'rect': rect, 'env': env}

def extract_reps(processed, distance_seconds=0.5, prominence=0.25,
                 min_len_seconds=0.3, max_len_seconds=10.0):
    fs = processed["fs"]
    env = processed["env"]

    peaks, props = segment_reps_by_envelope(
        env, fs, distance_seconds=distance_seconds, prominence=prominence
    )
    peaks = np.asarray(peaks, dtype=int)

    rep_windows = []

    if len(peaks) < 3:
        return peaks, rep_windows

    min_len = int(min_len_seconds * fs) if min_len_seconds is not None else None
    max_len = int(max_len_seconds * fs) if max_len_seconds is not None else None

    for i in range(1, len(peaks) - 1):
        p_prev, p, p_next = peaks[i - 1], peaks[i], peaks[i + 1]

        start = (p_prev + p) // 2
        end   = (p + p_next) // 2

        start = max(0, start)
        end = min(len(env), end)

        L = end - start
        if min_len is not None and L < min_len:
            continue
        if max_len is not None and L > max_len:
            continue

        rep_windows.append((start, end, p))

    return peaks, rep_windows

def compute_rep_features(rep_windows, processed, time):
    features = []
    fs = processed['fs']
    sig = processed['notch']
    env = processed['env']
    for i, (start, end, p) in enumerate(rep_windows):
        seg = sig[start:end]
        rep_rms = rms(seg)
        rep_mdf = median_frequency(seg, fs)
        peak_time = time[p]
        features.append({'rep': i+1, 'start': start, 'end': end, 'peak_idx': p, 'peak_time': peak_time,
                         'rms': rep_rms, 'mdf': rep_mdf, 'env_peak': env[p]})
    return pd.DataFrame(features)

def detect_optimal_rep(features, lookback=2):
    df = features.copy().reset_index(drop=True)
    if len(df) == 0:
        return None, 'no_reps'
    rms_norm = (df['rms'] - df['rms'].min()) / (np.ptp(df['rms'].values) + 1e-8)
    mdf_norm = (df['mdf'] - df['mdf'].min()) / (np.ptp(df['mdf'].values) + 1e-8)
    score = rms_norm - mdf_norm[::-1].values
    df['score'] = score
    thr = 0.4 * np.nanmax(score)
    candidates = df.index[df['score'] >= thr].tolist()
    if not candidates:
        idx = int(df['score'].idxmax())
        return int(df.loc[idx, 'rep']), 'max_score'
    earliest = [c for c in candidates if c >= lookback]
    if earliest:
        return int(df.loc[earliest[0], 'rep']), 'threshold_cross'
    else:
        return int(df.loc[candidates[0], 'rep']), 'threshold_cross_early'

def plot_rep_trends(time, processed, features, optimal_rep=None, ground_truth_failure_rep=None, title=""):
    fig, axs = plt.subplots(3, 1, figsize=(9, 8), sharex=True)
    axs[0].plot(time, processed['raw'], label='raw', alpha=0.4)
    axs[0].plot(time, processed['bp'], label='bandpass')
    axs[0].set_ylabel('EMG (a.u.)')
    axs[0].legend(loc='upper right', fontsize='small')
    axs[1].plot(time, processed['env'], label='envelope')
    axs[1].scatter([features.loc[i,'peak_time'] for i in range(len(features))],
                   features['env_peak'], marker='x', label='rep peaks')
    axs[1].set_ylabel('Envelope (a.u.)')
    axs[1].legend(loc='upper right', fontsize='small')
    axs[2].plot(features['rep'], np.log(features['rms']), marker='o', label='RMS')
    axs[2].plot(features['rep'], np.log(features['mdf']), marker='s', label='MDF')
    axs[2].set_xlabel('Repetition')
    axs[2].set_ylabel('Feature value')
    axs[2].legend(loc='best', fontsize='small')
    if optimal_rep is not None:
        axs[2].axvline(optimal_rep-0.5, color='k', linestyle='--', label='optimal rep')
    if ground_truth_failure_rep is not None:
        axs[2].axvline(ground_truth_failure_rep-0.5, color='red', linestyle=':', label='Ground Truth Failure Rep')
    plt.tight_layout()
    plt.suptitle(title)
    plt.show()

def add_baseline_features(g):
    g = g.sort_values("rep").copy()
    base = g.head(3)[["rms", "mdf", "env_peak", "rep_duration"]].mean()

    for col in ["rms", "mdf", "env_peak", "rep_duration"]:
        g[f"{col}_rel_base"] = g[col] / (base[col] + 1e-9)
        g[f"{col}_delta_base"] = g[col] - base[col]

    # simple dynamics (no future leakage)
    for col in ["rms", "mdf", "env_peak"]:
        g[f"{col}_diff1"] = g[col].diff().fillna(0)
        g[f"{col}_roll3_mean"] = g[col].rolling(3, min_periods=1).mean()

    g["peak_time_diff1"] = g["peak_time"].diff().fillna(0)
    return g