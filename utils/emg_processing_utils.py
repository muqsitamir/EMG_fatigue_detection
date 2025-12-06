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

def extract_reps(time, processed, distance_seconds=0.5, prominence=0.25):
    fs = processed['fs']
    env = processed['env']
    peaks, props = segment_reps_by_envelope(env, fs, distance_seconds=distance_seconds, prominence=prominence)
    rep_windows = []
    for p in peaks:
        win_half = int(0.6 * fs)
        start = max(0, p - win_half)
        end = min(len(env), p + win_half)
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