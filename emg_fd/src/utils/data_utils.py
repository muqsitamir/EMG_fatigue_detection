from pyomeca import Analogs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import joblib

from emg_fd.src.utils.emg_processing_utils import compute_rep_features, extract_reps, process_emg, add_baseline_features


def load_and_extract_emg_from_c3d(file_path: str, channel_label: str):
    """
    Loads a C3D file and extracts the EMG signal for a specified channel label.

    Args:
        file_path (str): The path to the C3D file.
        channel_label (str): The exact label of the EMG channel to extract.

    Returns:
        tuple: A tuple containing:
            - np.ndarray: The EMG signal data for the specified channel.
            - float: The sampling rate of the analog data.
        Returns (None, None) if the channel is not found or no analog data exists.
    """
    try:
        analog_obj = Analogs.from_c3d(file_path)

        if analog_obj.values.size == 0:
            print(f"No analog data found in the C3D file: {file_path}")
            return None, None, None

        channel_names = list(analog_obj.coords['channel'].values)
        try:
            channel_index = channel_names.index(channel_label)
        except ValueError:
            print(f"Channel '{channel_label}' not found in {file_path}. Available channels: {', '.join(channel_names)}")
            return None, None, None

        signal_to_plot = analog_obj.values[channel_index, :]
        sampling_rate = analog_obj.rate

        print(f"Loaded C3D file: {file_path}")
        print(f"Extracted signal for channel: '{channel_label}'")
        print(f"data shape: {signal_to_plot.shape}")

        return signal_to_plot, sampling_rate, channel_label

    except Exception as e:
        print(f"Error loading or processing C3D file {file_path}: {e}")
        return None, None, None

def plot_emg_signals(folder_path, channel_to_extract):
    data = []
    for file in os.listdir(folder_path):
        if file.endswith(".c3d"):
          full_path = os.path.join(folder_path, file)
          signal_data, fs, signal_label = load_and_extract_emg_from_c3d(full_path, channel_to_extract)


          if signal_data is not None:
              time = np.arange(len(signal_data)) / fs

              plt.figure(figsize=(12, 6))
              plt.plot(time, signal_data)
              plt.title(f'Signal from {signal_label} in {file}')
              plt.xlabel('Time (s)')
              plt.ylabel('Amplitude')
              plt.grid(True)
              plt.show()
              data.append({"signal_data":signal_data, "fs":fs, "name":str(file), "time":time})
          else:
              print(f"Could not plot signal for channel '{channel_to_extract}'.")

    return data

def load_with_csv(folder_path="./data/Signals", csv_file_path="./data/filtered_signals.csv", channel_to_extract='Emg_1'):
    extracted_data_list = []
    df_labels = pd.read_csv(csv_file_path, sep=';', index_col=False)
    df_labels = df_labels.dropna(axis=1, how='all')
    for index, row in df_labels.iterrows():
        file_id = row['id']
        file_label = row['label']
        c3d_filename = file_id + ".c3d"
        c3d_file_path = os.path.join(folder_path, c3d_filename)

        signal_data, fs, signal_label = load_and_extract_emg_from_c3d(c3d_file_path, channel_to_extract)

        if signal_data is not None:
            time = np.arange(len(signal_data)) / fs

            extracted_data_list.append({
                "id": file_id,
                "label": file_label,
                "signal_data": signal_data,
                "fs": fs,
                "name": c3d_filename,
                "time": time
            })
            print(f"Successfully associated data for ID: {file_id} with label: {file_label}")
        else:
            print(f"Skipping ID: {file_id}. Could not load/process C3D file or channel.")
            extracted_data_list.append({
                "id": file_id,
                "label": file_label,
                "signal_data": None,
                "fs": None,
                "name": c3d_filename,
                "time": None
            })

    return extracted_data_list

def create_master_df(data):
    all_reps_data = []

    print("Processing files to generate ML dataset...")

    for item in data:
        if item['signal_data'] is None:
            continue

        failure_rep_threshold = item['label']

        processed = process_emg(item['time'], item['signal_data'], fs=item['fs'])

        peaks, rep_windows = extract_reps(processed, distance_seconds=2.0, prominence=0.2)

        # 3. Compute Features
        df_features = compute_rep_features(rep_windows, processed, item['time'])
        df_features["rep_duration"] = df_features["end"] - df_features["start"]

        # --- Labeling Logic ---
        df_features['is_fatigued'] = df_features['rep'].apply(lambda x: 1 if x >= failure_rep_threshold else 0)

        df_features['file_id'] = item["id"]

        df_features = df_features.groupby("file_id", group_keys=False).apply(add_baseline_features)

        all_reps_data.append(df_features)

    master_df = pd.concat(all_reps_data, ignore_index=True)

    master_df = master_df.replace([np.inf, -np.inf], np.nan).dropna()

    print(f"Dataset created with {len(master_df)} total repetitions.")
    print(master_df['is_fatigued'].value_counts())
    master_df.to_csv('./data/master_df.csv', index=False)
    return master_df

# -------------------------
# Inference helpers
# -------------------------

def _safe_extract_reps(processed, time=None, distance_seconds=2.0, prominence=0.2):
    """Call extract_reps with whichever signature is available in your project."""
    try:
        # Your current create_master_df uses this signature
        return extract_reps(processed, distance_seconds=distance_seconds, prominence=prominence)
    except TypeError:
        # Fallback to the signature you showed earlier: extract_reps(time, processed, ...)
        if time is None:
            raise
        return extract_reps(time, processed, distance_seconds=distance_seconds, prominence=prominence)


def _align_features_for_model(df_feat: pd.DataFrame, feature_cols):
    """Ensure inference features match training columns exactly."""
    X = df_feat.copy()
    # Keep numeric columns only
    X = X.select_dtypes(include=[np.number])

    # Add any missing columns
    for c in feature_cols:
        if c not in X.columns:
            X[c] = 0.0

    # Drop any extra columns
    X = X[feature_cols]
    return X


def load_model_bundle(bundle_path: str = "./models/fatigue_model_bundle.joblib"):
    """Load a saved model bundle created by save_model_bundle()."""
    bundle = joblib.load(bundle_path)
    return bundle


def save_model_bundle(model, feature_cols, best_threshold: float,
                      bundle_path: str = "./models/fatigue_model_bundle.joblib",
                      trigger_M: int = 2, trigger_N: int = 3, smooth_alpha: float | None = None):
    """Save everything needed for inference in one file."""
    os.makedirs(os.path.dirname(bundle_path), exist_ok=True)
    bundle = {
        "model": model,
        "feature_cols": list(feature_cols),
        "best_threshold": float(best_threshold),
        "trigger_M": int(trigger_M),
        "trigger_N": int(trigger_N),
        "smooth_alpha": None if smooth_alpha is None else float(smooth_alpha),
    }
    joblib.dump(bundle, bundle_path)
    return bundle_path


def trigger_index_m_of_n(proba: np.ndarray, thr: float, M: int = 2, N: int = 3):
    """Return first rep index where >=M of the last N reps exceed thr."""
    above = (proba >= thr).astype(int)
    for i in range(len(above)):
        win = above[max(0, i - N + 1): i + 1]
        if win.sum() >= M:
            return int(i)
    return None


def predict_fatigue_on_emg(
    signal_data: np.ndarray,
    fs: float,
    model_bundle: dict,
    file_id: str = "new",
    distance_seconds: float = 2.0,
    prominence: float = 0.2,
):
    """Preprocess a new EMG signal, extract reps, compute features, and predict fatigue per rep.

    Args:
        signal_data: raw EMG samples (1D array)
        fs: sampling rate
        model_bundle: dict from load_model_bundle/save_model_bundle
        file_id: id used for grouping/baseline features
        distance_seconds/prominence: rep peak detection params

    Returns:
        df_pred: per-rep dataframe with probabilities and binary predictions
        trigger_rep: first rep (by df_pred['rep']) where trigger condition fires, else None
    """
    model = model_bundle["model"]
    feature_cols = model_bundle["feature_cols"]
    thr = float(model_bundle.get("best_threshold", 0.5))
    M = int(model_bundle.get("trigger_M", 2))
    N = int(model_bundle.get("trigger_N", 3))
    smooth_alpha = model_bundle.get("smooth_alpha", None)

    time = np.arange(len(signal_data)) / fs

    processed = process_emg(time, signal_data, fs=fs)

    peaks, rep_windows = _safe_extract_reps(
        processed,
        time=time,
        distance_seconds=distance_seconds,
        prominence=prominence,
    )

    if len(rep_windows) == 0:
        df_empty = pd.DataFrame(columns=["rep", "proba", "pred"])
        return df_empty, None

    df_feat = compute_rep_features(rep_windows, processed, time)
    df_feat["file_id"] = file_id

    # Optional: keep rep_duration if it varies; if constant it will be harmless but not useful.
    if "end" in df_feat.columns and "start" in df_feat.columns:
        df_feat["rep_duration"] = df_feat["end"] - df_feat["start"]

    # Add baseline-normalized features (uses first reps inside this new file)
    df_feat = (
        df_feat
        .groupby("file_id", group_keys=False)[df_feat.columns]
        .apply(add_baseline_features)
    )

    # Align features exactly like training
    X_new = _align_features_for_model(df_feat, feature_cols)

    proba = model.predict_proba(X_new)[:, 1]

    # Optional smoothing before triggering/prediction
    if smooth_alpha is not None:
        s = pd.Series(proba)
        proba_used = s.ewm(alpha=float(smooth_alpha), adjust=False).mean().to_numpy()
    else:
        proba_used = proba

    pred = (proba_used >= thr).astype(int)

    df_pred = df_feat.copy()
    df_pred["proba"] = proba
    df_pred["proba_used"] = proba_used
    df_pred["pred"] = pred

    # Trigger rep index (0-based index in df_pred order)
    trig_idx = trigger_index_m_of_n(proba_used, thr=thr, M=M, N=N)
    if trig_idx is None:
        trigger_rep = None
    else:
        trigger_rep = int(df_pred.iloc[trig_idx]["rep"]) if "rep" in df_pred.columns else int(trig_idx)

    return df_pred, trigger_rep
