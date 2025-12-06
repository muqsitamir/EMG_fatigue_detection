from pyomeca import Analogs
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

from utils.emg_processing_utils import compute_rep_features, extract_reps, process_emg, add_baseline_features


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

def plot_emg_signals(folder_path="./data/Signals", channel_to_extract='Emg_1'):
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
