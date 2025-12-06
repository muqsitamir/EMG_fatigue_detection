from utils.emg_processing_utils import process_emg, extract_reps, compute_rep_features, detect_optimal_rep, plot_rep_trends


def signal_analysis_pipeline(data):
    for d in data:
        signal_data = d["signal_data"]
        fs = d["fs"]
        signal_label = d["label"]
        time = d["time"]
        if signal_data is None or fs is None:
            print(
                "Error: No signal data or sampling rate available. Please ensure the C3D loading cell ran successfully.")
        else:
            print(f"Applying algorithm to '{signal_label}' (Sampling Rate: {fs} Hz)")

            # 1. Process EMG signal
            processed_emg_c3d = process_emg(time, signal_data, fs=fs, lowcut=20, highcut=450, notch_freq=50.0)

            # 2. Extract repetitions
            # Adjust distance_seconds and prominence as needed for your specific data
            peaks_c3d, rep_windows_c3d = extract_reps(processed_emg_c3d, distance_seconds=2.1, prominence=0.35)

            # 3. Compute features per repetition
            features_c3d = compute_rep_features(rep_windows_c3d, processed_emg_c3d, time)

            print(f'Detected {len(features_c3d)} reps from {d["id"]}.')
            # display(features_c3d)

            # 4. Detect optimal repetition
            opt_rep_c3d, reason_c3d = detect_optimal_rep(features_c3d)
            print('Optimal rep estimate from C3D data:', opt_rep_c3d, 'method:', reason_c3d)

            # 5. Plot the trends
            plot_rep_trends(time, processed_emg_c3d, features_c3d, optimal_rep=opt_rep_c3d,
                            title=f'Trends for {d.get("name")}')