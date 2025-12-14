from src.utils.data_utils import predict_fatigue_on_emg, load_model_bundle, load_and_extract_emg_from_c3d

def inference_for_single_test_file():
    bundle = load_model_bundle("./models/fatigue_model_bundle.joblib")

    signal_data, fs, _ = load_and_extract_emg_from_c3d("./test_data/test.c3d", "Emg_1")

    df_pred, trigger_rep = predict_fatigue_on_emg(
        signal_data=signal_data,
        fs=fs,
        model_bundle=bundle,
        file_id="test_file",
        distance_seconds=2.0,
        prominence=0.2
    )

    return df_pred, trigger_rep