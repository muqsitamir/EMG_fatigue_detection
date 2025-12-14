from pathlib import Path
from importlib import resources

from emg_fd.src.utils.data_utils import (
    predict_fatigue_on_emg,
    load_model_bundle,
    load_and_extract_emg_from_c3d,
)

def _get_model_path(model_path: str | Path | None):
    if model_path:
        p = Path(model_path).expanduser()
        return p if p.is_absolute() else (Path.cwd() / p)

    # packaged pretrained model (you must include this file as package data)
    return resources.files("emg_fd").joinpath("models/fatigue_model_bundle.joblib")


def inference_for_single_test_file(file_path, channel_label, model_path: str | Path | None = None):
    model_ref = _get_model_path(model_path)

    if not isinstance(model_ref, Path):
        with resources.as_file(model_ref) as real_path:
            bundle = load_model_bundle(str(real_path))
    else:
        bundle = load_model_bundle(str(model_ref))

    signal_data, fs, _ = load_and_extract_emg_from_c3d(file_path, channel_label)

    df_pred, trigger_rep = predict_fatigue_on_emg(
        signal_data=signal_data,
        fs=fs,
        model_bundle=bundle,
        file_id="test_file",
        distance_seconds=2.0,
        prominence=0.2,
    )

    return df_pred, trigger_rep