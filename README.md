# Estimating Optimal Training Repetitions Using EMG-Based Muscle Fatigue Detection

A Python toolbox (and reference project) for **surface EMG (sEMG)** muscle fatigue detection during resistance training. It supports:

1) **Signal-analysis mode (no ML training required)** — filter EMG, segment repetitions, compute RMS/MDF trends, and estimate an “optimal rep” using lightweight heuristics.
2) **ML-trigger mode (recommended for deployment)** — train a classifier on per-repetition features and trigger a fatigue warning on new sessions.

**PyPI package:** `emg-fatigue-detection-toolbox`  
**Import name:** `emg_fd`

---

## Abstract
This project focuses on the measurement and analysis of surface electromyography (EMG) signals to detect muscle fatigue in human arms during resistance training. By analyzing EMG signal characteristics, we estimate the optimal number of repetitions per set for each individual, providing a personalized approach to training and rehabilitation.

---

## Research Objectives
- **Data Collection:** Collect surface EMG on the Biceps Brachii during resistance training until task failure.
- **Pipeline Development:** Provide a complete, reproducible data processing workflow in Python.
- **Feature Analysis:** Analyze extracted features like Root Mean Square (RMS) and Median Frequency (MDF) to identify fatigue thresholds.
- **Optimization:** Estimate “optimal reps” based on fatigue detection and validate estimates against subjective feedback.

---

## Hypothesis
- **Null Hypothesis ($H_0$):** There is no significant change in MDF or RMS of the EMG signal as contraction time increases during a set of repetitions.
- **Alternative Hypothesis ($H_1$):** As fatigue progresses, MDF decreases (slowing conduction velocity) while RMS increases (greater motor unit recruitment to maintain force).

---

## Methodology

### 1) Experimental Setup
- **Equipment:** Cometa MiniWave wireless EMG device with 3M Red Dot ECG surface electrodes.
- **Sampling Rate:** 2000 Hz.
- **Electrode Placement:** Biceps Brachii (2–3 cm inter-electrode distance).
- **Protocol:**
  - **Participants:** 12 healthy university students (aged 22–24) with varying fitness levels.
  - **Calibration:** Determine 1RM or suitable load (4×8 capability) for each subject.
  - **Exercise:** Bicep curls using 60–70% 1RM performed until task failure.
  - **Rest:** 90-second rest periods between sets (3–5 sets total).
  - **Ground Truth:** Subjective perceived soreness/fatigue recorded after each set.

<p align="center">
  <img src="https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/experiment.png" alt="Experiment" width="150"/>
  <img src="https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/experiment1.jpg" alt="Experiment 1" width="150"/>
</p>

### 2) Data Processing Pipeline
All processing steps are implemented in Python.

1. **Preprocessing & Filtering**
   - **Bandpass Filter:** 20–450 Hz to remove motion artifacts and high-frequency noise.
   - **Notch Filter:** 50 Hz to eliminate power line interference.
   - **Envelope Extraction:** Rectification + low-pass envelope smoothing.

2. **Segmentation**
   - Isolate individual repetitions (≈2-second windows) based on signal envelope peaks.

3. **Feature Extraction**
   - **RMS (Root Mean Square):** Represents signal amplitude and muscle force; often increases with fatigue.
     $$RMS = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$
   - **MDF (Median Frequency):** Frequency dividing the power spectrum into two equal parts; often decreases with fatigue.

4. **Analysis**
   - Trend analysis: monitor RMS↑ and MDF↓ across repetitions.
   - Fatigue threshold: identify significant deviation / trigger point in feature trends.
   - Optimal rep estimation: convert the trigger to an “optimal” repetition count.

---

## Installation

### Install from PyPI (recommended)
```bash
pip install emg-fatigue-detection-toolbox
```

### Install from source (development)
```bash
git clone https://github.com/muqsitamir/EMG_fatigue_detection.git
cd EMG_fatigue_detection
pip install -r requirements.txt
python main.py
```

---

## Package layout (post-packaging)
The PyPI package installs under the import name `emg_fd`.

Key modules you’ll interact with most:
- `emg_fd.src.utils.data_utils` — C3D loading, dataset creation, model bundle I/O, end-to-end inference helper
- `emg_fd.src.utils.emg_processing_utils` — filtering, envelope, repetition segmentation, RMS/MDF extraction
- `emg_fd.src.pipeline.train_model` — cross-validation, threshold selection, and final model training
- `emg_fd.src.pipeline.inference` — minimal inference demo / helpers
- `emg_fd.src.pipeline.signal_analysis_pipeline` — non-ML “optimal rep” heuristic workflow

> Tip: If you’re running from source and see import errors, prefer `pip install -e .` in your repo so imports match the packaged layout.

---

## Quickstart

### A) Run the bundled demo (pre-trained model + included test file)
A pre-trained model bundle and an example C3D file are included:
- Model: `models/fatigue_model_bundle.joblib`
- Example input: `test_data/test.c3d`

```bash
python -c "from emg_fd.src.pipeline.inference import inference_for_single_test_file; df_pred, trigger_rep = inference_for_single_test_file(); print('Fatigue detected at rep:', trigger_rep); print(df_pred[['rep','proba_used','pred']].head())"
```

### B) Use your own C3D recordings

#### Expected input format
- **Input format:** `.c3d` (analog EMG channels)
- **Default channel name:** `Emg_1` (configurable)

The loader prints available channel names if the requested channel is not found.

#### Recommended training folder layout
Create a `data/` folder at the repository root with:
- `data/Signals/` — your `.c3d` session files
- `data/filtered_signals.csv` — metadata/labels file

The label CSV is read with `sep=';'` and expects at least these columns:
- `id` — file identifier matching the C3D filename **without** `.c3d`
- `label` — fatigue onset repetition index (used to create `is_fatigued = 1` for reps `>= label`)

> Note: repetition indices in this pipeline are **1-based** (rep 1, 2, 3, ...). Provide your onset labels accordingly.

---

## API Reference (high-level)
Below is a practical overview of the most-used functions in the package. The exact signatures may evolve, but the intent and expected inputs/outputs are stable.

### `emg_fd.src.utils.data_utils`

#### `load_with_csv(folder_path, csv_file_path, channel_to_extract='Emg_1', ...)`
Loads a set of `.c3d` sessions and associates them with labels from the CSV.
- **Inputs**
  - `folder_path`: path to the folder containing `.c3d` files
  - `csv_file_path`: path to `filtered_signals.csv`
  - `channel_to_extract`: EMG channel name (default: `Emg_1`)
- **Returns**: an in-memory collection of sessions suitable for dataset generation.

#### `create_master_df(sessions, ...)`
Runs the full preprocessing → segmentation → feature extraction pipeline and creates the per-repetition table used for ML.
- **Inputs**: sessions returned by `load_with_csv`
- **Returns**: a pandas DataFrame and (by default) writes `./data/master_df.csv`

#### `load_and_extract_emg_from_c3d(c3d_path, channel_label='Emg_1', ...)`
Loads a single `.c3d` file and extracts the EMG signal.
- **Returns**: `(signal, fs, metadata)`

#### `load_model_bundle(model_path)`
Loads the saved model bundle (sklearn model + feature list + threshold + trigger config).

#### `predict_fatigue_on_emg(signal_data, fs, model_bundle, file_id='session', distance_seconds=2.0, prominence=0.2, ...)`
End-to-end inference on a single EMG session:
- preprocess + segment reps
- compute features
- run classifier
- apply trigger logic (threshold and optional M-of-N)

- **Returns**: `(df_pred, trigger_rep)` where:
  - `df_pred` contains per-rep predictions/probabilities
  - `trigger_rep` is the estimated fatigue onset rep (or `None` if never triggered)


### `emg_fd.src.utils.ml_utils`

#### `TrainConfig(n_splits=5, random_state=..., ...)`
Configuration object for training/evaluation.
- Holds CV parameters and any model/training settings used by the pipeline.


### `emg_fd.src.pipeline.train_model`

#### `run_training_eval(df, cfg, plot=False, ...)`
Runs grouped cross-validation (grouped by `file_id`) to reduce leakage across repetitions from the same session.
- **Inputs**
  - `df`: DataFrame like `master_df.csv`
  - `cfg`: `TrainConfig`
- **Returns**: `(results, best_threshold)`
  - `results` includes out-of-fold probabilities (`oof_proba`) and training metadata
  - `best_threshold` is selected to maximize balanced accuracy

#### `train_final_model(df, threshold, M, N, ...)`
Trains the final model on all available data and saves a model bundle to `models/fatigue_model_bundle.joblib`.
- `M, N` define the optional **M-of-N** trigger rule.


### `emg_fd.src.utils.eval_utils`

#### `evaluate_onset_timing(df, oof_proba, thr, M=2, N=3, m_of_n=True, ...)`
Evaluates how well the trigger timing matches labeled fatigue onset per session.
- **Returns**: `(timing_df, timing_summary)` with per-file timing deltas and summary metrics.


### `emg_fd.src.pipeline.inference`

#### `inference_for_single_test_file()`
Convenience demo function that runs inference on the bundled `test_data/test.c3d` using the bundled model.
- **Returns**: `(df_pred, trigger_rep)`

---

## End-to-End Examples

### 1) Generate the training dataset (`master_df.csv`)
```python
from emg_fd.src.utils.data_utils import load_with_csv, create_master_df

sessions = load_with_csv(
    folder_path="./data/Signals",
    csv_file_path="./data/filtered_signals.csv",
    channel_to_extract="Emg_1",
)

master_df = create_master_df(sessions)  # typically also writes ./data/master_df.csv
```

### 2) Train + evaluate the fatigue classifier
```python
import pandas as pd

from emg_fd.src.utils.ml_utils import TrainConfig
from emg_fd.src.pipeline.train_model import run_training_eval, train_final_model
from emg_fd.src.utils.eval_utils import evaluate_onset_timing

# Load the generated dataset
df = pd.read_csv("./data/master_df.csv")

# Cross-validated evaluation
cfg = TrainConfig(n_splits=5)
results, best_t = run_training_eval(df, cfg, plot=True)

# Optional: M-of-N consecutive/near-consecutive probabilities above threshold
m, n = 2, 3

timing_df, timing_summary = evaluate_onset_timing(
    df,
    results["oof_proba"],
    thr=best_t,
    M=m,
    N=n,
    m_of_n=True,
)

# Train final model on all data and save bundle for inference
train_final_model(df, best_t, m, n)
```

### 3) Run inference on a new session
```python
from emg_fd.src.utils.data_utils import (
    load_model_bundle,
    load_and_extract_emg_from_c3d,
    predict_fatigue_on_emg,
)
import os
import emg_fd

package_dir = os.path.dirname(emg_fd.__file__)
model_path = os.path.join(package_dir, "models", "fatigue_model_bundle.joblib")

bundle = load_model_bundle(model_path)

signal, fs, _ = load_and_extract_emg_from_c3d("/path/to/session.c3d", "Emg_1")

df_pred, trigger_rep = predict_fatigue_on_emg(
    signal_data=signal,
    fs=fs,
    model_bundle=bundle,
    file_id="my_session",
    distance_seconds=2.0,
    prominence=0.2,
)

print("Fatigue trigger rep:", trigger_rep)
print(df_pred[["rep", "proba_used", "pred"]].head())
```

---

## Key parameters you may need to tune
- **Channel selection:** `channel_to_extract` / `channel_label` (default `Emg_1`).
- **Power-line frequency:** notch filter defaults to **50 Hz**. If your mains is 60 Hz, update the notch frequency in the processing utilities.
- **Repetition segmentation:**
  - `distance_seconds`: minimum time between envelope peaks (depends on rep cadence)
  - `prominence`: how “strong” a peak must be (depends on signal quality and electrode placement)

If repetition detection is unstable, tune `distance_seconds` first (cadence), then `prominence` (noise/quality).

**Figure 1: Raw vs. Filtered EMG Signal**
> ![Raw vs Filtered Signal](https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/signal_filtering_example.png)
> *Comparison of raw EMG signal and the signal after Bandpass and Notch filtering.*

**Figure 2: Optimal Repetition Estimation**
> ![Optimal Rep Estimation](https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/peak_detection.png)
> *Peak detection on a 2-second interval.*

---

## Performance Summary (bundled model on current data)

**• `files_total = 32`**  
Total number of evaluated sessions (`file_id`s).

**• `files_with_onset_and_trigger = 26`**  
Sessions where a labeled fatigue onset exists *and* the trigger fired at least once.

**• `never_triggered = 2`**  
Sessions where the trigger never fired — these represent **file-level false negatives**.

**• `mean_delta_reps = 0.308`, `median_delta_reps = 0.0`**  
On average, the trigger fires about **0.3 reps late**, while the median indicates it fires **exactly on time** in half of the sessions.

**• `mae_reps = 1.231`**  
Mean absolute timing error: the trigger is typically within **about 1 rep** of the true onset.

**• `pct_within_0_reps = 0.269`**  
The trigger fires **exactly on the onset rep** in ~26.9% of cases.

**• `pct_within_1_rep = 0.692`**  
About **69.2%** of triggers occur within **±1 rep** of the true fatigue onset.

**• `pct_within_2_reps = 0.885`**  
Roughly **88.5%** of triggers occur within **±2 reps** of onset.

**• `early_rate = 0.308`, `late_rate = 0.423`**  

##### Among sessions where the trigger fires:
- Early in **30.8%** of cases
- Late in **42.3%** of cases

This reflects a system slightly biased toward late detection, which reduces false alarms but may delay early-warning functionality.

---

## Citation
If you use this package or repository in academic work, please cite the project/repository.

---

## License
Add your license here (e.g., MIT, Apache-2.0, GPL-3.0). If this repo already has a LICENSE file, this section can mirror it.
