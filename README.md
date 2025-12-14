# Estimating Optimal Training Repetitions Using EMG-Based Muscle Fatigue Detection

## Team: BioTeam7
**Course:** Biomedical Signal Processing — 2025/26/1  
**Team Members:** Dario Ranieri, Islam Muhammad Muqsit, Zsuzsanna Rohán

---

## Abstract
This project focuses on the independent measurement and analysis of surface electromyography (EMG) signals to detect muscle fatigue in human arms during resistance training. By analyzing EMG signal characteristics, we aim to estimate the optimal number of repetitions per set for each individual, providing a personalized approach to training and rehabilitation.

---

## Research Objectives
* **Data Collection:** Conduct self-collected biomedical signal measurements using surface EMG on the Biceps Brachii during resistance training until task failure.
* **Pipeline Development:** Design and document a complete, reproducible data processing workflow in Python.
* **Feature Analysis:** Perform statistical and mathematical analysis on extracted features like Root Mean Square (RMS) and Median Frequency (MDF) to identify fatigue thresholds.
* **Optimization:** Estimate the optimal number of repetitions based on fatigue detection and validate these estimates against subjective feedback.

---

##  Hypothesis
* **Null Hypothesis ($H_0$):** There is no significant change in the Median Frequency (MDF) or Root Mean Square (RMS) of the EMG signal as contraction time increases during a set of repetitions.
* **Alternative Hypothesis ($H_1$):** As muscle fatigue progresses, the Median Frequency (MDF) will significantly decrease due to slowing muscle fiber conduction velocity, while the RMS amplitude will increase due to motor unit recruitment to maintain force.

---

## Methodology

### 1. Experimental Setup
* **Equipment:** Cometa MiniWave wireless EMG device with 3M Red Dot ECG surface electrodes.
* **Sampling Rate:** 2000 Hz.
* **Electrode Placement:** Biceps Brachii (2-3 cm inter-electrode distance).
* **Protocol:**
    * **Participants:** 12 healthy university students (aged 22-24) with varying fitness levels.
    * **Calibration:** Determined 1-Repetition Maximum (1RM) or suitable load (4x8 capability) for each subject.
    * **Exercise:** Bicep curls using a load of 60-70% 1RM performed until task failure.
    * **Rest:** 90-second rest periods between sets (3-5 sets total).
    * **Ground Truth:** Subjective feedback on perceived soreness/fatigue recorded after each set.

<p align="center">
  <img src="https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/experiment.png" alt="Experiment" width="150"/>
  <img src="https://raw.githubusercontent.com/muqsitamir/EMG_fatigue_detection/main/docs/images/experiment1.jpg" alt="Experiment 1" width="150"/>
</p>


### 2. Data Processing Pipeline
All processing steps are implemented in Python within this repository.

1.  **Preprocessing & Filtering:**
    * **Bandpass Filter:** 20–450 Hz to remove motion artifacts and high-frequency noise.
    * **Notch Filter:** 50 Hz to eliminate power line interference.
    * **Envelope Extraction:** Signal rectification followed by a Low-pass Envelope for smoothing.

2.  **Segmentation:**
    * Isolation of individual repetitions (approx. 2-second windows) based on signal envelopes.

3.  **Feature Extraction:**
    * **RMS (Root Mean Square):** Represents signal amplitude and muscle force. Increases with fatigue.
        $$RMS = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$
    * **MDF (Median Frequency):** The frequency dividing the power spectrum into two equal parts. Decreases with fatigue due to metabolic changes.

4.  **Analysis:**
    * **Trend Analysis:** Monitoring the increase in RMS and decrease in MDF across repetitions.
    * **Fatigue Threshold:** Identifying the crossover or significant deviation points in feature trends.
    * **Optimal Rep Estimation:** Correlating the identified fatigue point with the "optimal" repetition count.

---

## Results & Validation

The project successfully identifies trends where RMS increases and MDF decreases as the set progresses toward failure. The algorithm's estimated "optimal repetition" is compared against the user's recorded subjective soreness point (Ground Truth) to validate accuracy.

### Visualizations

**Figure 1: Raw vs. Filtered EMG Signal**
> ![Raw vs Filtered Signal](docs/images/signal_filtering_example.png)
> *Comparison of raw EMG signal and the signal after Bandpass and Notch filtering.*

**Figure 2: Optimal Repetition Estimation**
> ![Optimal Rep Estimation](docs/images/peak_detection.png)
> *Peak detection on a 2-second interval.*

---

## Installation & Usage

**Prerequisites:** Ensure you have **Python 3.10** installed.


1. **Clone the repository:**
   ```bash
   git clone https://github.com/muqsitamir/EMG_fatigue_detection.git
   cd EMG_fatigue_detection
   pip install -r requirements.txt
   python main.py
   


## Performance Summary

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

## End-to-End Workflow

This repository supports **two complementary usage modes**:
1. **Signal-analysis mode (no ML training required):** process a set, segment repetitions, compute RMS/MDF trends, and estimate an “optimal rep” using a lightweight heuristic.
2. **ML-trigger mode (recommended for deployment):** train a classifier on labeled repetition features and use it to trigger a fatigue warning on new sessions.

### 1) Quick demo (runs on included test data)

A pre-trained model bundle and an example C3D file are included:
- Model: `models/fatigue_model_bundle.joblib`
- Example input: `test_data/test.c3d`

Run a simple inference demo:
```bash
python -c "from pipeline.inference import inference_for_single_test_file; df_pred, trigger_rep = inference_for_single_test_file(); print('Fatigue detected at rep:', trigger_rep); print(df_pred[['rep','proba_used','pred']].head())"
```

### 2) Use your own C3D recordings

#### Expected file format
- **Input format:** `.c3d` (analog EMG channels)
- **Default channel name:** `Emg_1` (configurable)

The loader prints available channel names if the requested channel is not found.

#### Recommended folder layout for training
Create a `data/` folder at the repository root with:
- `data/Signals/` — your `.c3d` session files
- `data/filtered_signals.csv` — metadata/labels file

The label CSV is read with `sep=';'` and expects at least these columns:
- `id` — the file identifier (must match the C3D filename **without** the `.c3d` extension)
- `label` — the **fatigue onset repetition index** (used to create `is_fatigued = 1` for reps `>= label`)

> Note: repetition indices in the pipeline are **1-based** (rep 1, 2, 3, ...). Provide your onset label accordingly.

### 3) Generate the training dataset (`master_df.csv`)

The ML pipeline trains on a per-repetition feature table saved to `./data/master_df.csv`.

In Python (e.g., a notebook or a small script):

```python
from emg_fd.src.utils.data_utils import load_with_csv, create_master_df

# 1) Load and extract EMG from C3D files using the label CSV
sessions = load_with_csv(
    folder_path="./data/Signals",
    csv_file_path="./data/filtered_signals.csv",
    channel_to_extract="Emg_1",
)

# 2) Process signals, segment reps, compute features, and write ./data/master_df.csv
master_df = create_master_df(sessions)
```

### 4) Train + evaluate the fatigue classifier

Training is grouped by `file_id` to reduce data leakage across repetitions from the same session.

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

# Optional: require M-of-N consecutive/near-consecutive probabilities above threshold
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

This will produce `models/fatigue_model_bundle.joblib`, containing:
- the trained sklearn model
- the feature column list (ensures inference uses the same feature set)
- the tuned probability threshold
- trigger configuration (M-of-N)

### 5) Run inference on a new session

```python
from emg_fd.src.utils.data_utils import load_model_bundle, load_and_extract_emg_from_c3d, predict_fatigue_on_emg

bundle = load_model_bundle("emg_fd/models/fatigue_model_bundle.joblib")

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

## Key Parameters You May Need to Tune

- **Channel selection:** `channel_to_extract` / `channel_label` (default `Emg_1`).
- **Power-line frequency:** notch filter defaults to **50 Hz**. If your mains frequency is 60 Hz, update `notch_freq` in `process_emg()`.
- **Repetition segmentation:**
  - `distance_seconds`: minimum time between envelope peaks (depends on rep cadence)
  - `prominence`: how “strong” a peak must be (depends on signal quality and electrode placement)

If rep detection is unstable, tune `distance_seconds` first (cadence) and then `prominence` (noise/quality).

---

## Suggested Entry Points for Developers

If you want to extend or modify the pipeline, these files are the best starting points:
- `utils/emg_processing_utils.py` — filtering, envelope, repetition segmentation, RMS/MDF extraction
- `utils/data_utils.py` — C3D loading, dataset creation, model-bundle I/O, inference helper
- `pipeline/train_model.py` — cross-validation, threshold selection, and final model training
- `pipeline/inference.py` — minimal inference example on a single C3D file
- `pipeline/signal_analysis_pipeline.py` — non-ML “optimal rep” heuristic workflow

> For reproducible experiments, avoid editing the feature definitions mid-project; instead, version feature changes and re-generate `master_df.csv`.

---

## Out-of-the-Box Results (Bundled Model on Current Data)

This repository includes a ready-to-use model bundle at `models/fatigue_model_bundle.joblib`. The results below document the **current** performance of the bundled model/training configuration on the dataset produced from the repository’s current `data/Signals/*.c3d` files and `data/filtered_signals.csv` labels.

> **Reproducibility note:** These results are **dataset- and configuration-dependent**. If you add/remove sessions, change repetition segmentation parameters (`distance_seconds`, `prominence`), adjust filtering (50/60 Hz notch), or modify the feature set, your numbers will change. This section is included so new users know what to expect when using the model “out of the box.”

### Dataset snapshot

- **Total repetitions:** 396
- **Class balance (`is_fatigued`):**
  - 0 (not fatigued): 264
  - 1 (fatigued): 132
- **Total evaluated sessions (`file_id`):** 32

### Cross-validated (OOF) classification performance

Threshold selection was performed on out-of-fold (OOF) probabilities by maximizing **balanced accuracy**.

- **OOF best threshold:** `0.385`
- **OOF Balanced Accuracy:** `0.856`
- **OOF ROC AUC:** `0.933`
- **OOF PR AUC:** `0.868`

**OOF confusion matrix (repetition-level):**

|               | Pred: Not fatigued | Pred: Fatigued |
|---|---:|---:|
| **True: Not fatigued** | 216 | 48 |
| **True: Fatigued**     | 14  | 118 |

**Classification report (OOF):**

- Not fatigued (0): precision **0.939**, recall **0.818**, F1 **0.874** (support 264)
- Fatigued (1): precision **0.711**, recall **0.894**, F1 **0.792** (support 132)
- Overall accuracy: **0.843** (396 repetitions)

# Estimating Optimal Training Repetitions Using EMG-Based Muscle Fatigue Detection

## Team: BioTeam7
**Course:** Biomedical Signal Processing — 2025/26/1  
**Team Members:** Dario Ranieri, Islam Muhammad Muqsit, Zsuzsanna Rohán

---

## Abstract
This project focuses on the independent measurement and analysis of surface electromyography (EMG) signals to detect muscle fatigue in human arms during resistance training. By analyzing EMG signal characteristics, we aim to estimate the optimal number of repetitions per set for each individual, providing a personalized approach to training and rehabilitation.

---

## Research Objectives
* **Data Collection:** Conduct self-collected biomedical signal measurements using surface EMG on the Biceps Brachii during resistance training until task failure.
* **Pipeline Development:** Design and document a complete, reproducible data processing workflow in Python.
* **Feature Analysis:** Perform statistical and mathematical analysis on extracted features like Root Mean Square (RMS) and Median Frequency (MDF) to identify fatigue thresholds.
* **Optimization:** Estimate the optimal number of repetitions based on fatigue detection and validate these estimates against subjective feedback.

---

##  Hypothesis
* **Null Hypothesis ($H_0$):** There is no significant change in the Median Frequency (MDF) or Root Mean Square (RMS) of the EMG signal as contraction time increases during a set of repetitions.
* **Alternative Hypothesis ($H_1$):** As muscle fatigue progresses, the Median Frequency (MDF) will significantly decrease due to slowing muscle fiber conduction velocity, while the RMS amplitude will increase due to motor unit recruitment to maintain force.

---

## Methodology

### 1. Experimental Setup
* **Equipment:** Cometa MiniWave wireless EMG device with 3M Red Dot ECG surface electrodes.
* **Sampling Rate:** 2000 Hz.
* **Electrode Placement:** Biceps Brachii (2-3 cm inter-electrode distance).
* **Protocol:**
    * **Participants:** 12 healthy university students (aged 22-24) with varying fitness levels.
    * **Calibration:** Determined 1-Repetition Maximum (1RM) or suitable load (4x8 capability) for each subject.
    * **Exercise:** Bicep curls using a load of 60-70% 1RM performed until task failure.
    * **Rest:** 90-second rest periods between sets (3-5 sets total).
    * **Ground Truth:** Subjective feedback on perceived soreness/fatigue recorded after each set.

<p align="center">
 <img src="docs/images/experiment.png" alt="Experiment" width="150"/>
<img src="docs/images/experiment1.jpg" alt="Experiment 1" width="150"/>
</p>

### 2. Data Processing Pipeline
All processing steps are implemented in Python within this repository.

1.  **Preprocessing & Filtering:**
    * **Bandpass Filter:** 20–450 Hz to remove motion artifacts and high-frequency noise.
    * **Notch Filter:** 50 Hz to eliminate power line interference.
    * **Envelope Extraction:** Signal rectification followed by a Low-pass Envelope for smoothing.

2.  **Segmentation:**
    * Isolation of individual repetitions (approx. 2-second windows) based on signal envelopes.

3.  **Feature Extraction:**
    * **RMS (Root Mean Square):** Represents signal amplitude and muscle force. Increases with fatigue.
        $$RMS = \sqrt{\frac{1}{n} \sum_{i=1}^{n} x_i^2}$$
    * **MDF (Median Frequency):** The frequency dividing the power spectrum into two equal parts. Decreases with fatigue due to metabolic changes.

4.  **Analysis:**
    * **Trend Analysis:** Monitoring the increase in RMS and decrease in MDF across repetitions.
    * **Fatigue Threshold:** Identifying the crossover or significant deviation points in feature trends.
    * **Optimal Rep Estimation:** Correlating the identified fatigue point with the "optimal" repetition count.

---

## Results & Validation

The project successfully identifies trends where RMS increases and MDF decreases as the set progresses toward failure. The algorithm's estimated "optimal repetition" is compared against the user's recorded subjective soreness point (Ground Truth) to validate accuracy.

### Visualizations

**Figure 1: Raw vs. Filtered EMG Signal**
> ![Raw vs Filtered Signal](docs/images/signal_filtering_example.png)
> *Comparison of raw EMG signal and the signal after Bandpass and Notch filtering.*

**Figure 2: Optimal Repetition Estimation**
> ![Optimal Rep Estimation](docs/images/peak_detection.png)
> *Peak detection on a 2-second interval.*

---

## Installation & Usage

**Prerequisites:** Ensure you have **Python 3.10** installed.


1. **Clone the repository:**
   ```bash
   git clone https://github.com/muqsitamir/EMG_fatigue_detection.git
   cd EMG_fatigue_detection
   pip install -r requirements.txt
   python main.py
   


## Performance Summary

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

## End-to-End Workflow (New Users)

This repository supports **two complementary usage modes**:
1. **Signal-analysis mode (no ML training required):** process a set, segment repetitions, compute RMS/MDF trends, and estimate an “optimal rep” using a lightweight heuristic.
2. **ML-trigger mode (recommended for deployment):** train a classifier on labeled repetition features and use it to trigger a fatigue warning on new sessions.

### 1) Quick demo (runs on included test data)

A pre-trained model bundle and an example C3D file are included:
- Model: `models/fatigue_model_bundle.joblib`
- Example input: `test_data/test.c3d`

Run a simple inference demo:
```bash
python -c "from pipeline.inference import inference_for_single_test_file; df_pred, trigger_rep = inference_for_single_test_file(); print('Fatigue detected at rep:', trigger_rep); print(df_pred[['rep','proba_used','pred']].head())"
```

### 2) Use your own C3D recordings

#### Expected file format
- **Input format:** `.c3d` (analog EMG channels)
- **Default channel name:** `Emg_1` (configurable)

The loader prints available channel names if the requested channel is not found.

#### Recommended folder layout for training
Create a `data/` folder at the repository root with:
- `data/Signals/` — your `.c3d` session files
- `data/filtered_signals.csv` — metadata/labels file

The label CSV is read with `sep=';'` and expects at least these columns:
- `id` — the file identifier (must match the C3D filename **without** the `.c3d` extension)
- `label` — the **fatigue onset repetition index** (used to create `is_fatigued = 1` for reps `>= label`)

> Note: repetition indices in the pipeline are **1-based** (rep 1, 2, 3, ...). Provide your onset label accordingly.

### 3) Generate the training dataset (`master_df.csv`)

The ML pipeline trains on a per-repetition feature table saved to `./data/master_df.csv`.

In Python (e.g., a notebook or a small script):

```python
from emg_fd.src.utils.data_utils import load_with_csv, create_master_df

# 1) Load and extract EMG from C3D files using the label CSV
sessions = load_with_csv(
    folder_path="./data/Signals",
    csv_file_path="./data/filtered_signals.csv",
    channel_to_extract="Emg_1",
)

# 2) Process signals, segment reps, compute features, and write ./data/master_df.csv
master_df = create_master_df(sessions)
```

### 4) Train + evaluate the fatigue classifier

Training is grouped by `file_id` to reduce data leakage across repetitions from the same session.

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

# Optional: require M-of-N consecutive/near-consecutive probabilities above threshold
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

This will produce `models/fatigue_model_bundle.joblib`, containing:
- the trained sklearn model
- the feature column list (ensures inference uses the same feature set)
- the tuned probability threshold
- trigger configuration (M-of-N)

### 5) Run inference on a new session

```python
from emg_fd.src.utils.data_utils import load_model_bundle, load_and_extract_emg_from_c3d, predict_fatigue_on_emg

bundle = load_model_bundle("emg_fd/models/fatigue_model_bundle.joblib")

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

## Key Parameters You May Need to Tune

- **Channel selection:** `channel_to_extract` / `channel_label` (default `Emg_1`).
- **Power-line frequency:** notch filter defaults to **50 Hz**. If your mains frequency is 60 Hz, update `notch_freq` in `process_emg()`.
- **Repetition segmentation:**
  - `distance_seconds`: minimum time between envelope peaks (depends on rep cadence)
  - `prominence`: how “strong” a peak must be (depends on signal quality and electrode placement)

If rep detection is unstable, tune `distance_seconds` first (cadence) and then `prominence` (noise/quality).

---

## Suggested Entry Points for Developers

If you want to extend or modify the pipeline, these files are the best starting points:
- `utils/emg_processing_utils.py` — filtering, envelope, repetition segmentation, RMS/MDF extraction
- `utils/data_utils.py` — C3D loading, dataset creation, model-bundle I/O, inference helper
- `pipeline/train_model.py` — cross-validation, threshold selection, and final model training
- `pipeline/inference.py` — minimal inference example on a single C3D file
- `pipeline/signal_analysis_pipeline.py` — non-ML “optimal rep” heuristic workflow

> For reproducible experiments, avoid editing the feature definitions mid-project; instead, version feature changes and re-generate `master_df.csv`.

---

## Out-of-the-Box Results (Bundled Model on Current Data)

This repository includes a ready-to-use model bundle at `models/fatigue_model_bundle.joblib`. The results below document the **current** performance of the bundled model/training configuration on the dataset produced from the repository’s current `data/Signals/*.c3d` files and `data/filtered_signals.csv` labels.

> **Reproducibility note:** These results are **dataset- and configuration-dependent**. If you add/remove sessions, change repetition segmentation parameters (`distance_seconds`, `prominence`), adjust filtering (50/60 Hz notch), or modify the feature set, your numbers will change. This section is included so new users know what to expect when using the model “out of the box.”

### Dataset snapshot

- **Total repetitions:** 396
- **Class balance (`is_fatigued`):**
  - 0 (not fatigued): 264
  - 1 (fatigued): 132
- **Total evaluated sessions (`file_id`):** 32

### Cross-validated (OOF) classification performance

Threshold selection was performed on out-of-fold (OOF) probabilities by maximizing **balanced accuracy**.

- **OOF best threshold:** `0.385`
- **OOF Balanced Accuracy:** `0.856`
- **OOF ROC AUC:** `0.933`
- **OOF PR AUC:** `0.868`

**OOF confusion matrix (repetition-level):**

|               | Pred: Not fatigued | Pred: Fatigued |
|---|---:|---:|
| **True: Not fatigued** | 216 | 48 |
| **True: Fatigued**     | 14  | 118 |

**Classification report (OOF):**

- Not fatigued (0): precision **0.939**, recall **0.818**, F1 **0.874** (support 264)
- Fatigued (1): precision **0.711**, recall **0.894**, F1 **0.792** (support 132)
- Overall accuracy: **0.843** (396 repetitions)

### Trigger timing performance (session-level)

Trigger evaluation measures how closely the first trigger aligns with labeled onset (when available). Errors are reported in **repetitions**.

- **Sessions with onset + a trigger:** 26
- **Never triggered (file-level false negatives):** 2
- **Mean Δ reps (trigger − onset):** 0.308 (median 0.0)
- **MAE (timing error):** 1.231 reps
- **% within ±0 reps:** 0.269
- **% within ±1 rep:** 0.692
- **% within ±2 reps:** 0.885
- **Early vs late triggers:** early_rate 0.308, late_rate 0.423

### Included inference demo (`test_data/test.c3d`)

Running the bundled model on the included test file triggers fatigue at:

- **Fatigue trigger repetition:** **9**

Predicted class stream (`pred` per repetition) from the demo run:
```text
0 0 0 0 0 0 0 1 1 1 1 1 1
```

### Figures (Current Model)

The figures below were generated from the current dataset/training run and are included in `docs/images/`.

#### Model evaluation (OOF)

![OOF predicted probabilities](docs/images/predicted_probabilities.png)
![Confusion matrix](docs/images/confusion_matrix.png)
![Precision-Recall curve](docs/images/precision_recall_curve.png)
![ROC curve](docs/images/roc_curve.png)
![Threshold sweep](docs/images/threshold_sweep.png)

#### Signal-analysis pipeline (example)

![Signal analysis trends](docs/images/signal_analysis_trends.png)

### Included inference demo (`test_data/test.c3d`)

Running the bundled model on the included test file triggers fatigue at:

- **Fatigue trigger repetition:** **9**

Predicted class stream (`pred` per repetition) from the demo run:
```text
0 0 0 0 0 0 0 1 1 1 1 1 1