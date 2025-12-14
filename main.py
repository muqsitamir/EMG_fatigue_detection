import pandas as pd

from emg_fd.src.pipeline.inference import inference_for_single_test_file
from emg_fd.src.pipeline.train_model import run_training_eval, train_final_model
from emg_fd.src.utils.eval_utils import evaluate_onset_timing
from emg_fd.src.utils.ml_utils import TrainConfig
from emg_fd.src.utils.print_utils import print_results, print_timing_summary


def main():
    # raw_data = plot_emg_signals()
    # data = load_with_csv()
    # signal_analysis_pipeline(data)
    # df = create_master_df(data)

    df = pd.read_csv("./data/master_df.csv")

    cfg = TrainConfig(n_splits=5)

    results, best_t = run_training_eval(df, cfg, plot=False)

    print_results(results)

    m, n = 2, 3

    timing_df, timing_summary = evaluate_onset_timing(df, results["oof_proba"], thr=best_t, M=m, N=n, m_of_n=True)

    print_timing_summary(timing_summary)
    print(timing_df)

    train_final_model(df, best_t, m, n)

    df_pred, trigger_rep = inference_for_single_test_file("./test_data/test.c3d", "Emg_1", "./models/fatigue_model_bundle.joblib")

    print(f"Fatigue detected at {trigger_rep}")
    print(df_pred.pred)

if __name__ == "__main__":
    main()
