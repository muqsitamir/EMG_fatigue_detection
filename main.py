import pandas as pd
from pipeline.signal_analysis_pipeline import signal_analysis_pipeline
from pipeline.train_model import run_training_eval
from utils.data_utils import load_with_csv, create_master_df
from utils.eval_utils import evaluate_onset_timing
from utils.ml_utils import TrainConfig
from utils.print_utils import print_results, print_timing_summary


def main():
    # raw_data = plot_emg_signals()
    # data = load_with_csv()
    # signal_analysis_pipeline(data)
    # df = create_master_df(data)

    df = pd.read_csv("./data/master_df.csv")

    cfg = TrainConfig(n_splits=5)

    results, best_t = run_training_eval(df, cfg, plot=False)

    print_results(results)

    timing_df, timing_summary = evaluate_onset_timing(df, results["oof_proba"], thr=best_t, M=2)

    print_timing_summary(timing_summary)
    print(timing_df)


if __name__ == "__main__":
    main()
