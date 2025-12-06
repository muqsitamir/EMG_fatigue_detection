from utils.data_utils import load_with_csv
from signal_analysis_pipeline import signal_analysis_pipeline


def main():
    # raw_data = plot_emg_signals()
    data = load_with_csv()
    signal_analysis_pipeline(data)

if __name__ == "__main__":
    main()
