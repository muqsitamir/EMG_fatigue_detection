from typing import Dict


def print_results(results: Dict):
    print("OOF best threshold:", round(results["threshold"], 3))
    print("OOF Balanced Acc:", round(results["balanced_accuracy"], 3))
    print("OOF ROC AUC:", round(results["roc_auc"], 3))
    print("OOF PR AUC:", round(results["pr_auc"], 3))
    print("Confusion matrix:\n", results["confusion_matrix"])
    print(results["classification_report"])

def print_timing_summary(metrics: Dict):
    print(f"""
    Fatigue Trigger Evaluation Summary
    ----------------------------------
    
    • files_total = {metrics['files_total']}
      Total number of evaluated sessions (file_ids).
    
    • files_with_onset_and_trigger = {metrics['files_with_onset_and_trigger']}
      Sessions where a labeled fatigue onset exists and the trigger fired at least once.
    
    • never_triggered = {metrics['never_triggered']}
      Sessions where the trigger never fired (file-level false negatives).
    
    • mean_delta_reps = {metrics['mean_delta_reps']:.3f}, median_delta_reps = {metrics['median_delta_reps']}
      Average and median rep-lag of the trigger relative to onset.
    
    • mae_reps = {metrics['mae_reps']:.3f}
      Mean absolute timing error (in reps).
    
    • pct_within_0_reps = {metrics['pct_within_0_reps']:.3f}
      % of triggers that fired exactly on the onset rep.
    
    • pct_within_1_rep = {metrics['pct_within_1_rep']:.3f}
      % of triggers that fired within ±1 rep.
    
    • pct_within_2_reps = {metrics['pct_within_2_reps']:.3f}
      % of triggers that fired within ±2 reps.
    
    • early_rate = {metrics['early_rate']:.3f}, late_rate = {metrics['late_rate']:.3f}
      Rates of early vs. late triggers.
    """)
