"""
rule_learning.py
────────────────────────────────────────────────────────────────────────────────
CS6053 – Credit Risk Expert System  |  Member B
Knowledge-based learning using the RIPPER algorithm (wittgenstein library).

This module:
  1. Loads and splits the cleaned credit risk dataset
  2. Trains a RIPPER classifier on the training set
  3. Prints and saves the learned IF-THEN rules
  4. Evaluates the rules on the test set (accuracy, precision, recall, F1)
  5. Saves train/test splits for use by evaluation.py

Usage:
  python rule_learning.py
  # or import from other modules:
  from rule_learning import load_and_split, train_ripper, get_rules_as_text
────────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import pandas as pd
import numpy as np
import wittgenstein as lw
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score, confusion_matrix)

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH    = 'data/processed/cleaned_credit_risk_data.csv'   # adjust if needed
RESULTS_DIR  = 'results/metrics'
TARGET_COL   = 'loan_status'
POS_CLASS    = '1'          # '1' = default (high risk)
TEST_SIZE    = 0.20
RANDOM_STATE = 42

# ── 1. Data loading & splitting ───────────────────────────────────────────────

def load_and_split(data_path: str = DATA_PATH):
    """
    Load the cleaned CSV and produce stratified train/test splits.

    Returns
    -------
    X_train, X_test, y_train, y_test  (all pandas objects)
    """
    df = pd.read_csv(data_path)
    print(f"[load] Dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"[load] Class balance:\n{df[TARGET_COL].value_counts()}\n")

    # wittgenstein works on a single DataFrame, so we keep target inside
    # but also return X/y separately for sklearn compatibility
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(str)   # RIPPER expects string labels

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print(f"[load] Train: {len(X_train):,} rows  |  Test: {len(X_test):,} rows\n")
    return X_train, X_test, y_train, y_test


# ── 2. RIPPER training ────────────────────────────────────────────────────────

def train_ripper(X_train: pd.DataFrame,
                 y_train: pd.Series) -> lw.RIPPER:
    """
    Train a RIPPER classifier on the training data.

    RIPPER (Repeated Incremental Pruning to Produce Error Reduction) is a
    knowledge-based learning algorithm that produces human-readable IF-THEN
    rules directly from labelled data (Cohen, 1995).

    Parameters
    ----------
    X_train : feature DataFrame
    y_train : target Series (string labels '0' / '1')

    Returns
    -------
    Fitted lw.RIPPER instance
    """
    train_df = X_train.copy()
    train_df[TARGET_COL] = y_train.values

    print("[ripper] Training RIPPER classifier …")
    ripper = lw.RIPPER(random_state=RANDOM_STATE)
    ripper.fit(train_df, class_feat=TARGET_COL, pos_class=POS_CLASS)
    print(f"[ripper] Training complete. Rules learned: {len(ripper.ruleset_)}\n")
    return ripper


# ── 3. Rule extraction & display ─────────────────────────────────────────────

def get_rules_as_text(ripper: lw.RIPPER) -> list[str]:
    """
    Return each RIPPER rule as a human-readable string.

    Each rule has the form:
        IF <condition1> AND <condition2> … THEN loan_status = 1 (DEFAULT)
    """
    rules_text = []
    for i, rule in enumerate(ripper.ruleset_, start=1):
        conds = str(rule).replace('^', ' AND ')
        line  = f"Rule {i:02d}: IF {conds} THEN loan_status = 1 (DEFAULT)"
        rules_text.append(line)
    return rules_text


def print_rules(ripper: lw.RIPPER) -> None:
    """Pretty-print all learned RIPPER rules."""
    print("=" * 70)
    print("  LEARNED RIPPER RULES  (predicting loan default)")
    print("=" * 70)
    for line in get_rules_as_text(ripper):
        print(f"  {line}")
    print("=" * 70)
    print(f"  Default: IF none of the above THEN loan_status = 0 (NON-DEFAULT)")
    print("=" * 70 + "\n")


# ── 4. Evaluation ─────────────────────────────────────────────────────────────

def evaluate_ripper(ripper: lw.RIPPER,
                    X_test: pd.DataFrame,
                    y_test: pd.Series) -> dict:
    """
    Evaluate the RIPPER ruleset on the held-out test set.

    Returns a dict of metrics for use in evaluation.py.
    """
    y_pred = ripper.predict(X_test)

    # wittgenstein may return booleans or strings depending on version —
    # normalise both to strings '0' / '1' for sklearn compatibility
    y_test_arr = y_test.values.astype(str)
    y_pred_arr = np.array(['1' if p else '0' for p in y_pred])

    acc  = accuracy_score(y_test_arr, y_pred_arr)
    prec = precision_score(y_test_arr, y_pred_arr, pos_label=POS_CLASS,
                           zero_division=0)
    rec  = recall_score(y_test_arr, y_pred_arr, pos_label=POS_CLASS,
                        zero_division=0)
    f1   = f1_score(y_test_arr, y_pred_arr, pos_label=POS_CLASS,
                    zero_division=0)
    cm   = confusion_matrix(y_test_arr, y_pred_arr, labels=['0', '1'])

    metrics = {
        'model':     'RIPPER (wittgenstein)',
        'accuracy':  round(acc,  4),
        'precision': round(prec, 4),
        'recall':    round(rec,  4),
        'f1_score':  round(f1,   4),
        'confusion_matrix': cm.tolist(),
    }

    print("── RIPPER Evaluation (test set) ──────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted):")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  {cm[0,0]:>7}  {cm[0,1]:>7}")
    print(f"  Actual 1  {cm[1,0]:>7}  {cm[1,1]:>7}")
    print()

    return metrics


# ── 5. Save outputs ───────────────────────────────────────────────────────────

def save_outputs(ripper: lw.RIPPER,
                 metrics: dict,
                 results_dir: str = RESULTS_DIR) -> None:
    """
    Save learned rules (txt) and evaluation metrics (json) to disk.
    """
    os.makedirs(results_dir, exist_ok=True)

    # Save rules as plain text
    rules_path = os.path.join(results_dir, 'ripper_rules.txt')
    with open(rules_path, 'w') as f:
        f.write("RIPPER LEARNED RULES — CS6053 Credit Risk Expert System\n")
        f.write("=" * 70 + "\n")
        for line in get_rules_as_text(ripper):
            f.write(line + "\n")
        f.write("=" * 70 + "\n")
        f.write("Default: IF none of the above THEN loan_status = 0 (NON-DEFAULT)\n")
    print(f"[save] Rules saved to {rules_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(results_dir, 'ripper_metrics.json')
    metrics_to_save = {k: v for k, v in metrics.items()
                       if k != 'confusion_matrix'}
    with open(metrics_path, 'w') as f:
        json.dump(metrics_to_save, f, indent=2)
    print(f"[save] Metrics saved to {metrics_path}\n")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_pipeline(data_path: str = DATA_PATH) -> tuple:
    """
    Full rule-learning pipeline:
      load → split → train RIPPER → print rules → evaluate → save

    Returns
    -------
    ripper    : fitted RIPPER model
    X_train, X_test, y_train, y_test : splits for use in evaluation.py
    metrics   : dict of evaluation results
    """
    X_train, X_test, y_train, y_test = load_and_split(data_path)
    ripper  = train_ripper(X_train, y_train)
    print_rules(ripper)
    metrics = evaluate_ripper(ripper, X_test, y_test)
    save_outputs(ripper, metrics)
    return ripper, X_train, X_test, y_train, y_test, metrics


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys

    # Allow passing a custom data path as CLI argument
    # e.g.  python rule_learning.py data/processed/cleaned_credit_risk_data.csv
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH

    ripper, X_train, X_test, y_train, y_test, metrics = run_pipeline(path)

    print("\n[done] Rule learning complete.")
    print(f"       {len(ripper.ruleset_)} rules learned.")
    print(f"       F1 Score on test set: {metrics['f1_score']}")