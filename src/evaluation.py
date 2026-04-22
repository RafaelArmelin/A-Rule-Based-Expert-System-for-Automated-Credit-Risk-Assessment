"""
evaluation.py
────────────────────────────────────────────────────────────────────────────────
CS6053 – Credit Risk Expert System  |  Member B

Evaluates and compares three approaches on the same held-out test set:

  1. RIPPER rules (wittgenstein)       — data-learned rules
  2. Expert System (experta, 24 rules) — hand-crafted regulatory rules
  3. Decision Tree (sklearn)           — baseline ML model

Outputs
-------
  results/figures/confusion_matrices.png  — side-by-side confusion matrices
  results/figures/roc_curves.png          — overlaid ROC curves
  results/figures/f1_comparison.png       — F1 bar chart
  results/metrics/evaluation_summary.json — all metrics in one file
────────────────────────────────────────────────────────────────────────────────
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
    RocCurveDisplay,
)
from sklearn.preprocessing import LabelEncoder

# Local modules
from rule_learning import load_and_split, train_ripper
from expert_system import assess_applicant

# ── Config ────────────────────────────────────────────────────────────────────

DATA_PATH   = 'data/processed/cleaned_credit_risk_data.csv'
FIGURES_DIR = 'results/figures'
METRICS_DIR = 'results/metrics'

# Map expert system decisions to binary labels for evaluation
# APPROVE → 0 (non-default / low risk)
# REFER   → 1 (treat as high risk for conservative evaluation)
# REJECT  → 1 (high risk)
DECISION_MAP = {'APPROVE': 0, 'REFER': 1, 'REJECT': 1}

os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(METRICS_DIR, exist_ok=True)


# ── Helper: compute all metrics ───────────────────────────────────────────────

def compute_metrics(name: str,
                    y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray = None) -> dict:
    """
    Compute accuracy, precision, recall, F1 and optionally ROC-AUC.

    Parameters
    ----------
    name   : model label for display
    y_true : ground-truth binary labels (int 0/1)
    y_pred : predicted binary labels (int 0/1)
    y_prob : predicted probabilities for class 1 (optional, for ROC-AUC)
    """
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    cm   = confusion_matrix(y_true, y_pred)

    roc_auc = None
    if y_prob is not None:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

    metrics = {
        'model':      name,
        'accuracy':   round(float(acc),  4),
        'precision':  round(float(prec), 4),
        'recall':     round(float(rec),  4),
        'f1_score':   round(float(f1),   4),
        'roc_auc':    round(float(roc_auc), 4) if roc_auc else 'N/A',
        'confusion_matrix': cm.tolist(),
    }

    print(f"\n── {name} ──────────────────────────────────────────────────────")
    print(f"  Accuracy  : {acc:.4f}")
    print(f"  Precision : {prec:.4f}")
    print(f"  Recall    : {rec:.4f}")
    print(f"  F1 Score  : {f1:.4f}")
    if roc_auc:
        print(f"  ROC-AUC   : {roc_auc:.4f}")
    print(f"  Confusion Matrix:")
    print(f"              Pred 0   Pred 1")
    print(f"  Actual 0  {cm[0,0]:>7}  {cm[0,1]:>7}")
    print(f"  Actual 1  {cm[1,0]:>7}  {cm[1,1]:>7}")

    return metrics


# ── Model 1: RIPPER ───────────────────────────────────────────────────────────

def evaluate_ripper(ripper, X_test: pd.DataFrame,
                    y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    """Run RIPPER predictions and return metrics + binary predictions."""
    raw_pred = ripper.predict(X_test)
    y_pred   = np.array([1 if p else 0 for p in raw_pred])

    # RIPPER doesn't natively output probabilities — use predict_proba if
    # available, otherwise fall back to hard labels for ROC
    try:
        proba  = ripper.predict_proba(X_test)
        y_prob = proba[:, 1]
    except AttributeError:
        y_prob = y_pred.astype(float)   # fallback: binary as probability

    metrics = compute_metrics('RIPPER (wittgenstein)', y_test, y_pred, y_prob)
    return metrics, y_pred, y_prob


# ── Model 2: Expert System (your 24 hand-crafted rules) ──────────────────────

def run_expert_system_on_test(X_test: pd.DataFrame) -> np.ndarray:
    """
    Apply the experta expert system (24 hand-crafted rules) to every row
    in X_test and return binary predictions (0 = low risk, 1 = high risk).

    The expert system uses your regulatory rules:
      APPROVE → 0  |  REFER → 1  |  REJECT → 1
    """
    predictions = []
    for _, row in X_test.iterrows():
        result = assess_applicant(
            person_age               = row.get('person_age'),
            person_income            = row.get('person_income'),
            loan_amnt                = row.get('loan_amnt'),
            loan_percent_income      = row.get('loan_percent_income'),
            loan_grade               = row.get('loan_grade'),
            loan_int_rate            = row.get('loan_int_rate'),
            cb_person_cred_hist_length = row.get('cb_person_cred_hist_length'),
            person_emp_length        = row.get('person_emp_length'),
            person_home_ownership    = row.get('person_home_ownership'),
            loan_intent              = row.get('loan_intent'),
            # dataset encodes default as 0/1 int; convert to 'Y'/'N' for engine
            cb_person_default_on_file = 'Y' if row.get('cb_person_default_on_file') == 1 else 'N',
            aml_concern              = False,
        )
        predictions.append(DECISION_MAP.get(result['decision'], 0))

    return np.array(predictions)


def evaluate_expert_system(X_test: pd.DataFrame,
                            y_test: np.ndarray) -> tuple[dict, np.ndarray]:
    """Evaluate the expert system on the test set."""
    print("\n[expert system] Running 24-rule engine on test set "
          f"({len(X_test):,} rows) … this may take a moment …")
    y_pred  = run_expert_system_on_test(X_test)
    y_prob  = y_pred.astype(float)   # binary scores (no soft probabilities)
    metrics = compute_metrics('Expert System (24 rules, experta)', y_test, y_pred, None)
    return metrics, y_pred, y_prob


# ── Model 3: Decision Tree baseline ──────────────────────────────────────────

def evaluate_decision_tree(X_train: pd.DataFrame, y_train: np.ndarray,
                            X_test:  pd.DataFrame, y_test:  np.ndarray,
                            ) -> tuple[dict, np.ndarray, np.ndarray, DecisionTreeClassifier]:
    """
    Train and evaluate a Decision Tree classifier as the baseline model.
    Categorical columns are label-encoded before fitting.
    """
    # Encode categorical columns
    cat_cols = X_train.select_dtypes(include='object').columns.tolist()
    encoders = {}

    X_train_enc = X_train.copy()
    X_test_enc  = X_test.copy()

    for col in cat_cols:
        le = LabelEncoder()
        X_train_enc[col] = le.fit_transform(X_train[col].astype(str))
        # handle unseen labels in test set gracefully
        X_test_enc[col]  = X_test[col].astype(str).map(
            lambda x, le=le: le.transform([x])[0]
            if x in le.classes_ else -1
        )
        encoders[col] = le

    dt = DecisionTreeClassifier(max_depth=5, random_state=42)
    dt.fit(X_train_enc, y_train)

    y_pred = dt.predict(X_test_enc)
    y_prob = dt.predict_proba(X_test_enc)[:, 1]

    metrics = compute_metrics('Decision Tree (sklearn baseline)',
                              y_test, y_pred, y_prob)
    return metrics, y_pred, y_prob, dt


# ── Figures ───────────────────────────────────────────────────────────────────

def plot_confusion_matrices(results: list[dict]) -> None:
    """Plot side-by-side confusion matrices for all three models."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle('Confusion Matrices — Credit Risk Expert System',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax, res in zip(axes, results):
        cm = np.array(res['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=['Pred: Low', 'Pred: High'],
                    yticklabels=['Actual: Low', 'Actual: High'])
        ax.set_title(res['model'], fontsize=10, fontweight='bold')
        ax.set_ylabel('Actual')
        ax.set_xlabel('Predicted')

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n[figure] Confusion matrices saved → {path}")


def plot_roc_curves(y_test: np.ndarray,
                    probs: dict) -> None:
    """Plot overlaid ROC curves for models that support probabilities."""
    fig, ax = plt.subplots(figsize=(7, 6))

    colors = {'RIPPER (wittgenstein)': '#2196F3',
              'Decision Tree (sklearn baseline)': '#4CAF50'}

    for name, y_prob in probs.items():
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc     = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f'{name}  (AUC = {roc_auc:.3f})',
                color=colors.get(name, 'grey'), lw=2)

    ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random classifier')
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('ROC Curves — Credit Risk Models', fontsize=13, fontweight='bold')
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(alpha=0.3)

    path = os.path.join(FIGURES_DIR, 'roc_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[figure] ROC curves saved → {path}")


def plot_f1_comparison(results: list[dict]) -> None:
    """Bar chart comparing F1 scores across all three models."""
    names  = [r['model'].split('(')[0].strip() for r in results]
    f1s    = [r['f1_score'] for r in results]
    colors = ['#2196F3', '#FF9800', '#4CAF50']

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(names, f1s, color=colors, edgecolor='white', width=0.5)

    for bar, val in zip(bars, f1s):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=11,
                fontweight='bold')

    ax.set_ylim(0, 1.05)
    ax.set_ylabel('F1 Score', fontsize=12)
    ax.set_title('F1 Score Comparison — Credit Risk Models',
                 fontsize=13, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    path = os.path.join(FIGURES_DIR, 'f1_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[figure] F1 comparison saved → {path}")


def plot_metrics_table(results: list[dict]) -> None:
    """Save a clean metrics summary table as a figure (for slides/report)."""
    rows = []
    for r in results:
        rows.append({
            'Model':     r['model'],
            'Accuracy':  f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall':    f"{r['recall']:.4f}",
            'F1 Score':  f"{r['f1_score']:.4f}",
            'ROC-AUC':   f"{r['roc_auc']:.4f}" if r['roc_auc'] != 'N/A' else 'N/A',
        })

    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(12, 2))
    ax.axis('off')
    tbl = ax.table(cellText=df.values, colLabels=df.columns,
                   cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.2, 1.8)

    # Style header row
    for j in range(len(df.columns)):
        tbl[0, j].set_facecolor('#1565C0')
        tbl[0, j].set_text_props(color='white', fontweight='bold')

    plt.title('Model Evaluation Summary — CS6053 Credit Risk Expert System',
              fontsize=12, fontweight='bold', pad=20)

    path = os.path.join(FIGURES_DIR, 'metrics_table.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[figure] Metrics table saved → {path}")


# ── Save all metrics to JSON ──────────────────────────────────────────────────

def save_summary(results: list[dict]) -> None:
    """Save all metrics to a single JSON file."""
    summary = []
    for r in results:
        summary.append({k: v for k, v in r.items()
                        if k != 'confusion_matrix'})

    path = os.path.join(METRICS_DIR, 'evaluation_summary.json')
    with open(path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"[save] Evaluation summary saved → {path}")


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run_evaluation(data_path: str = DATA_PATH) -> None:
    """
    Full evaluation pipeline:
      1. Load & split data
      2. Train RIPPER
      3. Evaluate RIPPER, Expert System, Decision Tree
      4. Generate all figures
      5. Save metrics
    """
    print("=" * 70)
    print("  CS6053 — Credit Risk Expert System  |  Model Evaluation")
    print("=" * 70)

    # ── Load data & train RIPPER ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = load_and_split(data_path)

    # Convert labels to int for sklearn compatibility
    y_train_int = y_train.astype(int).values
    y_test_int  = y_test.astype(int).values

    ripper = train_ripper(X_train, y_train)

    # ── Evaluate all three models ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  EVALUATION RESULTS")
    print("=" * 70)

    m_ripper, _, y_prob_ripper = evaluate_ripper(ripper, X_test, y_test_int)
    m_expert, _, _             = evaluate_expert_system(X_test, y_test_int)
    m_dt, _, y_prob_dt, _      = evaluate_decision_tree(
                                    X_train, y_train_int,
                                    X_test,  y_test_int)

    results = [m_ripper, m_expert, m_dt]

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  SUMMARY TABLE")
    print("=" * 70)
    print(f"{'Model':<40} {'Acc':>6} {'Prec':>6} {'Rec':>6} {'F1':>6} {'AUC':>6}")
    print("-" * 70)
    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r['roc_auc'] != 'N/A' else '  N/A'
        print(f"{r['model']:<40} {r['accuracy']:>6.4f} {r['precision']:>6.4f} "
              f"{r['recall']:>6.4f} {r['f1_score']:>6.4f} {auc_str:>6}")
    print("=" * 70)

    # ── Generate figures ──────────────────────────────────────────────────
    print("\n[figures] Generating plots …")
    plot_confusion_matrices(results)
    plot_roc_curves(y_test_int,
                    probs={'RIPPER (wittgenstein)':        y_prob_ripper,
                           'Decision Tree (sklearn baseline)': y_prob_dt})
    plot_f1_comparison(results)
    plot_metrics_table(results)

    # ── Save metrics ──────────────────────────────────────────────────────
    save_summary(results)

    print("\n[done] Evaluation complete. All figures saved to results/figures/")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else DATA_PATH
    run_evaluation(path)