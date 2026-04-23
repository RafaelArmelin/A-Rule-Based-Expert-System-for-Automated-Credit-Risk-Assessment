"""
app.py
────────────────────────────────────────────────────────────────────────────────
CS6053 – Credit Risk Expert System  |  Streamlit Dashboard

A professional web interface for the credit risk expert system.
Provides three views:
  1. Individual Assessment — input applicant details, get a decision
     with full rule-based explanation
  2. Model Comparison — evaluate RIPPER, Expert System, and Decision Tree
     on the same held-out test set with metrics and visualisations
  3. Dataset Overview — summary statistics and distributions

Usage:
  streamlit run app.py
────────────────────────────────────────────────────────────────────────────────
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_curve, auc,
)
from sklearn.preprocessing import LabelEncoder

# Add src/ to path so we can import project modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from expert_system import assess_applicant

# ── Page configuration ────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Credit Risk Expert System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom styling ────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Clean, professional typography — theme-aware */
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    .subtitle {
        font-size: 0.95rem;
        opacity: 0.7;
        margin-bottom: 2rem;
    }

    /* Sidebar navigation — ensure labels are always visible */
    [data-testid="stSidebar"] .stRadio > label {
        font-size: 0.95rem !important;
        font-weight: 600 !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label {
        padding: 0.5rem 0.75rem !important;
        border-radius: 6px !important;
        margin-bottom: 0.25rem !important;
    }
    [data-testid="stSidebar"] .stRadio [role="radiogroup"] label:hover {
        background-color: rgba(128, 128, 128, 0.15);
    }
    [data-testid="stSidebar"] h3 {
        font-size: 0.85rem !important;
        text-transform: uppercase !important;
        letter-spacing: 0.08em !important;
        opacity: 0.6;
        margin-bottom: 0.75rem !important;
    }

    /* Decision cards — works in both light and dark */
    .decision-card {
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
        border-left: 5px solid;
    }
    .decision-approve {
        background-color: rgba(22, 163, 74, 0.12);
        border-color: #16a34a;
    }
    .decision-reject {
        background-color: rgba(220, 38, 38, 0.12);
        border-color: #dc2626;
    }
    .decision-refer {
        background-color: rgba(217, 119, 6, 0.12);
        border-color: #d97706;
    }
    .decision-label {
        font-size: 1.4rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }

    /* Rule list — theme-aware */
    .rule-item {
        padding: 0.6rem 1rem;
        background: rgba(128, 128, 128, 0.08);
        border-left: 3px solid #94a3b8;
        margin: 0.4rem 0;
        font-size: 0.9rem;
        border-radius: 0 4px 4px 0;
    }

    /* Remove Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

DATA_PATH = os.path.join('data', 'processed', 'cleaned_credit_risk_data.csv')
TARGET_COL = 'loan_status'


@st.cache_data
def load_data():
    """Load and cache the processed dataset."""
    if not os.path.exists(DATA_PATH):
        st.error(f"Dataset not found at {DATA_PATH}. "
                 "Run preprocessing first.")
        st.stop()
    return pd.read_csv(DATA_PATH)


@st.cache_data
def get_train_test_split(test_size=0.20, random_state=42):
    """Cached train/test split matching rule_learning.py parameters."""
    df = load_data()
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL].astype(int)
    return train_test_split(X, y, test_size=test_size,
                            random_state=random_state, stratify=y)


# ── Sidebar navigation ───────────────────────────────────────────────────────

st.sidebar.markdown("### Navigation")
page = st.sidebar.radio(
    "Select a view:",
    ["Individual Assessment", "Model Comparison", "Dataset Overview"],
    label_visibility="collapsed",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "<div style='font-size: 0.75rem; color: #9ca3af;'>"
    "CS6053 AI and Machine Learning<br>"
    "London Metropolitan University<br>"
    "Spring 2026"
    "</div>",
    unsafe_allow_html=True,
)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Individual Assessment
# ══════════════════════════════════════════════════════════════════════════════

if page == "Individual Assessment":

    st.markdown('<div class="main-title">Credit Risk Assessment</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">'
                'Enter applicant details to receive a risk decision '
                'with full rule-based explanation.'
                '</div>', unsafe_allow_html=True)

    # ── Input form ────────────────────────────────────────────────────────

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Personal Details**")
        person_age = st.number_input("Age", min_value=0, max_value=120,
                                     value=30, step=1)
        person_income = st.number_input("Annual Income", min_value=0,
                                        value=45000, step=1000)
        person_emp_length = st.number_input("Employment Length (years)",
                                            min_value=0, max_value=80,
                                            value=5, step=1)
        person_home_ownership = st.selectbox(
            "Home Ownership",
            ["RENT", "OWN", "MORTGAGE", "OTHER"],
        )

    with col2:
        st.markdown("**Loan Details**")
        loan_amnt = st.number_input("Loan Amount", min_value=0,
                                     value=10000, step=500)
        loan_intent = st.selectbox(
            "Loan Purpose",
            ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE",
             "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"],
        )
        loan_grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
        loan_int_rate = st.number_input("Interest Rate (%)",
                                         min_value=0.0, max_value=35.0,
                                         value=10.5, step=0.5)

    with col3:
        st.markdown("**Credit History**")
        cb_person_cred_hist_length = st.number_input(
            "Credit History Length (years)",
            min_value=0, max_value=50, value=4, step=1,
        )
        cb_person_default_on_file = st.selectbox(
            "Previous Default on File",
            ["N", "Y"],
            format_func=lambda x: "Yes" if x == "Y" else "No",
        )

        # Calculated field
        if person_income > 0:
            loan_percent_income = round(loan_amnt / person_income, 4)
        else:
            loan_percent_income = 0.0
        st.markdown(f"**Loan-to-Income Ratio:** {loan_percent_income:.2%}")

    st.markdown("---")

    # ── Run assessment ────────────────────────────────────────────────────

    if st.button("Run Assessment", type="primary", use_container_width=True):
        result = assess_applicant(
            person_age=person_age,
            person_income=person_income,
            loan_amnt=loan_amnt,
            loan_percent_income=loan_percent_income,
            loan_grade=loan_grade,
            loan_int_rate=loan_int_rate,
            cb_person_cred_hist_length=cb_person_cred_hist_length,
            person_emp_length=person_emp_length,
            person_home_ownership=person_home_ownership,
            loan_intent=loan_intent,
            cb_person_default_on_file=cb_person_default_on_file,
            aml_concern=False,
        )

        decision = result['decision']
        reasons = result['reasons']

        # Decision card
        style_map = {
            'APPROVE': ('decision-approve', 'APPROVED'),
            'REJECT':  ('decision-reject', 'REJECTED'),
            'REFER':   ('decision-refer', 'REFERRED FOR MANUAL REVIEW'),
        }
        css_class, label = style_map.get(decision, ('', decision))

        st.markdown(
            f'<div class="decision-card {css_class}">'
            f'<div class="decision-label">{label}</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Rules that fired
        st.markdown("**Rules triggered:**")
        for reason in reasons:
            st.markdown(
                f'<div class="rule-item">{reason}</div>',
                unsafe_allow_html=True,
            )

        # Applicant summary
        with st.expander("Applicant Summary"):
            summary_data = {
                "Field": [
                    "Age", "Annual Income", "Employment Length",
                    "Home Ownership", "Loan Amount", "Loan Purpose",
                    "Loan Grade", "Interest Rate", "Loan-to-Income Ratio",
                    "Credit History Length", "Previous Default",
                ],
                "Value": [
                    f"{person_age}", f"{person_income:,}",
                    f"{person_emp_length} years", person_home_ownership,
                    f"{loan_amnt:,}", loan_intent, loan_grade,
                    f"{loan_int_rate}%", f"{loan_percent_income:.2%}",
                    f"{cb_person_cred_hist_length} years",
                    "Yes" if cb_person_default_on_file == "Y" else "No",
                ],
            }
            st.table(pd.DataFrame(summary_data))


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Model Comparison
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Model Comparison":

    st.markdown('<div class="main-title">Model Comparison</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">'
                'Performance evaluation of RIPPER, Expert System, and '
                'Decision Tree on the same held-out test set.'
                '</div>', unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = get_train_test_split()

    # ── Decision Tree ─────────────────────────────────────────────────────

    @st.cache_resource
    def train_decision_tree():
        cat_cols = X_train.select_dtypes(include='object').columns.tolist()
        X_tr_enc = X_train.copy()
        X_te_enc = X_test.copy()
        encoders = {}
        for col in cat_cols:
            le = LabelEncoder()
            X_tr_enc[col] = le.fit_transform(X_train[col].astype(str))
            X_te_enc[col] = X_test[col].astype(str).map(
                lambda x, le=le: le.transform([x])[0]
                if x in le.classes_ else -1
            )
            encoders[col] = le
        dt = DecisionTreeClassifier(max_depth=5, random_state=42)
        dt.fit(X_tr_enc, y_train)
        y_pred = dt.predict(X_te_enc)
        y_prob = dt.predict_proba(X_te_enc)[:, 1]
        return dt, y_pred, y_prob

    # ── RIPPER ────────────────────────────────────────────────────────────

    @st.cache_resource
    def train_ripper_model():
        import wittgenstein as lw
        train_df = X_train.copy()
        train_df[TARGET_COL] = y_train.values
        ripper = lw.RIPPER(random_state=42)
        ripper.fit(train_df, class_feat=TARGET_COL, pos_class='1')
        raw_pred = ripper.predict(X_test)
        y_pred = np.array([1 if p else 0 for p in raw_pred])
        y_prob = y_pred.astype(float)
        return ripper, y_pred, y_prob

    # ── Expert System ─────────────────────────────────────────────────────

    @st.cache_data
    def run_expert_system():
        decision_map = {'APPROVE': 0, 'REFER': 1, 'REJECT': 1}
        predictions = []
        for _, row in X_test.iterrows():
            result = assess_applicant(
                person_age=row.get('person_age'),
                person_income=row.get('person_income'),
                loan_amnt=row.get('loan_amnt'),
                loan_percent_income=row.get('loan_percent_income'),
                loan_grade=row.get('loan_grade'),
                loan_int_rate=row.get('loan_int_rate'),
                cb_person_cred_hist_length=row.get('cb_person_cred_hist_length'),
                person_emp_length=row.get('person_emp_length'),
                person_home_ownership=row.get('person_home_ownership'),
                loan_intent=row.get('loan_intent'),
                cb_person_default_on_file=(
                    'Y' if row.get('cb_person_default_on_file') == 1 else 'N'
                ),
                aml_concern=False,
            )
            predictions.append(decision_map.get(result['decision'], 0))
        return np.array(predictions)

    # ── Run all models ────────────────────────────────────────────────────

    with st.spinner("Training models and running evaluation..."):
        dt_model, dt_pred, dt_prob = train_decision_tree()
        ripper_model, ripper_pred, ripper_prob = train_ripper_model()
        es_pred = run_expert_system()

    y_test_arr = y_test.values

    # ── Compute metrics ───────────────────────────────────────────────────

    def get_metrics(name, y_true, y_pred, y_prob=None):
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)
        roc_auc_val = None
        if y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc_val = auc(fpr, tpr)
        return {
            'model': name, 'accuracy': acc, 'precision': prec,
            'recall': rec, 'f1_score': f1, 'roc_auc': roc_auc_val,
            'confusion_matrix': cm,
        }

    results = [
        get_metrics('RIPPER', y_test_arr, ripper_pred, ripper_prob),
        get_metrics('Expert System (24 Rules)', y_test_arr, es_pred, None),
        get_metrics('Decision Tree', y_test_arr, dt_pred, dt_prob),
    ]

    # ── Metrics summary ──────────────────────────────────────────────────

    st.markdown("### Performance Metrics")

    metrics_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Accuracy': f"{r['accuracy']:.4f}",
            'Precision': f"{r['precision']:.4f}",
            'Recall': f"{r['recall']:.4f}",
            'F1 Score': f"{r['f1_score']:.4f}",
            'ROC-AUC': f"{r['roc_auc']:.4f}" if r['roc_auc'] else 'N/A',
        }
        for r in results
    ])
    st.dataframe(metrics_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Visualisations ────────────────────────────────────────────────────

    col_left, col_right = st.columns(2)

    # F1 comparison bar chart
    with col_left:
        st.markdown("### F1 Score Comparison")
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        names = [r['model'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        colours = ['#2563eb', '#d97706', '#16a34a']
        bars = ax1.bar(names, f1_scores, color=colours, width=0.5,
                       edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, f1_scores):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.01,
                     f'{val:.4f}', ha='center', va='bottom',
                     fontsize=10, fontweight='bold', color='#1a1a2e')
        ax1.set_ylim(0, 1.05)
        ax1.set_ylabel('F1 Score', fontsize=11)
        ax1.tick_params(axis='x', labelsize=9)
        ax1.grid(axis='y', alpha=0.2)
        ax1.spines['top'].set_visible(False)
        ax1.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig1)
        plt.close()

    # ROC curves
    with col_right:
        st.markdown("### ROC Curves")
        fig2, ax2 = plt.subplots(figsize=(6, 4))
        roc_models = {
            'RIPPER': ripper_prob,
            'Decision Tree': dt_prob,
        }
        roc_colours = {'RIPPER': '#2563eb', 'Decision Tree': '#16a34a'}
        for name, y_prob in roc_models.items():
            fpr, tpr, _ = roc_curve(y_test_arr, y_prob)
            roc_auc_val = auc(fpr, tpr)
            ax2.plot(fpr, tpr, color=roc_colours[name], lw=2,
                     label=f'{name} (AUC = {roc_auc_val:.3f})')
        ax2.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.4,
                 label='Random classifier')
        ax2.set_xlabel('False Positive Rate', fontsize=11)
        ax2.set_ylabel('True Positive Rate', fontsize=11)
        ax2.legend(loc='lower right', fontsize=9, framealpha=0.9)
        ax2.grid(alpha=0.2)
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig2)
        plt.close()

    # Confusion matrices
    st.markdown("### Confusion Matrices")
    cm_cols = st.columns(3)

    for i, (col, r) in enumerate(zip(cm_cols, results)):
        with col:
            fig_cm, ax_cm = plt.subplots(figsize=(4, 3.5))
            cm = r['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Pred: Low', 'Pred: High'],
                        yticklabels=['Actual: Low', 'Actual: High'],
                        cbar=False, linewidths=0.5, linecolor='white')
            ax_cm.set_title(r['model'], fontsize=10, fontweight='bold',
                            color='#1a1a2e', pad=10)
            ax_cm.set_ylabel('Actual', fontsize=9)
            ax_cm.set_xlabel('Predicted', fontsize=9)
            ax_cm.tick_params(labelsize=8)
            plt.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

    # ── RIPPER learned rules ──────────────────────────────────────────────

    st.markdown("---")
    st.markdown("### RIPPER Learned Rules")
    st.markdown("Rules extracted automatically from the training data "
                "using the RIPPER algorithm (Cohen, 1995).")

    for i, rule in enumerate(ripper_model.ruleset_, start=1):
        conds = str(rule).replace('^', ' AND ')
        st.markdown(
            f'<div class="rule-item">'
            f'<strong>Rule {i:02d}:</strong> IF {conds} '
            f'THEN loan_status = 1 (DEFAULT)</div>',
            unsafe_allow_html=True,
        )
    st.markdown(
        '<div class="rule-item" style="border-color: #16a34a;">'
        '<strong>Default:</strong> IF none of the above '
        'THEN loan_status = 0 (NON-DEFAULT)</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3: Dataset Overview
# ══════════════════════════════════════════════════════════════════════════════

elif page == "Dataset Overview":

    st.markdown('<div class="main-title">Dataset Overview</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="subtitle">'
                'Summary statistics and distributions for the '
                'Kaggle Credit Risk Dataset.'
                '</div>', unsafe_allow_html=True)

    df = load_data()

    # Key stats
    total = len(df)
    defaults = df[TARGET_COL].sum()
    non_defaults = total - defaults

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.metric("Total Records", f"{total:,}")
    with col_b:
        st.metric("Non-Default (Class 0)", f"{non_defaults:,}")
    with col_c:
        st.metric("Default (Class 1)", f"{defaults:,}")

    st.markdown("---")

    # Class distribution
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### Class Distribution")
        fig_cls, ax_cls = plt.subplots(figsize=(5, 3.5))
        counts = df[TARGET_COL].value_counts().sort_index()
        bars = ax_cls.bar(['Non-Default', 'Default'], counts.values,
                          color=['#2563eb', '#dc2626'], width=0.4,
                          edgecolor='white', linewidth=1.5)
        for bar, val in zip(bars, counts.values):
            ax_cls.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + total * 0.01,
                        f'{val:,}', ha='center', va='bottom',
                        fontsize=10, fontweight='bold')
        ax_cls.set_ylabel('Count', fontsize=11)
        ax_cls.grid(axis='y', alpha=0.2)
        ax_cls.spines['top'].set_visible(False)
        ax_cls.spines['right'].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig_cls)
        plt.close()

    with col_right:
        st.markdown("### Feature Summary")
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if TARGET_COL in num_cols:
            num_cols.remove(TARGET_COL)
        st.dataframe(
            df[num_cols].describe().round(2).T,
            use_container_width=True,
        )

    # Feature distributions
    st.markdown("---")
    st.markdown("### Feature Distributions")

    num_features = df.select_dtypes(include=[np.number]).columns.tolist()
    if TARGET_COL in num_features:
        num_features.remove(TARGET_COL)

    selected_feature = st.selectbox("Select feature:", num_features)

    fig_dist, ax_dist = plt.subplots(figsize=(8, 3.5))
    for label, colour in [(0, '#2563eb'), (1, '#dc2626')]:
        subset = df[df[TARGET_COL] == label][selected_feature].dropna()
        ax_dist.hist(subset, bins=40, alpha=0.6, color=colour,
                     label=f'Class {label}', edgecolor='white', linewidth=0.5)
    ax_dist.set_xlabel(selected_feature, fontsize=11)
    ax_dist.set_ylabel('Count', fontsize=11)
    ax_dist.legend(fontsize=9, framealpha=0.9)
    ax_dist.grid(axis='y', alpha=0.2)
    ax_dist.spines['top'].set_visible(False)
    ax_dist.spines['right'].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig_dist)
    plt.close()

    # Raw data preview
    st.markdown("---")
    st.markdown("### Data Preview")
    st.dataframe(df.head(50), use_container_width=True, hide_index=True)
