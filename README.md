# A Rule-Based Expert System for Automated Credit Risk Assessment

CS6053 Artificial Intelligence and Machine Learning — Spring 2026, London Metropolitan University

A Python-based expert system that learns credit risk classification rules from data using the RIPPER algorithm and operationalises them through a forward-chaining inference engine built with `experta`.

## How It Works

1. **Preprocessing** — Cleans and encodes the [Kaggle Credit Risk Dataset](https://www.kaggle.com/datasets/laotse/credit-risk-dataset) (32,575 records)
2. **Rule Learning** — Extracts human-readable IF-THEN rules using RIPPER (`wittgenstein`)
3. **Expert System** — 24 hand-crafted rules in an `experta` knowledge engine covering data validation, affordability, and manual review
4. **Evaluation** — Compares performance against a Decision Tree baseline
5. **Dashboard** — Streamlit web interface for individual assessment, model comparison, and dataset exploration

## Setup

```bash
git clone https://github.com/RafaelArmelin/A-Rule-Based-Expert-System-for-Automated-Credit-Risk-Assessment.git
cd A-Rule-Based-Expert-System-for-Automated-Credit-Risk-Assessment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
# Run the Streamlit dashboard
streamlit run app.py

# Or run individual components
python src/rule_learning.py
python src/expert_system.py
python src/evaluation.py
```

## Project Structure

```
├── app.py                  # Streamlit dashboard
├── requirements.txt
├── data/
│   ├── raw/                # Original dataset
│   └── processed/          # Cleaned data and preprocessing script
├── src/
│   ├── preprocessing.py    # Data cleaning and encoding
│   ├── rule_learning.py    # RIPPER training and rule extraction
│   ├── expert_system.py    # experta inference engine (24 rules)
│   ├── evaluation.py       # Metrics and visualisations
│   └── baseline.py         # Decision Tree comparison
├── notebooks/
│   └── exploration.ipynb   # Exploratory data analysis
├── results/
│   ├── figures/            # Saved plots
│   └── metrics/            # Evaluation outputs
└── report/                 # Coursework report
```

## Key Results

| Model | Accuracy | F1 Score | ROC-AUC |
|-------|----------|----------|---------|
| RIPPER | 0.9130 | 0.7557 | 0.8075 |
| Expert System (24 rules) | 0.8037 | 0.4343 | N/A |
| Decision Tree (baseline) | 0.9110 | 0.7438 | 0.8847 |

The expert system trades predictive accuracy for full decision transparency — every outcome is accompanied by the specific rules that triggered it.

## Requirements

- Python 3.10+
- pandas, numpy, scikit-learn
- wittgenstein (RIPPER implementation)
- experta (expert system shell)
- matplotlib, seaborn
- streamlit

## Authors

- Rafael Armelin (22034439)
- Sebastian Mills (23002128)
- Sergiu Mita (22030807)

## Licence

This project is for academic purposes only.
