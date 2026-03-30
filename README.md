# 💳 Credit Risk Expert System

> A rule-based AI system for automated credit risk assessment using interpretable decision rules.

---

## 📌 Overview

This project builds a **rule-based expert system** that evaluates the creditworthiness of applicants.
It combines **machine learning (rule induction)** with **symbolic AI (expert systems)** to produce transparent and explainable decisions.

---

## 🎯 Key Features

* ✅ Rule-based classification (Low / Medium / High risk)
* ✅ Interpretable AI decisions
* ✅ Machine learning + expert system integration
* ✅ Baseline comparison (Decision Tree)
* ✅ Visual evaluation (confusion matrix, ROC curves)

---

## 🧠 How It Works

```text
Data → Preprocessing → Rule Learning → Expert System → Evaluation
```

1. **Preprocessing** – clean and prepare data
2. **Rule Learning** – extract decision rules
3. **Expert System** – apply rules using inference engine
4. **Evaluation** – compare with baseline model

---

## 🛠️ Tech Stack

| Tool         | Purpose            |
| ------------ | ------------------ |
| Python       | Core language      |
| Pandas       | Data processing    |
| Scikit-learn | Baseline + metrics |
| Experta      | Rule-based system  |
| Matplotlib   | Visualisations     |

---

## 📂 Project Structure

```text
credit-risk-expert-system/
├── src/            # Core logic
├── data/           # Datasets
├── notebooks/      # Exploration
├── results/        # Outputs
├── report/         # Coursework
└── video/          # Presentation
```

---

## ▶️ Run the Project

```bash
pip install -r requirements.txt
python main.py
```

---

## 📊 Output

* Risk classification
* Model performance metrics
* Visual evaluation graphs

---

## 🧩 Example Rule

```text
IF income > 50,000 AND debt < 10,000 THEN Risk = Low
```

---

## 🧠 Why This Matters

Unlike black-box models, this system:

* Explains decisions clearly
* Mimics human expert reasoning
* Supports transparent AI systems

