Rule-Based Expert System for Automated Credit Risk Assessment
📌 Overview
This project implements a rule-based expert system to assess the credit risk of loan applicants.
The system uses machine learning (rule induction) to generate decision rules, which are then applied through an expert system to classify applicants as Low, Medium, or High risk.
🎯 Objectives
Develop a rule-based model for credit risk classification
Extract interpretable rules from data (RIPPER-style approach)
Implement an expert system using these rules
Compare performance with a baseline model (Decision Tree)
Provide explainable AI outputs for decision transparency
🛠️ Technologies Used
Python
Pandas & NumPy (data processing)
Scikit-learn (baseline model & evaluation)
Experta (rule-based expert system)
Matplotlib / Seaborn (visualisation)
📂 Project Structure
src/            → Core system logic  
data/           → Raw and processed datasets  
notebooks/      → Data exploration (EDA)  
results/        → Evaluation metrics and plots  
report/         → Coursework report files  
video/          → Presentation materials  
⚙️ How It Works
Preprocessing
Clean and prepare the dataset
Rule Learning
Generate classification rules from data
Expert System
Apply rules using a knowledge-based system
Evaluation
Compare performance against a Decision Tree baseline
▶️ How to Run
pip install -r requirements.txt
python main.py
📊 Expected Output
Risk classification (Low / Medium / High)
Evaluation metrics (accuracy, precision, recall)
Visualisations (confusion matrix, ROC curve)
🧠 Key Features
Interpretable decision-making
Rule-based reasoning
Comparison with machine learning baseline
Explainable AI approach
