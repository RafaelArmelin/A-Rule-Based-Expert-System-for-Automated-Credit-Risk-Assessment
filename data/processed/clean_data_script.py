import pandas as pd

# Load the dataset
df = pd.read_csv('credit_risk_dataset.csv')

# 1. Convert 'cb_person_default_on_file' to binary (Y = 1, N = 0)
df['cb_person_default_on_file'] = df['cb_person_default_on_file'].map({'Y': 1, 'N': 0})

# 2. Missing cells in loan_int_rate are replaced with calculation for the median
median_int_rate = df['loan_int_rate'].median()
df['loan_int_rate'] = df['loan_int_rate'].fillna(median_int_rate)

# 3. Missing cells in person_emp_length are replaced with calculation for the median
median_emp_length = df['person_emp_length'].median()
df['person_emp_length'] = df['person_emp_length'].fillna(median_emp_length)


# 4. Removing ages over 100 and remove error in employment length e.g. 22 year old with 123 years of experience
df = df[df['person_age'] < 100]
df = df[df['person_emp_length'] < 60]

# Save the cleaned version
df.to_csv('cleaned_credit_risk_data.csv', index=False)

print("Data Cleaning Complete.")
