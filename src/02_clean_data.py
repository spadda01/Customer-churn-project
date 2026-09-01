import pandas as pd

dataset_file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(dataset_file_path)

# Turn 'TotalCharges' into a true numeric column without breaking the dataset
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors = 'coerce')

# After converting 'TotalCharges' to numeric, check for missing values in the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("missing values info after converting 'TotalCharges' to numeric: ")
print(df['TotalCharges'].isnull().sum())

# Display the rows with missing values in 'TotalCharges' to understand the context of these missing values
print("------------------------------------------------------------------------------------------------------------------")
print(df[df.isnull().any(axis = 1)])

# Fill the missing values in 'TotalCharges' with 0 to maintain the integrity of the dataset and avoid losing valuable information
print("------------------------------------------------------------------------------------------------------------------")
print("Filling missing values in 'TotalCharges' with 0...")
df['TotalCharges'] = df['TotalCharges'].fillna(0)

# After filling the missing values, check again for any remaining missing values in the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("missing values info after filling missing values in 'TotalCharges': ")
print(df.isnull().sum())

# Save the cleaned dataset to a new CSV file for future use
print("------------------------------------------------------------------------------------------------------------------")
df.to_csv("data/processed/telco_customer_churn_cleaned.csv", index=False)
print("Cleaned dataset saved successfully.")