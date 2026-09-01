from matplotlib import ticker
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

new_file_path = "data/processed/telco_customer_churn_cleaned.csv"
df = pd.read_csv(new_file_path)

# Graph 1: churn rate by contract type

# Make a temporary numeric churn column and then plot the average churn value by contract type
new_df = df.copy()
new_df['ChurnNumeric'] = new_df['Churn'].map({'Yes': 1, 'No': 0})

# This is for graph 3, where we want to display the internet service type in a more readable format
new_df["InternetServiceLabel"] = new_df["InternetService"].replace({
    "No": "No Internet Service",
    "DSL": "DSL",
    "Fiber optic": "Fiber Optic"
})

sns.set_theme(style = "whitegrid")
plt.figure(figsize=(8, 5))
sns.barplot(data = new_df, x = "Contract", y = "ChurnNumeric", errorbar = None)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
plt.title("Churn Rate by Contract Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Contract Type")
plt.tight_layout()
plt.savefig("visuals/churn_rate_by_contract_type.png")
plt.close()

# Graph 2: churn rate by internet service type

plt.figure(figsize=(8, 5))
sns.barplot(data = new_df, x = "InternetServiceLabel", y = "ChurnNumeric", errorbar = None)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
plt.title("Churn Rate by Internet Service Type")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Internet Service Type")
plt.tight_layout()
plt.savefig("visuals/churn_rate_by_internet_service.png")
plt.close()

# Graph 3: churn rate by payment method

plt.figure(figsize=(8, 5))
sns.barplot(data = new_df, x = "PaymentMethod", y = "ChurnNumeric",
errorbar = None)
plt.gca().yaxis.set_major_formatter(ticker.PercentFormatter(1.0))
plt.title("Churn Rate by Payment Method")
plt.ylabel("Churn Rate (%)")
plt.xlabel("Payment Method")
plt.xticks(rotation = 45)
plt.tight_layout()
plt.savefig("visuals/churn_rate_by_payment_method.png")
plt.close()

# Graph 4: tenure by churn

plt.figure(figsize=(8, 5))
sns.boxplot(data = new_df, x = "ChurnNumeric", y = "tenure")
plt.title("Customer Tenure by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Tenure (Months)")
plt.xticks([0, 1], ["Did not churn", "Churned"])
plt.tight_layout()
plt.savefig("visuals/tenure_by_churn.png")
plt.close()

# Graph 5: monthly charges by churn

plt.figure(figsize=(8, 5))
sns.boxplot(data = new_df, x = "ChurnNumeric", y = "MonthlyCharges")
plt.title("Monthly Charges by Churn Status")
plt.xlabel("Churn")
plt.ylabel("Monthly Charges ($)")
plt.xticks([0, 1], ["Did not churn", "Churned"])
plt.tight_layout()
plt.savefig("visuals/monthly_charges_by_churn.png")
plt.close()
