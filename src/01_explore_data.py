import pandas as pd

dataset_file_path = "data/raw/WA_Fn-UseC_-Telco-Customer-Churn.csv"
df = pd.read_csv(dataset_file_path)


# Display the shape of the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("shape of dataframe: ", df.shape)

# Display the columns in the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("columns in the dataset: ")
print(df.columns.tolist())

# Display the data types of each column
print("------------------------------------------------------------------------------------------------------------------")
print("data types of the columns: ")
print(df.dtypes)

print("------------------------------------------------------------------------------------------------------------------")
print(df.info())

# Display summary statistics of the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("dataframe info: ")
print(df.describe())

# missing values info
print("------------------------------------------------------------------------------------------------------------------")
print("missing values info: ")
print(df.isnull().sum())

# check for duplicates
print("------------------------------------------------------------------------------------------------------------------")
print("number of duplicate rows: ")
print(df.duplicated().sum())

# Display the first few rows of the dataset
print("------------------------------------------------------------------------------------------------------------------")
print("first five rows of the dataset: ")
print(df.head())

# Begin the targeted exploration of the dataset
print("##############################################################################################################################################################################################")
print("Begin the targeted exploration of the dataset")
print("##############################################################################################################################################################################################")

# check for unique values in SeniorCitizen column
print("------------------------------------------------------------------------------------------------------------------")
print("unique values in SeniorCitizen column: ")
print(df['SeniorCitizen'].unique())

# check for unique values in churn column
print("------------------------------------------------------------------------------------------------------------------")
print("unique values in Churn column: ")
print(df['Churn'].unique())

# count the number of 'Yes' and 'No' values in the Churn column and find the percentage of 'Yes' values
print("------------------------------------------------------------------------------------------------------------------")
countNoInChurn = df.Churn == 'No'
countYesInChurn = df.Churn == 'Yes'
print("Number of 'No' values in Churn column:", countNoInChurn.sum())
print("Number of 'Yes' values in Churn column:", countYesInChurn.sum())
percentage = (countYesInChurn.sum() / (countNoInChurn.sum() + countYesInChurn.sum())) * 100
print("Percentage of 'Yes' values in Churn column: {:.2f}%".format(percentage))

# investigating why TotalCharges column is object data type instead of numeric
print("------------------------------------------------------------------------------------------------------------------")
print("Check if TotalCharges contains spaces:", df['TotalCharges'].str.contains(' ').sum())
print("Check if TotalCharges contains alphabetic characters:", df['TotalCharges'].astype(str).str.contains(r'[a-zA-Z]').sum())
print("Check if TotalCharges contains only alphabetic characters:", df['TotalCharges'].astype(str).str.isalpha().sum())

# Categorical search of Contract, InternetService and PaymentMethod columns to determine the percentage of Churn for each category in these columns
print("------------------------------------------------------------------------------------------------------------------")
print("Churn value counts for each category in Contract column: ")
print(df.groupby('Contract')['Churn'].value_counts())
#search for every 'Yes' value in the Churn column for each category in the Contract column and calculate the percentage of 'Yes' values for each category. Loop through.
for category in df['Contract'].unique():
    # count the number of 'Yes' and 'No' values in the Churn column for each unique category in the Contract column
    countYesInChurnForCategory = df[(df['Contract'] == category) & (df['Churn'] == 'Yes')].shape[0]
    countNoInChurnForCategory = df[(df['Contract'] == category) & (df['Churn'] == 'No')].shape[0]
    # do the math
    percentageForCategory = (countYesInChurnForCategory / (countYesInChurnForCategory + countNoInChurnForCategory)) * 100
    # print the result of the math
    print(countYesInChurnForCategory, "churned out of", (countYesInChurnForCategory + countNoInChurnForCategory), "customers in category '{}'".format(category), "which is {:.2f}%".format(percentageForCategory))

print("------------------------------------------------------------------------------------------------------------------")
print("Churn value counts for each category in InternetService column: ")
print(df.groupby('InternetService')['Churn'].value_counts())
#search for every 'Yes' value in the Churn column for each category in the InternetService column and calculate the percentage of 'Yes' values for each category. Loop through.
for category in df['InternetService'].unique():
    # count the number of 'Yes' and 'No' values in the Churn column for each unique category in the InternetService column
    countYesInChurnForCategory = df[(df['InternetService'] == category) & (df['Churn'] == 'Yes')].shape[0]
    countNoInChurnForCategory = df[(df['InternetService'] == category) & (df['Churn'] == 'No')].shape[0]
    # do the math
    percentageForCategory = (countYesInChurnForCategory / (countYesInChurnForCategory + countNoInChurnForCategory)) * 100
    # print the result of the math
    print(countYesInChurnForCategory, "churned out of", (countYesInChurnForCategory + countNoInChurnForCategory), "customers in category '{}'".format(category), "which is {:.2f}%".format(percentageForCategory))

print("------------------------------------------------------------------------------------------------------------------")
print("Churn value counts for each category in PaymentMethod column: ")
print(df.groupby('PaymentMethod')['Churn'].value_counts())
#search for every 'Yes' value in the Churn column for each category in the PaymentMethod column and calculate the percentage of 'Yes' values for each category. Loop through.
for category in df['PaymentMethod'].unique():
    # count the number of 'Yes' and 'No' values in the Churn column for each unique category in the PaymentMethod column
    countYesInChurnForCategory = df[(df['PaymentMethod'] == category) & (df['Churn'] == 'Yes')].shape[0]
    countNoInChurnForCategory = df[(df['PaymentMethod'] == category) & (df['Churn'] == 'No')].shape[0]
    # do the math
    percentageForCategory = (countYesInChurnForCategory / (countYesInChurnForCategory + countNoInChurnForCategory)) * 100
    # print the result of the math
    print(countYesInChurnForCategory, "churned out of", (countYesInChurnForCategory + countNoInChurnForCategory), "customers in category '{}'".format(category), "which is {:.2f}%".format(percentageForCategory))