import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


###################### Setup from Part 4 #############################################################
    # Step 1: Load the cleaned dataset
new_file_path = "data/processed/telco_customer_churn_cleaned.csv"
df = pd.read_csv(new_file_path)

    # step 2: Create X and y
X = df.drop(columns = ['Churn', 'customerID'])
churn_map = {'Yes': 1, 'No': 0}
y = df['Churn'].map(churn_map)

    # step 3: split X and y into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

    # step 4: Create processor
categorical_selector = make_column_selector(dtype_include = ['object', 'category'])
processor = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), categorical_selector)], remainder = 'passthrough')
processor.set_output(transform = "pandas")
X_train_encoded = processor.fit_transform(X_train)
X_test_encoded = processor.transform(X_test)

####################### End of setup ######################################################################


"""
Step 1: 
    Scale the features
"""
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)


"""
Step 2: 
    Initialize and train the model:
        - Fit the logistic regression model on the scaled training data
            - Logistic Regression because churn prediction is a binary classification problem
"""
model = LogisticRegression(random_state = 42) # intializing the model
model.fit(X_train_scaled, y_train) # training the model


"""
step 3:
    Make predictions
"""
predictions = model.predict(X_test_scaled) # predict binary classes (0 or 1)
probabilities = model.predict_proba(X_test_scaled) # predict class probabilities (e.g., [probability of 0, probability of 1])


"""
step 4: 
    Evaluate the model
"""
    # calculate the total accuracy
accuracy = accuracy_score(y_test, predictions)
print(f"Accuracy: {accuracy}\n")

    # Display the confusion matrix
print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
print("\nClassification Report:")
print(classification_report(y_test, predictions))