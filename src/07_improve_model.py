import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


###################### Setup from Part 4 #########################################################################
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
####################### End of setup part 4 ######################################################################

###################### Setup from Part 5 #########################################################################
    # Step 1: Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_encoded)
X_test_scaled = scaler.transform(X_test_encoded)

    # Step 2: Initialize and train the model
model = LogisticRegression(random_state = 42) # initializing the model
model.fit(X_train_scaled, y_train) # training the model

    # Step 3: Make predictions
predictions = model.predict(X_test_scaled) # predict binary classes (0 or 1)
probabilities = model.predict_proba(X_test_scaled) # predict class probabilities (e.g., [probability of 0, probability of 1])

    # Step 4: Store evaluation metrics for Part 6
accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
class_rep_text = classification_report(y_test, predictions)
class_rep = classification_report(y_test, predictions, output_dict=True)
####################### End of setup part 5 ######################################################################


"""
Step 1: 
    train a balanced Logistic Regression model to see if it improves churn recall.
"""
model_ver2 = LogisticRegression(random_state = 42, class_weight = 'balanced') # pays more attention to the minority class (in this case the smaller churn class)
model_ver2.fit(X_train_scaled, y_train) # training the model
predictions_ver2 = model_ver2.predict(X_test_scaled) # predict binary classes (0 or 1)
accuracy_ver2 = accuracy_score(y_test, predictions_ver2)
conf_matrix_ver2 = confusion_matrix(y_test, predictions_ver2)
class_rep_ver2_text = classification_report(y_test, predictions_ver2)
class_rep_ver2 = classification_report(y_test, predictions_ver2, output_dict=True)

"""
Step 2: 
    train random forest model to see if it improves accuracy.
"""
model_ver3 = RandomForestClassifier(random_state = 42, class_weight = 'balanced')
model_ver3.fit(X_train_scaled, y_train)
predictions_ver3 = model_ver3.predict(X_test_scaled)
accuracy_ver3 = accuracy_score(y_test, predictions_ver3)
conf_matrix_ver3 = confusion_matrix(y_test, predictions_ver3)
class_rep_ver3_text = classification_report(y_test, predictions_ver3)
class_rep_ver3 = classification_report(y_test, predictions_ver3, output_dict=True)

"""
Step 3:
    Evaluate all models using the same test set
"""
print("Accuracy: ")
print(f"\t Logistic Regression Model: {accuracy}")
print(f"\t Balanced Logistic Regression Model: {accuracy_ver2}")
print(f"\t Random Forest Model: {accuracy_ver3}")

print("Confusion Matrix of Logistic Regression Model:")
print(conf_matrix)
print("Confusion Matrix of Balanced Logistic Regression Model:")
print(conf_matrix_ver2)
print("Confusion Matrix of Random Forest Model:")
print(conf_matrix_ver3)

print("Classification Report of Logistic Regression Model:")
print(class_rep_text)
print("Classification Report of Balanced Logistic Regression Model:")
print(class_rep_ver2_text)
print("Classification Report of Random Forest Model:")
print(class_rep_ver3_text)

"""
Step 4:
    Comparison summary table of all models
"""
comparison_table = pd.DataFrame({
    "Model": [
        "Logistic Regression",
        "Balanced Logistic Regression",
        "Random Forest"
    ],
    "Accuracy": [
        accuracy,
        accuracy_ver2,
        accuracy_ver3
    ],
    "Churn Precision": [
        class_rep["1"]["precision"],
        class_rep_ver2["1"]["precision"],
        class_rep_ver3["1"]["precision"]
    ],
    "Churn Recall": [
        class_rep["1"]["recall"],
        class_rep_ver2["1"]["recall"],
        class_rep_ver3["1"]["recall"]
    ],
    "Churn F1": [
        class_rep["1"]["f1-score"],
        class_rep_ver2["1"]["f1-score"],
        class_rep_ver3["1"]["f1-score"]
    ]
})

print(comparison_table.round(2))

comparison_table.to_csv("data/processed/model_comparison.csv", index=False)