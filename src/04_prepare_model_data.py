import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

"""Step 1: Load the cleaned dataset"""
print('-----------------Step 1: Load the cleaned dataset-----------------')
new_file_path = "data/processed/telco_customer_churn_cleaned.csv"
df = pd.read_csv(new_file_path)

    # list the columns of the DataFrame
print('All the columns: \n', df.columns.to_list(), '\n')


"""
Step 2: 
    Create Features column(X)
"""
X = df.drop(columns = ['Churn', 'customerID'])


"""
Step 3: 
    Create y by mapping Churn Yes/No to 1/0
    Create target column (y)
"""
churn_map = {'Yes': 1, 'No': 0}
    # target column
y = df['Churn'].map(churn_map)


"""
Step 4: 
    Check X shape and y shape
"""
print('X shape:', X.shape)
print('y shape:', y.shape)


"""
Step 5: 
    Check y value counts
"""
print('\n y value counts:')
print(y.value_counts())


"""
Step 6: 
    Encode categorical variables
"""
    # 1. Select only object or categorical data types dynamically
categorical_selector = make_column_selector(dtype_include = ['object', 'category'])
    # 2. Set up the transformer to apply OneHotEncoder to categorical columns
processor = ColumnTransformer(transformers = [('cat', OneHotEncoder(handle_unknown = 'ignore', sparse_output = False), categorical_selector)], remainder = 'passthrough')
    # 3. Configure scikit-learn to return a Pandas DataFrame directly
processor.set_output(transform = "pandas")
    # 4. Fit and transform the data
X_encoded = processor.fit_transform(X)


"""
    Step 6 checkpoint:
        encoded DataFrame shape
        encoded DataFrame columns
        encoded DataFrame data types
        encoded DataFrame object columns remaining after encoding
"""
print('Encoded DataFrame shape:', X_encoded.shape)
print('Encoded DataFrame columns that are object types:', X_encoded.select_dtypes(include=["object"]).columns.tolist())


"""
    Step 7: 
        Train/test split
        Verify the shapes
"""
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size = 0.2, random_state = 42, stratify = y)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)
print('y_train shape:', y_train.shape)
print('y_test shape:', y_test.shape)