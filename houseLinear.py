import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler


# Step 1: Load the dataset from a local file or download from Kaggle
def load_data():
    train = pd.read_csv('data/houseData/train.csv')  # Update the path to where your training data is located
    test = pd.read_csv('data/houseData/test.csv')    # Update the path to where your test data is located
    return train, test

# Step 2: Preprocess the dataset
def preprocess_data(train):
    # Handle missing values separately for numerical and categorical columns
    numerical_features = train.select_dtypes(include=[np.number])
    categorical_features = train.select_dtypes(include=[object])

    # Fill missing values
    numerical_features = numerical_features.fillna(numerical_features.median())
    categorical_features = categorical_features.fillna(categorical_features.mode().iloc[0])

    # Scale numerical features
    scaler = StandardScaler()
    numerical_features_scaled = scaler.fit_transform(numerical_features)

    # One-hot encode categorical features
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    categorical_encoded = encoder.fit_transform(categorical_features)

    # Combine scaled numerical and one-hot encoded categorical features
    X = np.hstack([numerical_features_scaled, categorical_encoded])

    # Target variable (SalePrice)
    y = numerical_features['SalePrice']

    return X, y


# Step 3: Train and predict house prices using Linear Regression
def linear_regression_model(X_train, y_train, X_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

# Step 4: Train and predict whether the house will sell for > 180000
def logistic_regression_model(X_train, y_train, X_test):
    # Convert prices to binary: 1 if SalePrice > 180000, 0 otherwise
    y_train_binary = np.where(y_train > 180000, 1, 0)

    # Train logistic regression model with increased max_iter
    model = LogisticRegression(max_iter=500)
    model.fit(X_train, y_train_binary)

    # Make predictions
    y_pred_binary = model.predict(X_test)
    return y_pred_binary

# Step 5: Evaluate models
def evaluate_models(y_true, y_pred_prices, y_pred_binary):
    # Evaluate the regression task using MAE
    mae = mean_absolute_error(y_true, y_pred_prices)
    print(f'Mean Absolute Error (MAE) for price prediction: {mae}')

    # Evaluate the classification task using accuracy
    y_true_binary = np.where(y_true > 180000, 1, 0)
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    print(f'Accuracy for predicting > 180000: {accuracy}')

def main():
    # Load the dataset
    train, test = load_data()

    # Preprocess the data
    X, y = preprocess_data(train)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    # Predict house prices using Linear Regression
    y_pred_prices = linear_regression_model(X_train, y_train, X_val)

    # Predict if house prices are > 180000 using Logistic Regression
    y_pred_binary = logistic_regression_model(X_train, y_train, X_val)

    # Evaluate the models
    evaluate_models(y_val, y_pred_prices, y_pred_binary)

if __name__ == "__main__":
    main()