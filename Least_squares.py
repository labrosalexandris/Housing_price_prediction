import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = "housing.csv"
data = pd.read_csv(file_path)

# Check for missing values and handle them, we observed that the only column with empty values was total bedrooms 
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

X = data.drop("median_house_value", axis=1)
y = data["median_house_value"].values.reshape(-1, 1)  # Reshape for scaler

# Define numeric and ategorical features
numeric_features = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                    "total_bedrooms", "population", "households", "median_income"]
categorical_features = ["ocean_proximity"]

# Create preprocessing pipelines
numeric_transformer = Pipeline(steps=[
    ("scaler",MinMaxScaler())
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to features
X_processed = preprocessor.fit_transform(X)

# Scale target (y)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Add bias term (column of ones) to the features
X_bias = np.hstack([np.ones((X_processed.shape[0], 1)), X_processed])

# Function for predictions
def predict(X_new, weights):
    """Predict values using the trained weights."""
    if X_new.shape[1] + 1 == weights.shape[0]:
        # Add bias term if missing
        X_new = np.hstack([np.ones((X_new.shape[0], 1)), X_new])
    return X_new @ weights

# 10-fold cross-validation
kf = KFold(n_splits=10, shuffle=True, random_state=42)
mse_train, mse_test = [], []
mae_train, mae_test = [], []

for train_index, test_index in kf.split(X_bias):
    # Split data into training and testing sets
    X_train, X_test = X_bias[train_index], X_bias[test_index]
    y_train, y_test = y_scaled[train_index], y_scaled[test_index]

    # Compute weights using the pseudo-inverse
    X_transpose = X_train.T
    weights = np.linalg.pinv(X_transpose @ X_train) @ X_transpose @ y_train

    # Predictions for training and testing sets
    y_train_pred = predict(X_train, weights)
    y_test_pred = predict(X_test, weights)

    # Calculate metrics
    mse_train.append(mean_squared_error(y_train, y_train_pred))
    mse_test.append(mean_squared_error(y_test, y_test_pred))
    mae_train.append(mean_absolute_error(y_train, y_train_pred))
    mae_test.append(mean_absolute_error(y_test, y_test_pred))

# Compute average metrics across folds
avg_mse_train = np.mean(mse_train)
avg_mse_test = np.mean(mse_test)
avg_mae_train = np.mean(mae_train)
avg_mae_test = np.mean(mae_test)

# Print results
print("10-Fold Cross-Validation Results:")
print(f"Average MSE (Train): {avg_mse_train:.6f}")
print(f"Average MSE (Test): {avg_mse_test:.6f}")
print(f"Average MAE (Train): {avg_mae_train:.6f}")
print(f"Average MAE (Test): {avg_mae_test:.6f}")

# Example usage
predictions = predict(X_bias, weights)

# Rescale predictions to original target scale
predictions_rescaled = y_scaler.inverse_transform(predictions)

# Print weights and example predictions
print("Trained Weights:", weights.flatten())
print("Example Predictions:", predictions_rescaled[:5].flatten())
