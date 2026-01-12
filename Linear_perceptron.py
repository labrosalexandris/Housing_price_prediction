import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
file_path = "housing.csv"
data = pd.read_csv(file_path)

# Check for missing values in the original data
print("Missing Values Before Any Processing:\n", data.isnull().sum())

# Fill missing values in `total_bedrooms` with its median
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())
# Verify the change
print("\nMissing Values After Filling `total_bedrooms` with Median:\n", data.isnull().sum())

# Separate features (X) and target (y)
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"].values.reshape(-1, 1)  # Reshape for scaler

# Feature categorization
numeric_features = ["longitude", "latitude", "housing_median_age", "total_rooms", 
                    "total_bedrooms", "population", "households", "median_income"]
categorical_features = ["ocean_proximity"]

# Create preprocessing pipelines for features
numeric_transformer = Pipeline(steps=[
    ("scaler", MinMaxScaler())  # Scaling numeric features
])

categorical_transformer = Pipeline(steps=[
    ("onehot", OneHotEncoder(handle_unknown="ignore"))  # One-hot encode categorical features
])

# Combine preprocessing steps for features
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# Apply preprocessing to the feature data
X_processed = preprocessor.fit_transform(X)

# Scale target (y)
y_scaler = MinMaxScaler()
y_scaled = y_scaler.fit_transform(y)

# Convert target to binary (+1/-1) based on the scaled median
threshold = y_scaler.transform([[np.median(y)]])[0, 0]
y_binary = np.where(y_scaled >= threshold, 1, -1)

# Define the Perceptron algorithm with exponential decay
def perceptron_train(X_train, y_train, eta_0=0.01, decay_rate=0.1, max_iterations=100):
    n_samples, n_features = X_train.shape
    weights = np.zeros(n_features)
    bias = 0
    
    for t in range(max_iterations):
        learning_rate = eta_0 * np.exp(-decay_rate * t) # Exponential decay for learning tate
        
        converged = True
        for i in range(n_samples):
            z = np.dot(X_train[i], weights) + bias
            y_pred = np.sign(z)
            if y_pred != y_train[i]:
                weights += learning_rate * y_train[i] * X_train[i]
                
                bias += learning_rate * y_train[i]
                converged = False
        if converged:
            break
    return weights, bias

def perceptron_predict(X, weights, bias):
    z = np.dot(X, weights) + bias
    return np.sign(z)

# Initialize 10-fold cross-validation
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

# Storage for metrics
train_mse, test_mse = [], []
train_mae, test_mae = [], []

# Perform 10-fold cross-validation
for train_index, test_index in skf.split(X_processed, y_binary):
    # Split data
    X_train, X_test = X_processed[train_index], X_processed[test_index]
    y_train, y_test = y_binary[train_index], y_binary[test_index]
    
    # Train perceptron
    weights, bias = perceptron_train(X_train, y_train)
   
    # Predict on training and testing sets
    y_train_pred = perceptron_predict(X_train, weights, bias)
    y_test_pred = perceptron_predict(X_test, weights, bias)
 
    # Compute MSE and MAE
    train_mse.append(mean_squared_error(y_train, y_train_pred))
    test_mse.append(mean_squared_error(y_test, y_test_pred))
    train_mae.append(mean_absolute_error(y_train, y_train_pred))
    test_mae.append(mean_absolute_error(y_test, y_test_pred))

# Calculate average metrics
avg_train_mse = np.mean(train_mse)
avg_test_mse = np.mean(test_mse)
avg_train_mae = np.mean(train_mae)
avg_test_mae = np.mean(test_mae)

# Print results
print(f"10-Fold Cross-Validation Results with Exponential Decay:")
print(f"Average Training MSE: {avg_train_mse:.4f}")
print(f"Average Testing MSE: {avg_test_mse:.4f}")
print(f"Average Training MAE: {avg_train_mae:.4f}")
print(f"Average Testing MAE: {avg_test_mae:.4f}")
