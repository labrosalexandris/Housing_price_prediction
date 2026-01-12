import pandas as pd
import numpy as np
import torch 
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
file_path = "housing.csv"
data = pd.read_csv(file_path)

# Check for missing values in the original data
data["total_bedrooms"] = data["total_bedrooms"].fillna(data["total_bedrooms"].median())

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

# Define the MLP model in PyTorch
class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)  # First hidden layer
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(64, 32)  # Second hidden layer
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(32, 1)  # Output layer

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        return x

# Cross-validation parameters
kf = KFold(n_splits=10, shuffle=True, random_state=42)
num_epochs = 50
batch_size = 32
input_dim = X_processed.shape[1]

# Initialize lists to store metrics
fold_train_mse = []
fold_test_mse = []
fold_train_mae = []
fold_test_mae = []

# Perform 10-fold cross-validation
for fold, (train_index, val_index) in enumerate(kf.split(X_processed)):
    print(f"Fold {fold + 1}")
    
    # Split data into train and validation sets
    X_train, X_val = X_processed[train_index], X_processed[val_index]
    y_train, y_val = y_scaled[train_index], y_scaled[val_index]
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
    y_val_tensor = torch.tensor(y_val, dtype=torch.float32)
    
    # Define the model for this fold
    model = MLP(input_dim)
    
    # Define loss function and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.005, weight_decay=1e-8)
    # Convert data to PyTorch DataLoader for batching
    train_dataset = torch.utils.data.TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        for batch_X, batch_y in train_loader:
            # Forward pass
            outputs = model(batch_X).squeeze()
            loss = criterion(outputs, batch_y.squeeze())
            
            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Evaluate the model on training and validation sets
    model.eval()
    with torch.no_grad():
        train_predictions = model(X_train_tensor).squeeze()
        val_predictions = model(X_val_tensor).squeeze()
    
    # Compute metrics for this fold
    train_mse = mean_squared_error(y_train, train_predictions.numpy())
    val_mse = mean_squared_error(y_val, val_predictions.numpy())
    train_mae = mean_absolute_error(y_train, train_predictions.numpy())
    val_mae = mean_absolute_error(y_val, val_predictions.numpy())
    
    fold_train_mse.append(train_mse)
    fold_test_mse.append(val_mse)
    fold_train_mae.append(train_mae)
    fold_test_mae.append(val_mae)
    
# Compute average metrics across all folds
avg_train_mse = np.mean(fold_train_mse)
avg_test_mse = np.mean(fold_test_mse)
avg_train_mae = np.mean(fold_train_mae)
avg_test_mae = np.mean(fold_test_mae)

print(f"\nAverage Training MSE: {avg_train_mse:.4f}, MAE: {avg_train_mae:.4f}")
print(f"Average Validation MSE: {avg_test_mse:.4f}, MAE: {avg_test_mae:.4f}")
