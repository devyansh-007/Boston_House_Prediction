import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Load datasets from Google Drive (adjust file paths based on the actual location)
train_df = pd.read_csv('/content/drive/MyDrive/ML/Boston.csv')  # Adjust path if necessary
test_df = pd.read_csv('/content/drive/MyDrive/ML/test.csv')     # Adjust path if necessary

# Check columns to ensure correct ones are used
print("Train DataFrame Columns:", train_df.columns)
print("Test DataFrame Columns:", test_df.columns)

# Drop 'Unnamed: 0' if it exists in train_df (it might be an index column)
if 'Unnamed: 0' in train_df.columns:
    train_df = train_df.drop(columns=['Unnamed: 0'])

# Prepare the data for training
X_train = train_df.drop(columns=['medv'])
y_train = train_df['medv']

# Split training data for validation
X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train_split, y_train_split)

# Validate the model
y_val_pred = model.predict(X_val_split)
mse = mean_squared_error(y_val_split, y_val_pred)
print(f"Mean Squared Error: {mse}")

# Drop 'ID' from the test set if it exists
X_test = test_df.drop(columns=['ID'])

# Make predictions on the test set
test_predictions = model.predict(X_test)

# Save the predictions to Google Drive
test_df['predictions'] = test_predictions
test_df.to_csv('/content/drive/MyDrive/ML/test_predictions.csv', index=False)  # Ensure the folder exists in this path
