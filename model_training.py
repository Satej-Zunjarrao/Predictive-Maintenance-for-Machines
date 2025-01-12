import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Function to prepare data for training by splitting into features and labels
def prepare_data(data: pd.DataFrame, label_column: str):
    """
    This function prepares the data for training by splitting it into features and labels.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor data with engineered features.
    - label_column (str): The name of the column that contains the labels (i.e., machine states).
    
    Returns:
    - X_train (DataFrame): Features for training.
    - X_test (DataFrame): Features for testing.
    - y_train (Series): Labels for training.
    - y_test (Series): Labels for testing.
    """
    # Split data into features (X) and labels (y)
    X = data.drop(columns=[label_column])
    y = data[label_column]
    
    # Split data into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Data successfully split into training and testing sets (80% train, 20% test).")
    return X_train, X_test, y_train, y_test

# Function to train a Random Forest classifier
def train_random_forest(X_train, y_train, X_test, y_test):
    """
    This function trains a Random Forest classifier and evaluates its performance.
    
    Args:
    - X_train (DataFrame): Features for training.
    - y_train (Series): Labels for training.
    - X_test (DataFrame): Features for testing.
    - y_test (Series): Labels for testing.
    
    Returns:
    - None
    """
    # Initialize and train the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Predict on the test set
    y_pred = rf_model.predict(X_test)
    
    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Random Forest Model Accuracy: {accuracy * 100:.2f}%")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

# Function to train a Neural Network using TensorFlow
def train_neural_network(X_train, y_train, X_test, y_test):
    """
    This function trains a Neural Network model using TensorFlow and evaluates its performance.
    
    Args:
    - X_train (DataFrame): Features for training.
    - y_train (Series): Labels for training.
    - X_test (DataFrame): Features for testing.
    - y_test (Series): Labels for testing.
    
    Returns:
    - None
    """
    # Convert data to NumPy arrays
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values
    
    # Build a simple neural network model
    model = Sequential()
    model.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # Binary classification
    
    # Compile the model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))
    
    # Evaluate the model
    _, accuracy = model.evaluate(X_test, y_test)
    print(f"Neural Network Model Accuracy: {accuracy * 100:.2f}%")

# Main function to train and evaluate models
def train_and_evaluate_models(data: pd.DataFrame, label_column: str):
    """
    This function trains and evaluates multiple models (Random Forest and Neural Network).
    
    Args:
    - data (DataFrame): The input pandas DataFrame with sensor data and engineered features.
    - label_column (str): The column that contains the labels (machine state).
    
    Returns:
    - None
    """
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data, label_column)
    
    # Train and evaluate Random Forest
    print("\nTraining Random Forest Classifier...")
    train_random_forest(X_train, y_train, X_test, y_test)
    
    # Train and evaluate Neural Network
    print("\nTraining Neural Network...")
    train_neural_network(X_train, y_train, X_test, y_test)

# Example usage of model training
if __name__ == "__main__":
    # Example engineered data (replace with actual data from feature engineering)
    engineered_data = pd.read_csv("satej_data/engineered_sensor_data.csv")  # Replace with your file path
    
    # Define label column (e.g., 'machine_state' for this example)
    label_column = 'machine_state'  # Replace with your actual label column name
    
    # Train and evaluate models
    train_and_evaluate_models(engineered_data, label_column)
