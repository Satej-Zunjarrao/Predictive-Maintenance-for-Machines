import boto3
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import os
import json
import numpy as np

# Function to save the trained model to AWS S3 (for Random Forest model)
def save_random_forest_model_to_s3(model, model_name: str, bucket_name: str):
    """
    This function saves the trained Random Forest model to AWS S3.
    
    Args:
    - model (RandomForestClassifier): The trained Random Forest model to save.
    - model_name (str): The name of the model file to save.
    - bucket_name (str): The name of the S3 bucket to store the model.
    
    Returns:
    - None
    """
    # Save the model to a local file
    model_path = f'/tmp/{model_name}.pkl'
    joblib.dump(model, model_path)
    
    # Upload the model to S3
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket_name, model_name)
    print(f"Model saved to S3 bucket: {bucket_name} with file name: {model_name}")

# Function to save the trained neural network model to AWS S3
def save_neural_network_model_to_s3(model, model_name: str, bucket_name: str):
    """
    This function saves the trained Neural Network model to AWS S3.
    
    Args:
    - model (Sequential): The trained Neural Network model to save.
    - model_name (str): The name of the model file to save.
    - bucket_name (str): The name of the S3 bucket to store the model.
    
    Returns:
    - None
    """
    # Save the model to a local file
    model_path = f'/tmp/{model_name}'
    model.save(model_path)
    
    # Upload the model to S3
    s3 = boto3.client('s3')
    s3.upload_file(model_path, bucket_name, model_name)
    print(f"Model saved to S3 bucket: {bucket_name} with file name: {model_name}")

# Function to load the Random Forest model from S3
def load_random_forest_model_from_s3(model_name: str, bucket_name: str):
    """
    This function loads the trained Random Forest model from S3.
    
    Args:
    - model_name (str): The name of the model file in the S3 bucket.
    - bucket_name (str): The name of the S3 bucket where the model is stored.
    
    Returns:
    - model: The loaded Random Forest model.
    """
    s3 = boto3.client('s3')
    
    # Download the model from S3 to a local path
    model_path = f'/tmp/{model_name}.pkl'
    s3.download_file(bucket_name, model_name, model_path)
    
    # Load the model using joblib
    model = joblib.load(model_path)
    print(f"Random Forest model loaded from S3: {model_name}")
    return model

# Function to load the Neural Network model from S3
def load_neural_network_model_from_s3(model_name: str, bucket_name: str):
    """
    This function loads the trained Neural Network model from S3.
    
    Args:
    - model_name (str): The name of the model file in the S3 bucket.
    - bucket_name (str): The name of the S3 bucket where the model is stored.
    
    Returns:
    - model: The loaded Neural Network model.
    """
    s3 = boto3.client('s3')
    
    # Download the model from S3 to a local path
    model_path = f'/tmp/{model_name}'
    s3.download_file(bucket_name, model_name, model_path)
    
    # Load the model using TensorFlow
    model = load_model(model_path)
    print(f"Neural Network model loaded from S3: {model_name}")
    return model

# Example usage of model deployment functions
if __name__ == "__main__":
    # Example of saving models to S3
    bucket_name = 'satej-models-bucket'  # Replace with your S3 bucket name
    rf_model = None  # Replace with the actual trained Random Forest model
    nn_model = None  # Replace with the actual trained Neural Network model
    
    save_random_forest_model_to_s3(rf_model, 'rf_model.pkl', bucket_name)
    save_neural_network_model_to_s3(nn_model, 'nn_model.h5', bucket_name)
    
    # Example of loading models from S3
    rf_model_loaded = load_random_forest_model_from_s3('rf_model.pkl', bucket_name)
    nn_model_loaded = load_neural_network_model_from_s3('nn_model.h5', bucket_name)
