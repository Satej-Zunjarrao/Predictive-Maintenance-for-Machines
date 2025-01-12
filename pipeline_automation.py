import os
import time
import pandas as pd
from data_ingestion import ingest_data
from data_cleaning import clean_sensor_data
from feature_engineering import engineer_features
from model_training import train_and_evaluate_models
from model_deployment import save_random_forest_model_to_s3, save_neural_network_model_to_s3
import boto3

# Function to automate the data pipeline
def automate_data_pipeline(bucket_name: str, file_key: str, model_save_bucket: str):
    """
    This function automates the end-to-end data pipeline, including data ingestion, cleaning,
    feature engineering, model training, and model deployment.
    
    Args:
    - bucket_name (str): The S3 bucket name where the raw data is stored.
    - file_key (str): The file key in the S3 bucket where the raw data is located.
    - model_save_bucket (str): The S3 bucket name where the trained models will be saved.
    
    Returns:
    - None
    """
    # Step 1: Ingest raw data from S3
    print("Ingesting raw data from S3...")
    raw_data = ingest_data('s3', {'bucket_name': bucket_name, 'file_key': file_key})
    
    # Step 2: Clean the sensor data
    print("Cleaning raw sensor data...")
    cleaned_data = clean_sensor_data(raw_data)
    
    # Step 3: Perform feature engineering
    print("Engineering features from cleaned data...")
    engineered_data = engineer_features(cleaned_data)
    
    # Step 4: Train and evaluate models
    print("Training and evaluating models...")
    train_and_evaluate_models(engineered_data, label_column='machine_state')  # Replace with your label column
    
    # Step 5: Save the trained models to S3
    print("Saving trained models to S3...")
    rf_model = None  # Replace with actual trained model
    nn_model = None  # Replace with actual trained model
    
    save_random_forest_model_to_s3(rf_model, 'rf_model.pkl', model_save_bucket)
    save_neural_network_model_to_s3(nn_model, 'nn_model.h5', model_save_bucket)

# Function to schedule the pipeline automation at specified intervals
def schedule_pipeline_automation(bucket_name: str, file_key: str, model_save_bucket: str, interval: int = 86400):
    """
    This function schedules the data pipeline automation to run at specified intervals.
    
    Args:
    - bucket_name (str): The S3 bucket name where the raw data is stored.
    - file_key (str): The file key in the S3 bucket where the raw data is located.
    - model_save_bucket (str): The S3 bucket name where the trained models will be saved.
    - interval (int): The interval (in seconds) between each pipeline run (default is 86400 seconds or 24 hours).
    
    Returns:
    - None
    """
    while True:
        automate_data_pipeline(bucket_name, file_key, model_save_bucket)
        print(f"Next pipeline run in {interval} seconds...")
        time.sleep(interval)

# Example usage of pipeline automation
if __name__ == "__main__":
    # Example S3 details (replace with actual details)
    bucket_name = 'satej-raw-sensor-data'  # Replace with your S3 bucket name
    file_key = 'sensor_data/raw_data.csv'  # Replace with your S3 file key
    model_save_bucket = 'satej-models-bucket'  # Replace with your S3 model save bucket
    
    # Schedule the pipeline to run every 24 hours (86400 seconds)
    schedule_pipeline_automation(bucket_name, file_key, model_save_bucket)
