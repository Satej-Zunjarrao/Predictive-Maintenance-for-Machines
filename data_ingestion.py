import pandas as pd
import numpy as np
import boto3
import os
from azure.storage.blob import BlobServiceClient

# Function to read data from AWS S3
def read_data_from_s3(bucket_name: str, file_key: str):
    """
    This function reads data from an AWS S3 bucket.
    
    Args:
    - bucket_name (str): The name of the S3 bucket where the data is stored.
    - file_key (str): The file key (path) to the file in the S3 bucket.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the data from the file.
    """
    s3 = boto3.client('s3')
    
    # Fetch the file from S3 and load it into a DataFrame
    obj = s3.get_object(Bucket=bucket_name, Key=file_key)
    data = pd.read_csv(obj['Body'])
    
    print(f"Data successfully read from S3 bucket: {bucket_name}, file: {file_key}")
    return data

# Function to read data from Azure Blob Storage
def read_data_from_azure(blob_connection_string: str, container_name: str, file_name: str):
    """
    This function reads data from Azure Blob Storage.
    
    Args:
    - blob_connection_string (str): The connection string for the Azure Blob Storage.
    - container_name (str): The name of the container where the data is stored.
    - file_name (str): The name of the file in the Azure container.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the data from the file.
    """
    blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
    blob_client = blob_service_client.get_blob_client(container=container_name, blob=file_name)
    
    # Download blob data into memory and load into pandas DataFrame
    download_stream = blob_client.download_blob()
    data = pd.read_csv(download_stream)
    
    print(f"Data successfully read from Azure Blob: {container_name}, file: {file_name}")
    return data

# Main function to ingest data from a specified source (S3 or Azure)
def ingest_data(source_type: str, source_details: dict):
    """
    This function handles the ingestion of data from either AWS S3 or Azure Blob Storage based on the given source type.
    
    Args:
    - source_type (str): The source type (either 's3' or 'azure').
    - source_details (dict): A dictionary containing the necessary details for the data source.
    
    Returns:
    - DataFrame: A pandas DataFrame containing the ingested data.
    """
    if source_type == 's3':
        bucket_name = source_details.get('bucket_name')
        file_key = source_details.get('file_key')
        return read_data_from_s3(bucket_name, file_key)
    
    elif source_type == 'azure':
        connection_string = source_details.get('connection_string')
        container_name = source_details.get('container_name')
        file_name = source_details.get('file_name')
        return read_data_from_azure(connection_string, container_name, file_name)
    
    else:
        raise ValueError("Invalid source type. Please choose either 's3' or 'azure'.")

# Example usage of the ingestion functions
if __name__ == "__main__":
    # Ingest data from AWS S3
    s3_source_details = {
        'bucket_name': 'satej-sensor-data-bucket',  # Replace with your S3 bucket name
        'file_key': 'sensor_data/raw_data.csv'  # Replace with your file key
    }
    sensor_data_s3 = ingest_data('s3', s3_source_details)
    
    # Ingest data from Azure Blob Storage
    azure_source_details = {
        'connection_string': 'DefaultEndpointsProtocol=https;AccountName=satejstorageaccount;AccountKey=your_account_key;EndpointSuffix=core.windows.net',
        'container_name': 'satej-container',  # Replace with your container name
        'file_name': 'sensor_data/raw_data.csv'  # Replace with your file name
    }
    sensor_data_azure = ingest_data('azure', azure_source_details)
