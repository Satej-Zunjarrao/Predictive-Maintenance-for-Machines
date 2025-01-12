import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy import stats

# Function to clean missing values
def clean_missing_values(data: pd.DataFrame):
    """
    This function handles missing values by replacing them with the column mean.
    
    Args:
    - data (DataFrame): The input pandas DataFrame with missing values.
    
    Returns:
    - DataFrame: A cleaned DataFrame with missing values handled.
    """
    cleaned_data = data.fillna(data.mean())
    print("Missing values successfully handled.")
    return cleaned_data

# Function to detect and remove outliers using Z-score method
def remove_outliers(data: pd.DataFrame, threshold: float = 3.0):
    """
    This function detects and removes outliers from the data using the Z-score method.
    
    Args:
    - data (DataFrame): The input pandas DataFrame with potential outliers.
    - threshold (float): The Z-score threshold to consider as an outlier (default is 3).
    
    Returns:
    - DataFrame: The data with outliers removed.
    """
    z_scores = np.abs(stats.zscore(data.select_dtypes(include=[np.number])))
    data_without_outliers = data[(z_scores < threshold).all(axis=1)]
    print(f"Outliers removed using Z-score method with threshold {threshold}.")
    return data_without_outliers

# Function to normalize sensor readings to a common scale
def normalize_data(data: pd.DataFrame):
    """
    This function normalizes the data using standard scaling (z-score normalization).
    
    Args:
    - data (DataFrame): The input pandas DataFrame with sensor readings.
    
    Returns:
    - DataFrame: The normalized DataFrame.
    """
    scaler = StandardScaler()
    normalized_data = pd.DataFrame(scaler.fit_transform(data.select_dtypes(include=[np.number])))
    normalized_data.columns = data.select_dtypes(include=[np.number]).columns
    print("Data successfully normalized using standard scaling.")
    return normalized_data

# Main function to clean the raw sensor data
def clean_sensor_data(raw_data: pd.DataFrame):
    """
    This function cleans the raw sensor data by handling missing values, removing outliers,
    and normalizing the sensor readings.
    
    Args:
    - raw_data (DataFrame): The raw sensor data to be cleaned.
    
    Returns:
    - DataFrame: The cleaned and processed data.
    """
    # Step 1: Handle missing values
    data_no_missing = clean_missing_values(raw_data)
    
    # Step 2: Remove outliers
    data_no_outliers = remove_outliers(data_no_missing)
    
    # Step 3: Normalize sensor readings
    cleaned_data = normalize_data(data_no_outliers)
    
    print("Sensor data cleaning process complete.")
    return cleaned_data

# Example usage of the data cleaning functions
if __name__ == "__main__":
    # Example data (replace with actual data from the ingestion process)
    raw_data = pd.read_csv("satej_data/raw_sensor_data.csv")  # Replace with your file path
    
    cleaned_data = clean_sensor_data(raw_data)
    cleaned_data.to_csv("satej_data/cleaned_sensor_data.csv", index=False)  # Save cleaned data
