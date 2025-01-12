import pandas as pd
import numpy as np

# Function to create rolling averages as features
def create_rolling_averages(data: pd.DataFrame, window_size: int = 5):
    """
    This function creates rolling averages for key sensor readings to capture trends.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor readings.
    - window_size (int): The window size for calculating the rolling average (default is 5).
    
    Returns:
    - DataFrame: A DataFrame with rolling averages added as new features.
    """
    rolling_avg_data = data.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        rolling_avg_data[f'{column}_rolling_avg'] = rolling_avg_data[column].rolling(window=window_size).mean()
    
    print(f"Rolling averages created with a window size of {window_size}.")
    return rolling_avg_data

# Function to create time-lagged features for trend analysis
def create_time_lagged_features(data: pd.DataFrame, lag: int = 1):
    """
    This function creates time-lagged features to capture trends and cyclic behavior of sensor data.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor readings.
    - lag (int): The lag time for generating time-lagged features (default is 1).
    
    Returns:
    - DataFrame: A DataFrame with time-lagged features added.
    """
    lagged_data = data.copy()
    for column in data.select_dtypes(include=[np.number]).columns:
        lagged_data[f'{column}_lag_{lag}'] = lagged_data[column].shift(lag)
    
    print(f"Time-lagged features created with a lag of {lag}.")
    return lagged_data

# Function to extract statistical features from sensor data
def create_statistical_features(data: pd.DataFrame):
    """
    This function extracts statistical features like mean, variance, and peak values from sensor data.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor readings.
    
    Returns:
    - DataFrame: A DataFrame with additional statistical features.
    """
    statistical_features = data.copy()
    statistical_features['mean'] = statistical_features.mean(axis=1)
    statistical_features['variance'] = statistical_features.var(axis=1)
    statistical_features['peak'] = statistical_features.max(axis=1)
    
    print("Statistical features (mean, variance, peak) created.")
    return statistical_features

# Main function to perform feature engineering
def engineer_features(raw_data: pd.DataFrame, window_size: int = 5, lag: int = 1):
    """
    This function performs feature engineering by combining rolling averages, time-lagged features,
    and statistical features for the sensor data.
    
    Args:
    - raw_data (DataFrame): The raw sensor data.
    - window_size (int): The window size for rolling averages (default is 5).
    - lag (int): The lag value for time-lagged features (default is 1).
    
    Returns:
    - DataFrame: A DataFrame with engineered features.
    """
    # Step 1: Create rolling averages
    data_with_rolling_avg = create_rolling_averages(raw_data, window_size)
    
    # Step 2: Create time-lagged features
    data_with_lags = create_time_lagged_features(data_with_rolling_avg, lag)
    
    # Step 3: Create statistical features
    engineered_data = create_statistical_features(data_with_lags)
    
    print("Feature engineering process complete.")
    return engineered_data

# Example usage of feature engineering
if __name__ == "__main__":
    # Example raw data (replace with actual data)
    raw_data = pd.read_csv("satej_data/cleaned_sensor_data.csv")  # Replace with your file path
    
    # Perform feature engineering
    engineered_data = engineer_features(raw_data, window_size=5, lag=1)
    
    # Save the engineered features to a new CSV file
    engineered_data.to_csv("satej_data/engineered_sensor_data.csv", index=False)  # Save to file
