import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import boto3
import json

# Function to visualize sensor data trends using line plots
def visualize_sensor_trends(data: pd.DataFrame, columns: list):
    """
    This function visualizes the sensor data trends over time for selected columns.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor data.
    - columns (list): A list of column names to visualize.
    
    Returns:
    - None
    """
    plt.figure(figsize=(10, 6))
    for column in columns:
        plt.plot(data['timestamp'], data[column], label=column)
    plt.xlabel('Timestamp')
    plt.ylabel('Sensor Reading')
    plt.title('Sensor Data Trends Over Time')
    plt.legend()
    plt.show()
    print("Sensor trends visualization complete.")

# Function to visualize the distribution of sensor readings using histograms
def visualize_sensor_distribution(data: pd.DataFrame, columns: list):
    """
    This function visualizes the distribution of sensor readings for selected columns using histograms.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor data.
    - columns (list): A list of column names to visualize.
    
    Returns:
    - None
    """
    data[columns].hist(bins=30, figsize=(10, 6))
    plt.suptitle('Sensor Data Distribution')
    plt.show()
    print("Sensor distribution visualization complete.")

# Function to visualize correlation between sensor readings using a heatmap
def visualize_sensor_correlation(data: pd.DataFrame):
    """
    This function visualizes the correlation matrix between different sensor readings using a heatmap.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor data.
    
    Returns:
    - None
    """
    corr = data.corr()
    plt.figure(figsize=(10, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Matrix of Sensor Readings')
    plt.show()
    print("Sensor correlation heatmap visualization complete.")

# Function to create a real-time monitoring dashboard (simulated)
def create_dashboard(data: pd.DataFrame):
    """
    This function creates a simple dashboard to visualize key metrics from the sensor data.
    
    Args:
    - data (DataFrame): The input pandas DataFrame containing sensor data.
    
    Returns:
    - None
    """
    # Visualize trends for selected sensors
    visualize_sensor_trends(data, ['temperature', 'vibration', 'pressure'])
    
    # Visualize distribution of sensor readings
    visualize_sensor_distribution(data, ['temperature', 'vibration', 'pressure'])
    
    # Visualize correlation matrix
    visualize_sensor_correlation(data)

# Example usage of dashboard visualization
if __name__ == "__main__":
    # Example sensor data (replace with actual data)
    sensor_data = pd.read_csv("satej_data/cleaned_sensor_data.csv")  # Replace with your file path
    
    # Create dashboard visualization
    create_dashboard(sensor_data)
