# Predictive-Maintenance-for-Machines
Built a predictive maintenance system to identify potential equipment failures before they occur, using historical sensor data and real-time monitoring.

# Predictive Maintenance System

## Overview
The **Predictive Maintenance System** is a Python-based solution designed to predict equipment failures before they occur by analyzing historical sensor data and real-time monitoring inputs. The system utilizes machine learning models to classify machine states into normal or at-risk categories, aiming to reduce unplanned downtime, optimize maintenance schedules, and enhance operational efficiency.

This project includes an automated pipeline for data collection, cleaning, feature engineering, predictive modeling, deployment, and real-time monitoring.

---

## Key Features
- **Data Collection**: Collects sensor data (e.g., temperature, vibration, pressure) from machines via IoT devices and integrates it into a centralized system.
- **Data Cleaning**: Preprocesses sensor data, handling missing values, normalizing readings, and removing outliers.
- **Feature Engineering**: Creates key features like rolling averages, time-lagged features, and statistical measures (mean, variance, peak values).
- **Predictive Modeling**: Builds ML models (Random Forest, Neural Networks) to classify machine states and predict failure likelihood.
- **Model Deployment**: Deploys trained models to AWS S3 for scalable usage and real-time prediction.
- **Real-time Monitoring**: Monitors machine health metrics in real-time and triggers alerts when failure risks are detected.
- **Visualization**: Generates real-time dashboards and trend analysis visualizations for monitoring and reporting.
- **Automation**: Automates data ingestion, cleaning, model training, and prediction processes to ensure continuous updates.

---

## Directory Structure
```
project/
│
├── data_ingestion.py           # Handles data ingestion from AWS S3 or Azure Blob Storage
├── data_cleaning.py            # Preprocesses sensor data (missing values, normalization, outlier removal)
├── feature_engineering.py      # Creates features like rolling averages, time-lagged features, and statistical measures
├── model_training.py           # Trains and evaluates machine learning models (Random Forest, Neural Networks)
├── model_deployment.py        # Saves and loads trained models to/from AWS S3
├── real_time_monitoring.py     # Monitors real-time sensor data and triggers alerts based on predictions
├── dashboard_visualization.py  # Generates visualizations of sensor data trends and failure risks
├── pipeline_automation.py      # Automates the pipeline for continuous data processing and model updates
├── config.py                   # Stores configuration details (e.g., AWS keys, file paths, model parameters)
├── utils.py                    # Provides helper functions for logging, model evaluation, etc.
├── README.md                   # Project documentation
```

# Modules

## 1. data_ingestion.py
- Handles data ingestion from cloud storage platforms (AWS S3, Azure Blob Storage).
- Integrates sensor data into a centralized system for preprocessing.

## 2. data_cleaning.py
- Preprocesses raw sensor data by handling missing values, normalizing readings, and removing outliers.
- Outputs a cleaned dataset for feature engineering and model training.

## 3. feature_engineering.py
- Creates engineered features such as rolling averages, time-lagged features, and statistical metrics.
- Prepares the dataset for model training.

## 4. model_training.py
- Trains machine learning models (e.g., Random Forest, Neural Networks) for classifying machine states and predicting breakdown likelihood.
- Evaluates model performance and selects the best-performing model.

## 5. model_deployment.py
- Saves trained models to AWS S3 for easy access and real-time deployment.
- Loads models for real-time predictions based on incoming sensor data.

## 6. real_time_monitoring.py
- Monitors sensor data in real-time, triggers failure predictions, and alerts based on model predictions.
- Ensures that machine health is continuously tracked and high-risk conditions are flagged.

## 7. dashboard_visualization.py
- Generates visualizations for trends in sensor data, highlighting key metrics such as failure risks.
- Displays visual reports for stakeholders to monitor machine health and failure likelihood.

## 8. pipeline_automation.py
- Automates the entire predictive maintenance pipeline, including data ingestion, cleaning, feature engineering, and model training.
- Ensures that the system operates continuously without manual intervention.

## 9. config.py
- Stores configuration settings, such as AWS credentials, model paths, and other reusable parameters.

## 10. utils.py
- Helper functions for logging, error handling, metric calculations, and managing file paths.

---

## Contact

For queries or collaboration, feel free to reach out:

- **Name**: Satej Zunjarrao  
- **Email**: zsatej1028@gmail.com
