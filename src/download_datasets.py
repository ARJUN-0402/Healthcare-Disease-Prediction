import pandas as pd
import requests
import os
from io import StringIO

def download_diabetes_dataset():
    """Download Pima Indians Diabetes Dataset from UCI"""
    try:
        # Try multiple sources for the diabetes dataset
        urls = [
            "https://www.kaggle.com/api/v1/datasets/download/uciml/pima-indians-diabetes-database",
            "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
        ]

        # Column names for the diabetes dataset
        columns = [
            'num_pregnant', 'glucose_concentration', 'blood_pressure',
            'skin_thickness', 'insulin', 'bmi', 'diabetes_pedigree',
            'age', 'class'
        ]

        # Try to download from UCI first
        response = requests.get(urls[1])
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text), names=columns)
            return data
        else:
            print(f"Failed to download from {urls[1]}")
            return None
    except Exception as e:
        print(f"Error downloading diabetes dataset: {e}")
        return None

def download_heart_disease_dataset():
    """Download Cleveland Heart Disease Dataset from UCI"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"

        # Column names for heart disease dataset
        columns = [
            'age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg',
            'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'
        ]

        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text), names=columns)
            return data
        else:
            print(f"Failed to download from {url}")
            return None
    except Exception as e:
        print(f"Error downloading heart disease dataset: {e}")
        return None

def download_kidney_disease_dataset():
    """Download Chronic Kidney Disease Dataset from UCI"""
    try:
        url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00528/dataset.csv"

        response = requests.get(url)
        if response.status_code == 200:
            data = pd.read_csv(StringIO(response.text))
            return data
        else:
            print(f"Failed to download from {url}")
            return None
    except Exception as e:
        print(f"Error downloading kidney disease dataset: {e}")
        return None

def main():
    """Main function to download all datasets"""
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')

    print("Downloading datasets...")

    # Download diabetes dataset
    print("Downloading diabetes dataset...")
    diabetes_data = download_diabetes_dataset()
    if diabetes_data is not None:
        diabetes_file = os.path.join(data_dir, 'diabetes.csv')
        diabetes_data.to_csv(diabetes_file, index=False)
        print(f"Diabetes dataset saved to {diabetes_file}")
        print(f"Shape: {diabetes_data.shape}")
    else:
        print("Failed to download diabetes dataset")

    # Download heart disease dataset
    print("\nDownloading heart disease dataset...")
    heart_data = download_heart_disease_dataset()
    if heart_data is not None:
        heart_file = os.path.join(data_dir, 'heart_disease.csv')
        heart_data.to_csv(heart_file, index=False)
        print(f"Heart disease dataset saved to {heart_file}")
        print(f"Shape: {heart_data.shape}")
    else:
        print("Failed to download heart disease dataset")

    # Download kidney disease dataset
    print("\nDownloading kidney disease dataset...")
    kidney_data = download_kidney_disease_dataset()
    if kidney_data is not None:
        kidney_file = os.path.join(data_dir, 'kidney_disease.csv')
        kidney_data.to_csv(kidney_file, index=False)
        print(f"Kidney disease dataset saved to {kidney_file}")
        print(f"Shape: {kidney_data.shape}")
    else:
        print("Failed to download kidney disease dataset")

if __name__ == "__main__":
    main()