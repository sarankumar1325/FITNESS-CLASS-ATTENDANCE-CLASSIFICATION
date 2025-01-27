import pandas as pd
import numpy as np

def load_data(file_path):
    """
    Load the dataset from the given file path.
    """
    df = pd.read_csv('data/fitness_class_2212.csv')
    return df

def clean_data(df):
    """
    Clean the dataset by handling missing values, correcting data types, and standardizing columns.
    """
    # Replace missing weight values with the mean
    df['weight'] = df['weight'].fillna(df['weight'].mean())

    # Convert 'days_before' to integer by removing 'days' suffix
    df['days_before'] = df['days_before'].str.replace(' days', '').astype(int)

    # Standardize 'day_of_week' to 3-letter abbreviations and map to numeric values
    df['day_of_week'] = df['day_of_week'].str[:3]
    day_mapping = {'Mon': 1, 'Tue': 2, 'Wed': 3, 'Thu': 4, 'Fri': 5, 'Sat': 6, 'Sun': 7}
    df['day_of_week'] = df['day_of_week'].map(day_mapping)

    # Replace '-' in 'category' with 'unknown'
    df['category'] = df['category'].replace('-', 'unknown')

    return df

def feature_engineering(df):
    """
    Create new features or transform existing ones to improve model performance.
    """
    # Create a new feature: ratio of months_as_member to days_before
    df['months_to_days_ratio'] = df['months_as_member'] / df['days_before']

    return df

def encode_categorical_variables(df):
    """
    Encode categorical variables using one-hot encoding.
    """
    df = pd.get_dummies(df, columns=['category', 'time'], drop_first=True)
    return df

def preprocess_data(file_path):
    """
    Main function to preprocess the data.
    """
    # Load the data
    df = load_data(file_path)

    # Clean the data
    df = clean_data(df)

    # Perform feature engineering
    df = feature_engineering(df)

    # Encode categorical variables
    df = encode_categorical_variables(df)

    return df

if __name__ == "__main__":
    # Example usage
    file_path = "data/fitness_class_2212.csv"
    processed_df = preprocess_data(file_path)
    print(processed_df.head())