"""
Data preprocessing functions for student marks prediction.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_clean_data(csv_path):
    """
    Loads the dataset and drops missing values.
    """
    df = pd.read_csv(csv_path)  # Read CSV file
    df = df.dropna()            # Remove rows with missing values
    return df

def encode_categorical(df, categorical_cols):
    """
    Encodes categorical columns using LabelEncoder.
    Returns the DataFrame and a dictionary of encoders.
    """
    le_dict = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])  # Encode column
        le_dict[col] = le                    # Store encoder
    return df, le_dict