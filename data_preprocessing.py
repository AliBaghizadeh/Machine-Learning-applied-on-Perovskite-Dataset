import pandas as pd
import numpy as np

def load_data(file_path):
    """"Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def clean_data(df):
    """Clean and preprocess the data for modeling."""
    # Convert specific columns to numeric
    col_change = ['Formation energy [eV/atom]', 'Stability [eV/atom]',
                  'Magnetic moment [mu_B]', 'Volume per atom [A^3/atom]', 
                  'Band gap [eV]', 'a [ang]', 'b [ang]', 'c [ang]', 
                  'alpha [deg]', 'beta [deg]', 'gamma [deg]', 'Vacancy energy [eV/O atom]']
    for col in col_change:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Encode the target variable (LowestDist)
    ordinal_encoding = {"cubic": 1, "orthorhombic": 2, "rhombohedral": 3, "tetragonal": 4, "-": 5}
    df["LowestDist"] = df["Lowest distortion"].replace(ordinal_encoding)
    
    # Drop columns and rows with missing values
    df = df.drop("Magnetic moment [mu_B]", axis=1)
    df = df.dropna(axis=0)
    
    return df

def split_features_labels(df, label_col="LowestDist"):
    """Split the data into features and labels."""
    features = df.drop(label_col, axis=1)
    labels = df[label_col]
    return features, labels

