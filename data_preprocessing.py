import pandas as pd
import numpy as np

def load_data(file_path):
    """"Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(df):
    # Step 1: Convert columns to numeric
    col_change = ['Formation energy [eV/atom]', 'Stability [eV/atom]', 
                  'Magnetic moment [mu_B]', 'Volume per atom [A^3/atom]', 
                  'Band gap [eV]', 'a [ang]', 'b [ang]', 'c [ang]', 
                  'alpha [deg]', 'beta [deg]', 'gamma [deg]', 
                  'Vacancy energy [eV/O atom]']
    
    for col in col_change:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Step 2: Encode 'Lowest distortion' column
    ordinal_encoding = {"cubic": 1, "orthorhombic": 2, "rhombohedral": 3, 
                        "tetragonal": 4, "-": 5}
    df["LowestDist"] = df["Lowest distortion"].replace(ordinal_encoding)

    # Step 3: Drop irrelevant or problematic columns
    df = df.drop(["Chemical formula", "Lowest distortion"], axis=1)

    # Step 4: Handle missing values (drop rows with NaN values)
    df = df.dropna()

    return df



def split_features_labels(df, label_col="LowestDist"):
    """Split the data into features and labels."""
    # Exclude non-numeric columns
    exclude_cols = ["A", "B", "Valence A", "Valence B"]
    features = df.drop([label_col] + exclude_cols, axis=1)

    # Ensure only numeric columns are included
    features = features.select_dtypes(include=["float64", "int64"])

    labels = df[label_col]

    # Debugging: Check filtered features
    print("\nFiltered Feature Data Types:")
    print(features.dtypes)

    return features, labels

