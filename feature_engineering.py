import pandas as pd

def encode_target_variable(df, target_col, encoding_map):
    """Encode target variable using a given mapping."""
    df[target_col] = df[target_col].replace(encoding_map)
    return df

def select_numeric_features(df):
    """Select numeric features from the dataframe."""
    return df.select_dtypes(include=['float64', 'int64'])

def drop_columns(df, cols_to_drop):
    """Drop specified columns from the dataframe."""
    return df.drop(cols_to_drop, axis=1)