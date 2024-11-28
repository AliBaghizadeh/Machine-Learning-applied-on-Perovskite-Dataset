import pytest
import pandas as pd
from data_preprocessing import clean_data
from feature_engineering import encode_target_variable

def test_clean_data():
    """Test the clean_data function for missing values."""
    raw_data = pd.DataFrame({
        'Formation energy [eV/atom]': ['1.2', 'NaN', '0.5'],
        'Lowest distortion': ['cubic', 'orthorhombic', '-']
    })
    cleaned_data = clean_data(raw_data)
    assert cleaned_data.isnull().sum().sum() == 0

def test_encode_target_variable():
    """Test encoding of target variable."""
    df = pd.DataFrame({'Lowest distortion': ['cubic', 'orthorhombic', '-']})
    encoding_map = {"cubic": 1, "orthorhombic": 2, "-": 3}
    df = encode_target_variable(df, 'Lowest distortion', encoding_map)
    assert all(df['Lowest distortion'] == [1, 2, 3])