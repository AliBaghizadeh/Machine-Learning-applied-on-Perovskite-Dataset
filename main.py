from data_preprocessing import load_data, clean_data, split_features_labels
from model_training import log_reg_model, random_forest_model
from visualization import plot_correlation_heatmap
from feature_engineering import encode_target_variable, select_numeric_features

def main():
    """Main function to execute the ML workflow."""
    # Load data
    df = load_data('data/HighthroughputDFTcalculations.csv')
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    features, labels = split_features_labels(df, label_col='LowestDist')
    
    # Train models
    log_reg = log_reg_model(features, labels)
    rf_model = random_forest_model(features, labels)
    
    # Visualizations
    plot_correlation_heatmap(df)

if __name__ == '__main__':
    main()