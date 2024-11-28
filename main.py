from data_preprocessing import load_data, preprocess_data, split_features_labels
from model_training import train_log_reg, train_random_forest, train_knn, train_ada_boost, train_stacking

def main():
    # Load and preprocess data
    file_path = "data/HighthroughputDFTcalculations.csv"
    df = load_data(file_path)
    df = preprocess_data(df)
    features, labels = split_features_labels(df)

    # Train individual models
    print("\nTraining Logistic Regression...")
    log_reg = train_log_reg(features, labels)
    print(f"Logistic Regression Test Accuracy: {log_reg[3]}")

    print("\nTraining Random Forest...")
    rf = train_random_forest(features, labels)
    print(f"Random Forest Test Accuracy: {rf[3]}")

    print("\nTraining K-Nearest Neighbors...")
    knn = train_knn(features, labels)
    print(f"KNN Test Accuracy: {knn[3]}")

    print("\nTraining AdaBoost...")
    ada = train_ada_boost(features, labels)
    print(f"AdaBoost Test Accuracy: {ada[3]}")

    # Train stacking classifier
    print("\nTraining Stacking Classifier...")
    stacking_clf = train_stacking(features, labels)
    print(f"Stacking Classifier Test Accuracy: {stacking_clf[2]}")

if __name__ == "__main__":
    main()
