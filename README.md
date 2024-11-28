# Machine Learning for Perovskite Materials

## Objective
This project applies machine learning techniques to predict the properties of perovskite materials (ABO3) using a dataset of high-throughput DFT calculations. It leverages multiple machine learning models, including Logistic Regression, Random Forests, and a Stacking Classifier, to handle the multi-class classification problem.

## Dataset
The dataset used is from Antoine Emery's work:

**Title:** High-throughput DFT calculations of formation energy, stability, and oxygen vacancy formation energy of ABO3 perovskites.
**DOI:** 10.6084/m9.figshare.5334142.v1
**Source:** matminer
The dataset contains features such as chemical formula, valence states, formation energy, stability, magnetic moment, and band gap.

 ## Project Structure

 machine-learning-materials-science/    
│
├── data/                        # Store datasets    
│   ├── raw/                     # Raw datasets    
│   ├── processed/               # Processed datasets    
│             
├── notebooks/                   # Jupyter notebooks for exploration and experiments    
│   ├── data_exploration.ipynb   # EDA and cleaning    
│   ├── model_training.ipynb     # Training various models    
│   └── model_evaluation.ipynb   # Evaluation and visualization   
│           
├── src/                         # Source code for reusable scripts    
│   ├── data_preprocessing.py    # Data loading and cleaning functions    
│   ├── feature_engineering.py   # Feature extraction and engineering functions    
│   ├── model_training.py        # Model training and evaluation pipelines   
│   ├── visualization.py         # Visualization utilities    
│            
├── tests/                       # Unit tests   
│   └── test_model.py            # Test cases for model pipelines    
│            
├── results/                     # Store results like confusion matrices, evaluation scores    
├── requirements.txt             # Python dependencies   
├── main.py                      # Entry point script   
├── README.md                    # Project description and instructions   
└── LICENSE                      # License (e.g., MIT)     

## Installation Instructions

### 1. Clone the Repository
Open a terminal and clone the repository:

git clone https://github.com/username/machine-learning-materials-science.git    
cd machine-learning-materials-science    

