
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler

def log_reg_model(X_train, y_train):
    """Train a logistic regression model with hyperparameter tuning."""
    param_grid = {
        "scale__with_mean": [True, False],
        "scale__with_std": [True, False],
        "poly__degree": [1, 2],
        "clf__C": [0.05, 0.1, 0.5],
        "clf__penalty": ["l2"],
        "clf__solver": ["saga"]
    }
    pipe_model = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures()),
        ("clf", LogisticRegression(max_iter=10000, random_state=42))
    ])
    grid_search = GridSearchCV(pipe_model, param_grid, cv=10, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def random_forest_model(X_train, y_train):
    """Train a random forest model with hyperparameter tuning."""
    param_grid = {
        "scale__with_mean": [True, False],
        "scale__with_std": [True, False],
        "rf__n_estimators": [10, 50, 100],
        "rf__max_depth": [4, 5, 6],
        "rf__max_features": [4, 5, 6]
    }
    pipe_model = Pipeline([
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(random_state=42))
    ])
    grid_search = GridSearchCV(pipe_model, param_grid, cv=10, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

