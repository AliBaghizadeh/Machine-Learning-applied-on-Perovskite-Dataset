from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures

def train_log_reg(features, labels):
    """Train Logistic Regression with Grid Search."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    pipe_model = Pipeline([
        ("scale", StandardScaler()),
        ("poly", PolynomialFeatures()),
        ("clf", LogisticRegression(max_iter=10000, random_state=42))
    ])

    param_grid = {
        "poly__degree": [1, 2],
        "clf__C": [0.1, 1, 10],
        "clf__penalty": ["l2"],
        "clf__solver": ["saga"]
    }

    grid_search = GridSearchCV(pipe_model, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.score(X_train, y_train), grid_search.score(X_test, y_test)


def train_random_forest(features, labels):
    """Train Random Forest with Grid Search."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [None, 10, 20],
        "max_features": ["sqrt", "log2"]
    }

    grid_search = GridSearchCV(rf, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.score(X_train, y_train), grid_search.score(X_test, y_test)


def train_knn(features, labels):
    """Train K-Nearest Neighbors with Grid Search."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    knn = KNeighborsClassifier()
    param_grid = {
        "n_neighbors": [3, 5, 7],
        "weights": ["uniform", "distance"],
        "p": [1, 2]
    }

    grid_search = GridSearchCV(knn, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.score(X_train, y_train), grid_search.score(X_test, y_test)


def train_ada_boost(features, labels):
    """Train AdaBoost with Grid Search."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    ada = AdaBoostClassifier(random_state=42, algorithm= "SAMME")
    param_grid = {
        "n_estimators": [50, 100, 200],
        "learning_rate": [0.01, 0.1, 1.0]
    }

    grid_search = GridSearchCV(ada, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
    grid_search.fit(X_train, y_train)

    return grid_search.best_estimator_, grid_search.best_params_, grid_search.score(X_train, y_train), grid_search.score(X_test, y_test)


def train_stacking(features, labels):
    """Train Stacking Classifier."""
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.3, random_state=42, stratify=labels)

    base_learners = [
        ("log_reg", train_log_reg(features, labels)[0]),
        ("rf", train_random_forest(features, labels)[0]),
        ("knn", train_knn(features, labels)[0]),
        ("ada", train_ada_boost(features, labels)[0])
    ]

    final_estimator = LogisticRegression(random_state=42)

    stacking_clf = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5, n_jobs=-1)
    stacking_clf.fit(X_train, y_train)

    return stacking_clf, stacking_clf.score(X_train, y_train), stacking_clf.score(X_test, y_test)
