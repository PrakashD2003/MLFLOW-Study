# Import required libraries for modeling, data handling, and tracking
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer
import pandas as pd
import mlflow
import dagshub

# Initialize DAGsHub tracking to connect MLflow with your DAGsHub repo
dagshub.init(repo_owner='PrakashD2003', repo_name='MLFLOW-Study', mlflow=True)

# Set the MLflow tracking URI to DAGsHub's MLflow server
mlflow.set_tracking_uri('https://dagshub.com/PrakashD2003/MLFLOW-Study.mlflow')

# ---------------------------
# DATA PREPARATION
# ---------------------------

# Load the Breast Cancer dataset from sklearn
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)  # Feature data
y = pd.Series(data.target, name='target')                # Target labels

# Split the dataset into training and testing sets (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ---------------------------
# MODEL SETUP
# ---------------------------

# Create a base Random Forest Classifier
rf = RandomForestClassifier(random_state=42)

# Define a grid of hyperparameters to search through
param_grid = {
    'n_estimators': [10, 50, 100],       # Number of trees in the forest
    'max_depth': [None, 10, 20, 30]      # Depth of the trees
}

# Setup GridSearchCV to find the best hyperparameter combination
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,            # 5-fold cross-validation
    n_jobs=-1,       # Use all CPU cores
    verbose=2        # Print detailed logs during fitting
)

# ---------------------------
# MLflow EXPERIMENT
# ---------------------------

# Set the experiment name to organize your runs
mlflow.set_experiment('breast-cancer-rf-hp')

# Start a parent MLflow run to group all child runs
with mlflow.start_run() as parent:

    # Fit the GridSearchCV model (this takes time)
    grid_search.fit(X_train, y_train)

    # Loop through all hyperparameter combinations (i.e., all child runs)
    for i in range(len(grid_search.cv_results_['params'])):

        # Start a nested MLflow run for each hyperparameter config
        with mlflow.start_run(nested=True) as child:
            mlflow.log_params(grid_search.cv_results_["params"][i])  # Log parameters for this run
            mlflow.log_metric("accuracy", grid_search.cv_results_["mean_test_score"][i])  # Log score

    # After all child runs, log the best configuration found
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Log the best parameters and score in the parent run
    mlflow.log_params(best_params)
    mlflow.log_metric("accuracy", best_score)

    # ---------------------------
    # DATA LOGGING
    # ---------------------------

    # Convert training data to a DataFrame and log it
    train_df = X_train.copy()
    train_df['target'] = y_train
    train_df = mlflow.data.from_pandas(train_df)  # Convert to MLflow Data object
    mlflow.log_input(train_df, "training")        # Log as input data

    # Convert test data to a DataFrame and log it
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df = mlflow.data.from_pandas(test_df)
    mlflow.log_input(test_df, "testing")

    # ---------------------------
    # ARTIFACTS AND MODEL LOGGING
    # ---------------------------

    # Log this Python script file for reproducibility
    mlflow.log_artifact(__file__)

    # Log the best model (RandomForest) from GridSearch
    mlflow.sklearn.log_model(grid_search.best_estimator_, "random_forest")

    # Set tags for metadata
    mlflow.set_tag("author", "Prakash Dwivedi") 

    # Print best parameters and score to console
    print(best_params)
    print(best_score)
