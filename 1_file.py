# Import necessary libraries
import mlflow                  # Core MLflow library
import mlflow.sklearn         # MLflow support for scikit-learn models
from sklearn.datasets import load_wine  # Load sample wine dataset
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest
from sklearn.model_selection import train_test_split  # For splitting dataset
from sklearn.metrics import accuracy_score, confusion_matrix  # Metrics
import pandas as pd
import matplotlib.pyplot as plt  # For plotting
import seaborn as sns            # For visualization of the confusion matrix

# Load the wine dataset
wine = load_wine()
X = wine.data      # Feature matrix
Y = wine.target    # Target labels (0, 1, 2 - different wine classes)

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Define model hyperparameters
max_depth = 10          # Maximum depth of each tree in the forest
n_estimators = 10       # Number of trees in the Random Forest

# Set or create an MLflow experiment (group of runs)
mlflow.set_experiment('Mlflows-Study')

# Start an MLflow run — this is where everything gets tracked
with mlflow.start_run():
    # Initialize and train the Random Forest model
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    clf.fit(X_train, Y_train)
   
    # Make predictions on the test data
    y_pred = clf.predict(X_test)
    
    # Calculate accuracy
    accuracy = accuracy_score(Y_test, y_pred)  

    # Log accuracy metric to MLflow
    mlflow.log_metric('accuracy', accuracy)  
    
    # Log model hyperparameters to MLflow
    mlflow.log_params({
        'max_depth': max_depth,
        'n_estimators': n_estimators
    })

    # Generate confusion matrix
    cm = confusion_matrix(Y_test, y_pred)

    # Plot the confusion matrix using seaborn heatmap
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Metrics')

    # Save the plot as an image locally
    plt.savefig("confusion-metrics.png")

    # Log the saved confusion matrix image as an artifact
    mlflow.log_artifact('confusion-metrics.png')

    # Log this Python script file itself as an artifact
    mlflow.log_artifact(__file__)

    # Add custom tags to the run (optional but useful for filtering/searching)
    mlflow.set_tags({
        'Author': 'Prakash',
        'Project': 'Wine Classification'
    })

    # Save the trained model to MLflow so it can be loaded later
    mlflow.sklearn.log_model(clf, 'Random-Forest-Model')
    
    # Print final accuracy
    print(accuracy)
