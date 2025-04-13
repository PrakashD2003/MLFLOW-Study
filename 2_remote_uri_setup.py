# Import necessary libraries for ML, visualization, and logging
import mlflow                      # MLflow for experiment tracking
import mlflow.sklearn             # MLflow support for scikit-learn
from sklearn.datasets import load_wine  # Built-in wine dataset
from sklearn.ensemble import RandomForestClassifier  # ML model
from sklearn.model_selection import train_test_split  # For splitting data
from sklearn.metrics import accuracy_score, confusion_matrix  # Evaluation metrics
import pandas as pd
import matplotlib.pyplot as plt   # For plotting
import seaborn as sns             # For better visualization of confusion matrix
import dagshub                    # To integrate MLflow with DagsHub

# Initialize Dagshub tracking with your repository (Enable MLflow logging on DagsHub)
dagshub.init(repo_owner='PrakashD2003', repo_name='MLFLOW-Study', mlflow=True)

# Set MLflow tracking URI to point to your Dagshub MLflow endpoint
mlflow.set_tracking_uri('https://dagshub.com/PrakashD2003/MLFLOW-Study.mlflow')

# Load the wine classification dataset
wine = load_wine()
X = wine.data      # Features (chemical analysis results)
Y = wine.target    # Labels (wine class)

# Split dataset into training and testing subsets (80% train, 20% test)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)

# Define hyperparameters for the Random Forest model
max_depth = 10        # Maximum depth of decision trees
n_estimators = 10     # Number of trees in the forest

# Set up (or create) an MLflow experiment named "Mlflows-Study"
mlflow.set_experiment('Mlflows-Study')

# Enable automatic logging of parameters, metrics, model, and more with MLflow
mlflow.sklearn.autolog()

# Start an MLflow run to log all the following actions
with mlflow.start_run():
    
    # Initialize and train the Random Forest classifier
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    clf.fit(X_train, Y_train)
   
    # Predict labels for the test set
    y_pred = clf.predict(X_test)

    # Compute model accuracy
    accuracy = accuracy_score(Y_test, y_pred)  

    # Log accuracy metric manually (even though autolog also handles this)
    mlflow.log_metric('accuracy', accuracy)  

    # Log model hyperparameters manually (optional with autolog)
    mlflow.log_params({'max_depth': max_depth, 'n_estimators': n_estimators})

    # Generate confusion matrix for evaluation
    cm = confusion_matrix(Y_test, y_pred)

    # Plot the confusion matrix
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=wine.target_names,
                yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Metrics')

    # Save the plot locally as an image
    plt.savefig("confusion-metrics.png")

    # Log the confusion matrix plot to MLflow/Dagshub
    mlflow.log_artifact('confusion-metrics.png')

    # Log this Python script file as an artifact (helps with reproducibility)
    mlflow.log_artifact(__file__)

    # Add custom tags for better experiment tracking
    mlflow.set_tags({
        'Author': 'Prakash',
        'Project': 'Wine Classification'
    })

    # Log the trained model to MLflow manually (autolog also logs it automatically)
    mlflow.sklearn.log_model(clf, 'Random-Forest-Model')

    # Print final accuracy
    print(accuracy)
