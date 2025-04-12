import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load Dataset
wine = load_wine()
X = wine.data
Y = wine.target

# Train-Test-Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=42)


# Defining Params for RF Model
max_depth = 10
n_estimators = 10  

# This function checks if an experiment named "My_New_Experiment" exists.
# If it doesn’t, it will create one
mlflow.set_experiment('Mlflows-Study')
mlflow.sklearn.autolog()
with mlflow.start_run():
    clf = RandomForestClassifier(max_depth=max_depth, n_estimators=n_estimators)
    clf.fit(X_train, Y_train)
   
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(Y_test, y_pred)  

    # Logging metrics and params using mlflow
    mlflow.log_metric('accuracy', accuracy)  
    mlflow.log_params({'max_depth': max_depth, 'n_estimators': n_estimators})

    
    # Creating Confusion metrics plot
    cm = confusion_matrix(Y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Metrics')

    # Save the fig locally
    plt.savefig("confusion-metrics.png")

    # Log artifacts using mlflow
    mlflow.log_artifact('confusion-metrics.png')
    mlflow.log_artifact(__file__) # Logging this python file
    
    print(accuracy)