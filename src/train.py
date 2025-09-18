import os
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    ConfusionMatrixDisplay
)



# Load dataset
data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.2, random_state=42
)

# Models to train
models = {
    "LogisticRegression": LogisticRegression(max_iter=200),
    "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
    "SVM": SVC(kernel="linear", probability=True, random_state=42)
}

# Train and log with MLflow
for name, model in models.items():
    with mlflow.start_run(run_name=name):
        mlflow.set_tag("model_name", name)

        # Train
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Evaluate
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")

        # Log important parameters
        mlflow.log_param("model", name)
        if name == "RandomForest":
            mlflow.log_param("n_estimators", model.n_estimators)
        if name == "SVM":
            mlflow.log_param("kernel", model.kernel)

        # Log metrics
        mlflow.log_metrics({
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        })

        # Save model locally
        model_path = f"models/{name}.pkl"
        joblib.dump(model, model_path)

        # Log model to MLflow
        mlflow.sklearn.log_model(model, artifact_path="model")

        # Confusion matrix plot
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)
        disp.plot(cmap="Blues", values_format="d")
        plot_path = f"results/{name}_confusion_matrix.png"
        plt.savefig(plot_path)
        plt.close()

        # Log confusion matrix artifact
        mlflow.log_artifact(plot_path, artifact_path="plots")

print("✅ Training complete. Metrics & artifacts logged to MLflow, models saved in /models.")