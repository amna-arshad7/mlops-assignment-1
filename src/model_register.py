import mlflow
import mlflow.sklearn
import joblib

print("🔗 Setting MLflow tracking URI...")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

best_model_name = "SVM"
print(f"📂 Loading best model from models/{best_model_name}.pkl")

model = joblib.load(f"models/{best_model_name}.pkl")

print("🚀 Starting MLflow run for registration...")
with mlflow.start_run(run_name=f"{best_model_name}_registration") as run:
    mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="model",
        registered_model_name=best_model_name
    )

print(f"✅ Best model '{best_model_name}' registered in MLflow Model Registry!")
