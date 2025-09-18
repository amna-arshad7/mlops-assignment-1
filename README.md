# MLOps Assignment 1 – GitHub Basics & MLflow Tracking

## 📌 Problem Statement
The goal of this assignment is to:
1. Understand GitHub basics by creating a repository and managing project structure.
2. Train and compare multiple machine learning models on a dataset.
3. Track experiments using **MLflow** (parameters, metrics, artifacts).
4. Register the best performing model in the MLflow Model Registry.
5. Document the entire workflow and submit via GitHub.

---

## 📊 Dataset Description
We used the **Iris dataset** from `scikit-learn`.  
- **Features:** 4 (Sepal Length, Sepal Width, Petal Length, Petal Width)  
- **Target Classes:** 3 (Setosa, Versicolor, Virginica)  
- **Samples:** 150  

---

## 🤖 Model Selection & Comparison
We trained and evaluated 3 models:  
1. **Logistic Regression**  
2. **Random Forest Classifier**  
3. **Support Vector Machine (SVM)**  

### Evaluation Metrics
- Accuracy  
- Precision (Weighted)  
- Recall (Weighted)  
- F1-score (Weighted)  

The metrics were logged in MLflow.  
👉 In our case, **SVM performed the best** with highest accuracy and F1 score.

---

## 📈 MLflow Experiment Tracking
We used **MLflow** to:
- Log parameters and metrics.
- Save and visualize confusion matrices.
- Track all runs for comparison.

---

## 🏆 Model Registration
The **best performing model (SVM)** was registered in the **MLflow Model Registry**.  


---

## ⚙️ How to Run the Code

### 1. Clone Repository
```bash
git clone https://github.com/<your-username>/mlops-assignment-1.git
cd mlops-assignment-1
