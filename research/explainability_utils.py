import joblib
import numpy as np
import matplotlib.pyplot as plt

def generate_feature_importance_report(model_path="../plugins/ai_models/rf_risk_model.joblib"):
    try:
        model = joblib.load(model_path)
        features = ["Device Security Score", "Location Trust", "Time Anomaly", "Network Anomaly"]
        importances = model.feature_importances_
        indices = np.argsort(importances)
        
        plt.figure(figsize=(10, 6))
        plt.title("ContexGuard AI Policy - Feature Importances")
        plt.barh(range(len(indices)), importances[indices], color='b', align='center')
        plt.yticks(range(len(indices)), [features[i] for i in indices])
        plt.xlabel('Relative Importance')
        plt.tight_layout()
        plt.savefig("feature_importance_report.png")
        print("Explainability report generated: feature_importance_report.png")
    except Exception as e:
        print(f"Could not generate report. Ensure the model exists. Error: {e}")

if __name__ == "__main__":
    generate_feature_importance_report()
