import os
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_training_data(num_samples=100000):
    print(f"Generating {num_samples} synthetic context samples...")
    X = []
    y = []
    
    for _ in range(num_samples):
        # 15% probability of generating an anomalous/attack sample
        is_attack = np.random.rand() < 0.15 
        
        if is_attack:
            device_score = np.random.uniform(0.0, 0.4)
            loc_trust = 0.0  # e.g., remote_untrusted
            time_anomaly = np.random.uniform(0.6, 1.0)
            net_anomaly = np.random.uniform(0.7, 1.0)
            label = 1  # 1 = Deny/Malicious
        else:
            device_score = np.random.uniform(0.7, 1.0)
            loc_trust = 1.0  # e.g., office
            time_anomaly = np.random.uniform(0.0, 0.3)
            net_anomaly = np.random.uniform(0.0, 0.2)
            label = 0  # 0 = Permit/Legitimate
            
        # Add some random noise to make the model robust
        device_score = np.clip(device_score + np.random.normal(0, 0.05), 0, 1)
        net_anomaly = np.clip(net_anomaly + np.random.normal(0, 0.05), 0, 1)
        
        # Feature array: [device_score, location_trust, time_anomaly, network_anomaly]
        X.append([device_score, loc_trust, time_anomaly, net_anomaly])
        y.append(label)
        
    return np.array(X), np.array(y)

def train_and_save_model():
    # 1. Generate Data
    X, y = generate_training_data(100000)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 2. Train Model
    print("Training Random Forest Classifier (n_estimators=100, max_depth=10)...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_train, y_train)
    
    # 3. Evaluate Model
    print("\nModel Evaluation on Test Set:")
    y_pred = clf.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Permit (0)', 'Deny (1)']))
    
    # 4. Save Model to Plugins Directory
    target_dir = "../plugins/ai_models"
    os.makedirs(target_dir, exist_ok=True)
    model_path = os.path.join(target_dir, "rf_risk_model.joblib")
    
    joblib.dump(clf, model_path)
    print(f"\nSuccess! Model saved to: {model_path}")

if __name__ == "__main__":
    train_and_save_model()
