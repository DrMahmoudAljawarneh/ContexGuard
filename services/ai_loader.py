import os
import joblib
import numpy as np
from core.logger import logger

class AILoader:
    def __init__(self, model_dir="plugins/ai_models/"):
        self.model_dir = model_dir
        self.model = None
        self.is_loaded = False
        self._load_default_model()

    def _load_default_model(self):
        model_path = os.path.join(self.model_dir, "rf_risk_model.joblib")
        if os.path.exists(model_path):
            try:
                self.model = joblib.load(model_path)
                self.is_loaded = True
                logger.info("Random Forest risk model loaded successfully.")
            except Exception as e:
                logger.error(f"Failed to load AI model: {e}")
        else:
            logger.warning("No pre-trained model found. AI risk scoring is disabled.")

    def is_model_loaded(self) -> bool:
        return self.is_loaded

    def predict_risk(self, context: dict) -> float:
        if not self.is_loaded:
            return 0.0
            
        # Transform context into feature array (simplified for 4 features)
        # Features: [device_score, location_trust, time_anomaly, network_anomaly]
        try:
            device_score = context.get('device', {}).get('security_score', 0.5)
            loc_trust = 1.0 if context.get('location', {}).get('type') == 'office' else 0.0
            net_anomaly = context.get('network', {}).get('anomaly_score', 0.0)
            
            features = np.array([[device_score, loc_trust, 0.0, net_anomaly]])
            risk_prob = self.model.predict_proba(features)[0][1] # Probability of 'Deny/Malicious'
            return float(risk_prob)
        except Exception as e:
            logger.error(f"Risk prediction failed: {e}")
            return 0.5
