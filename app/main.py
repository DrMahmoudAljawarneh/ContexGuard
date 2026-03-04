import sys
import os

# Ensure Python finds the framework root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.logger import logger
from services.sensor_manager import SensorManager
from services.mqtt_client import MQTTService
from services.policy_engine import PolicyEngine
from services.ai_loader import AILoader
from plugins.policies.ml_policy import MLPolicy

if __name__ == "__main__":
    node_id = os.environ.get("NODE_ID", "local_test_node")
    
    # 1. Initialize Core Services
    mqtt_svc = MQTTService(node_id=node_id)
    
    sensor_mgr = SensorManager()
    sensor_mgr.discover_and_load()
    
    # 2. Initialize the AI components
    ai_loader = AILoader()
    ml_policy = MLPolicy()
    
    # 3. Spin up the Policy Engine (PDP)
    pdp = PolicyEngine(ai_loader=ai_loader, mqtt_client=mqtt_svc)
    pdp.set_policy(ml_policy)
    
    # 4. Process a sample request with a SECURE context
    aggregated_context = sensor_mgr.get_aggregated_context()
    
    # Injecting simulated contextual data for the ML model
    aggregated_context['device'] = {"security_score": 0.9} # High security score
    aggregated_context['location'] = {"type": "office"}    # Trusted location
    aggregated_context['network'] = {"anomaly_score": 0.1} # Low network anomaly
    
    subject = {"id": "user_123", "role": "operator"}
    
    print("\n--- ContexGuard PDP Evaluation ---")
    decision = pdp.evaluate_request(subject, "critical_actuator", "WRITE", aggregated_context)
    print(f"Final Outcome: {decision}\n")