import time
import numpy as np
import sys
import os

# Ensure Python finds the framework root
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from services.policy_engine import PolicyEngine
from services.ai_loader import AILoader
from plugins.policies.ml_policy import MLPolicy
from research.synthetic_traffic_simulator import SyntheticTrafficSimulator

# --- Baseline Policies for Comparison ---
class RBACPolicy:
    def evaluate(self, subject, resource, action, context):
        return "Permit" if subject.get("role") in ["admin", "operator"] else "Deny"

class CABACPolicy:
    def evaluate(self, subject, resource, action, context):
        role = subject.get("role")
        loc = context.get("location", {}).get("type")
        dev_score = context.get("device", {}).get("security_score", 0)
        
        if role in ["admin", "operator"] and loc == "office" and dev_score > 0.8:
            return "Permit"
        return "Deny"

def run_latency_benchmark(num_requests=5000):
    print(f"--- Running Latency Benchmark ({num_requests} requests per policy) ---")
    
    # Initialize components
    ai_loader = AILoader() # Automatically loads the .joblib model
    simulator = SyntheticTrafficSimulator()
    
    # Define policies to test
    policies = {
        "RBAC (Baseline)": RBACPolicy(),
        "CABAC (Complex)": CABACPolicy(),
        "ML-Policy (Random Forest)": MLPolicy()
    }
    
    # Generate test context vectors
    test_contexts = [simulator.generate_context_vector(is_attack=(i % 5 == 0)) for i in range(num_requests)]
    subject = {"id": "user_123", "role": "operator"}
    
    # Create dummy MQTT client for PDP
    class DummyMQTT:
        def publish_decision(self, payload): pass
    pdp = PolicyEngine(ai_loader=ai_loader, mqtt_client=DummyMQTT())
    
    # Benchmark each policy
    for name, policy in policies.items():
        pdp.set_policy(policy)
        latencies = []
        
        for context in test_contexts:
            start_time = time.perf_counter()
            pdp.evaluate_request(subject, "critical_actuator", "WRITE", context)
            end_time = time.perf_counter()
            latencies.append((end_time - start_time) * 1000) # Convert to ms
            
        avg_latency = np.mean(latencies)
        std_dev = np.std(latencies)
        # Approximate throughput based on single-thread sequential evaluation
        throughput = 1000 / avg_latency if avg_latency > 0 else 0 
        
        print(f"{name.ljust(30)} | Avg Latency: {avg_latency:.2f} ms | Std Dev: {std_dev:.2f} ms | Est. Peak Throughput: ~{int(throughput)} RPS")

if __name__ == "__main__":
    run_latency_benchmark()