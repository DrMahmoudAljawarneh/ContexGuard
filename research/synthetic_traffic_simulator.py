import time
import random
import numpy as np

class SyntheticTrafficSimulator:
    def __init__(self, target_url="http://localhost:8000/api/evaluate"):
        self.target_url = target_url
        self.actions = ["READ", "WRITE", "EXECUTE"]
        self.resources = ["critical_actuator", "patient_records", "hvac_control"]
        self.roles = ["admin", "operator", "guest", "maintenance"]

    def generate_context_vector(self, is_attack=False):
        if is_attack:
            return {
                "device": {"security_score": random.uniform(0.0, 0.3), "status": "compromised"},
                "location": {"type": "remote_untrusted"},
                "network": {"anomaly_score": random.uniform(0.7, 1.0)}
            }
        else:
            return {
                "device": {"security_score": random.uniform(0.8, 1.0), "status": "secure"},
                "location": {"type": "office"},
                "network": {"anomaly_score": random.uniform(0.0, 0.2)}
            }

    def simulate_load(self, duration_seconds=60, lam=50): # lam = requests per second
        print(f"Starting simulation: lambda={lam} RPS for {duration_seconds} seconds.")
        end_time = time.time() + duration_seconds
        
        requests_sent = 0
        while time.time() < end_time:
            # Poisson arrival time
            sleep_time = np.random.exponential(1.0 / lam)
            time.sleep(sleep_time)
            
            is_attack = random.random() < 0.15 # 15% malicious traffic
            payload = {
                "subject": {"id": f"user_{random.randint(100, 999)}", "role": random.choice(self.roles)},
                "resource": random.choice(self.resources),
                "action": random.choice(self.actions),
                "context": self.generate_context_vector(is_attack)
            }
            
            # In a real test, you'd send this to the API endpoint
            requests_sent += 1
            
        print(f"Simulation complete. Total requests generated: {requests_sent}")

if __name__ == "__main__":
    simulator = SyntheticTrafficSimulator()
    simulator.simulate_load(duration_seconds=10, lam=100)
