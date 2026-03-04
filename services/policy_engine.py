from core.logger import logger

class PolicyEngine:
    def __init__(self, ai_loader, mqtt_client):
        self.ai_loader = ai_loader
        self.mqtt_client = mqtt_client
        self.active_policy = None

    def set_policy(self, policy):
        self.active_policy = policy

    def evaluate_request(self, subject: dict, resource: str, action: str, context: dict) -> str:
        if not self.active_policy:
            return "Indeterminate"

        risk_score = 0
        if self.ai_loader.is_model_loaded():
            risk_score = self.ai_loader.predict_risk(context)
            context['ai_risk_score'] = risk_score

        decision = self.active_policy.evaluate(subject, resource, action, context)
        
        # Broadcast decision via MQTT
        decision_payload = {"subject": subject['id'], "resource": resource, "decision": decision}
        self.mqtt_client.publish_decision(decision_payload)
        
        logger.info(f"PDP rendered decision: {decision} for {subject['id']}")
        return decision
