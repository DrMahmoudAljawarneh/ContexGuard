from core.logger import logger

class MLPolicy:
    def __init__(self):
        self.name = "Machine Learning Adaptive Policy"
        self.risk_threshold = 0.75 # Dynamic threshold

    def evaluate(self, subject: dict, resource: str, action: str, context: dict) -> str:
        risk_score = context.get('ai_risk_score')
        
        if risk_score is None:
            logger.warning("ML Policy invoked but no AI Risk Score found in context. Defaulting to Deny.")
            return "Deny"
            
        logger.debug(f"Evaluating ML Policy. Calculated Risk: {risk_score}, Threshold: {self.risk_threshold}")
        
        # Access is granted if the risk score is below the threshold
        if risk_score < self.risk_threshold:
            return "Permit"
        else:
            return "Deny"
