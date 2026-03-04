import paho.mqtt.client as mqtt
import json
from core.logger import logger

class MQTTService:
    def __init__(self, node_id, broker_url="localhost", port=1883):
        self.node_id = node_id
        self.client = mqtt.Client(client_id=f"contexguard_{node_id}")
        self.broker_url = broker_url
        self.port = port
        self.decision_topic = f"contexguard/node/{self.node_id}/decision"

    def connect(self):
        self.client.connect(self.broker_url, self.port, 60)
        self.client.loop_start()
        logger.info(f"MQTT Client connected to {self.broker_url}")

    def publish_decision(self, payload: dict):
        self.client.publish(self.decision_topic, json.dumps(payload), qos=1)
        logger.info(f"Published to {self.decision_topic}")
