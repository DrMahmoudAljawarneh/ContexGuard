import random

class AmbientLightSensor:
    def __init__(self, plugin_id, config=None):
        self.plugin_id = plugin_id
        self.name = "Ambient Light Sensor"
        self.min_lux = config.get("min_lux", 0) if config else 0
        self.max_lux = config.get("max_lux", 1000) if config else 1000

    def initialize(self, config):
        self.min_lux = config.get("min_lux", self.min_lux)
        self.max_lux = config.get("max_lux", self.max_lux)
        print(f"Ambient Light Sensor {self.plugin_id} initialized.")

    def get_context_data(self):
        current_lux = random.randint(self.min_lux, self.max_lux)
        return {"type": "ambient_light", "value": current_lux, "unit": "lux"}
