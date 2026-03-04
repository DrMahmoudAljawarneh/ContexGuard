import os
import importlib.util
from core.logger import logger

class SensorManager:
    def __init__(self, plugin_dir="plugins/sensors/"):
        self.plugin_dir = plugin_dir
        self.active_sensors = {}

    def discover_and_load(self):
        for filename in os.listdir(self.plugin_dir):
            if filename.endswith(".py") and filename != "__init__.py":
                filepath = os.path.join(self.plugin_dir, filename)
                module_name = filename[:-3]
                
                # Dynamic loading using importlib
                spec = importlib.util.spec_from_file_location(module_name, filepath)
                module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(module)
                
                for attribute_name in dir(module):
                    attribute = getattr(module, attribute_name)
                    if isinstance(attribute, type) and attribute_name.endswith("Sensor"):
                        sensor_instance = attribute(plugin_id=module_name)
                        self.active_sensors[module_name] = sensor_instance
                        logger.info(f"Loaded sensor plugin: {module_name}")

    def get_aggregated_context(self) -> dict:
        context_vector = {}
        for name, sensor in self.active_sensors.items():
            context_vector[name] = sensor.get_context_data()
        return context_vector
