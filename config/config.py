# config/config.py
import json
import os

class Config:
    def __init__(self, config_path=None):
        if config_path is None:
            base_path = os.path.dirname(os.path.abspath(__file__))
            config_path = os.path.join(base_path, "config.json")
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Konfigurationsdatei {config_path} nicht gefunden.")
        with open(config_path, "r") as f:
            self.data = json.load(f)

    def get(self, key, default=None):
        return self.data.get(key, default)

# Erstelle eine globale Instanz ohne expliziten Pfad
config = Config()
