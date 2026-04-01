import json
import os
from typing import Dict

class ConfigReader:
    """Utility for reading model configuration and hyperparameters"""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = base_dir
        self.model_config_path = os.path.join(base_dir, "models", "model_config.json")
        self.config_module_path = os.path.join(base_dir, "utils", "config.py")
    
    def get_model_config(self) -> Dict:
        """Read model_config.json if it exists"""
        try:
            if os.path.exists(self.model_config_path):
                with open(self.model_config_path, 'r') as f:
                    return json.load(f)
            return {}
        except Exception as e:
            print(f"Error reading model config: {e}")
            return {}
    
    def get_hyperparameters(self) -> Dict:
        """
        Extract hyperparameters from utils/config.py
        This is a simplified version - ideally would parse the Python file
        """
        try:
            # Import the config module dynamically
            import sys
            import importlib.util
            
            spec = importlib.util.spec_from_file_location("config", self.config_module_path)
            config_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config_module)
            
            # Extract relevant attributes
            params = {}
            for attr in dir(config_module):
                if not attr.startswith('_') and attr.isupper():
                    value = getattr(config_module, attr)
                    # Only include simple types
                    if isinstance(value, (int, float, str, bool, list)):
                        params[attr] = value
            
            return params
        except Exception as e:
            print(f"Error reading config.py: {e}")
            return {}
    
    def get_model_files_info(self) -> list:
        """Get information about saved model files"""
        try:
            models_dir = os.path.join(self.base_dir, "models")
            model_files = []
            
            for filename in os.listdir(models_dir):
                if filename.endswith('.pth'):
                    filepath = os.path.join(models_dir, filename)
                    stat = os.stat(filepath)
                    
                    model_files.append({
                        'filename': filename,
                        'size_mb': stat.st_size / (1024 * 1024),
                        'modified': stat.st_mtime
                    })
            
            return sorted(model_files, key=lambda x: x['modified'], reverse=True)
        except Exception as e:
            print(f"Error reading model files: {e}")
            return []
