"""Configuration loader utility."""
import os
from pathlib import Path
from typing import Any, Dict
import yaml

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """Loads configuration from YAML file and merges environment overrides."""
    full_path = ROOT_DIR / config_path if not os.path.isabs(config_path) else Path(config_path)
    if not full_path.exists():
        raise FileNotFoundError(f"Configuration file not found at: {full_path}")
    
    with open(full_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
        
    return config

def get_root_dir() -> Path:
    """Returns project root directory."""
    return ROOT_DIR
