"""I/O helpers for Parquet, JSON, YAML, and CSV operations."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Union
import pandas as pd
import yaml

def save_json(data: Union[Dict, List], filepath: Union[str, Path], indent: int = 2) -> None:
    """Saves dictionary or list to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)

def load_json(filepath: Union[str, Path]) -> Any:
    """Loads JSON file safely."""
    with open(filepath, "r", encoding="utf-8") as f:
        return json.load(f)

def save_jsonl(records: List[Dict[str, Any]], filepath: Union[str, Path]) -> None:
    """Saves list of dicts to JSON Lines file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, default=str) + "\n")

def load_jsonl(filepath: Union[str, Path]) -> List[Dict[str, Any]]:
    """Loads JSON Lines file into list of dicts."""
    records = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records

def load_yaml(filepath: Union[str, Path]) -> Dict[str, Any]:
    """Loads YAML file safely."""
    with open(filepath, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

def save_yaml(data: Dict[str, Any], filepath: Union[str, Path]) -> None:
    """Saves YAML file safely."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False)
