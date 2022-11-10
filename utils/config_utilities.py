import json
from typing import Dict, Any

from utils.paths import CONFIG_PATH


def load_config() -> Dict[str, Any]:
    with open(CONFIG_PATH, 'r') as f:
        config = json.load(f)
    return config
