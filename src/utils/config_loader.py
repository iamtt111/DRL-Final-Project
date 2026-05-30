import os
import yaml

def load_config(config_path: str = None) -> dict:
    """載入設定檔，預設載入 configs/env_default.yaml"""
    if config_path is None:
        # 尋找專案根目錄下的 configs/env_default.yaml
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        config_path = os.path.join(base_dir, "configs", "env_default.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}
