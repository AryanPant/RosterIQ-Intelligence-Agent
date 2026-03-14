import os

import yaml


def load_config(path):

    if os.path.isabs(path):
        config_path = path
    else:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(base_dir, path)

    with open(config_path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
