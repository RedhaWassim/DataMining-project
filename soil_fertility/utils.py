from pathlib import Path

from soil_fertility.logger import logging
import os
import pickle
import yaml


def retreive_base_path():
    current_path = Path(__file__).resolve()
    base_path = None

    while current_path:
        if (current_path / "README.md").exists():
            base_path = current_path
            break
        current_path = current_path.parent

    if base_path:
        return base_path
    else:
        logging.info("Base path not found")
        raise FileNotFoundError("Base path not found")


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e


def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            obj = pickle.load(file_obj)

        return obj

    except Exception as e:
        logging.error(f"Exception occured {e}")
        raise e


def read_params_from_yaml(yaml_path):
    with open(yaml_path, "r") as yaml_file:
        params = yaml.safe_load(yaml_file)
    return params
