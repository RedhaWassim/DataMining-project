from soil_fertility.logger import logging
from pathlib import Path
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
        logging.info('Base path not found')
        raise FileNotFoundError("Base path not found")
    