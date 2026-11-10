import json
import os
from datetime import datetime
from typing import Dict


def save_record(record: Dict[str, object], directory: str = "records") -> str:
    os.makedirs(directory, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(directory, f"{timestamp}.json")
    temporary_path = f"{path}.tmp"
    with open(temporary_path, "w", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2)
        handle.write("\n")
    os.replace(temporary_path, path)
    return path
