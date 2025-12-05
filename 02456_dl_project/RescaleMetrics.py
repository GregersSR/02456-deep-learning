import json
import pandas as pd 
from pathlib import Path



def scale_json_history_and_save(old_path, new_path, divide_by=20):
    with open(old_path, "r") as f:
        data = json.load(f)

    if "history" not in data:
        print(f"No 'history' in {old_path}, skipping.")
        return

    scaled_history = {}
    for key, values in data["history"].items():
        if isinstance(values, list) and all(isinstance(x, (int, float)) for x in values):
            scaled_history[key] = [x / divide_by for x in values]
        else:
            scaled_history[key] = values

    data["history"] = scaled_history

    new_path.parent.mkdir(parents=True, exist_ok=True)

    with open(new_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Saved scaled file in {new_path.name}")


MODELS_TXT = Path('models.txt')

OLD_DIR = Path('results_filtered')
NEW_DIR = Path('results_filtered/results_filtered/newscaled')

NEW_DIR.mkdir(parents=True, exist_ok=True)

with open(MODELS_TXT, "r") as f:
    model_names = [line.strip() for line in f if line.strip()][:15]

for model_name in model_names:
    matches = list(OLD_DIR.glob(f"{model_name}_results-*.json"))

    if not matches:
        print(f"No matching JSON found for model '{model_name}'")
        continue

    matches.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    old_json = matches[0]

    new_json = NEW_DIR / old_json.name

    scale_json_history_and_save(old_json, new_json)