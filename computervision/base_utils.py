import json
import os

def load_json_file(path) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return None

def dump_json_file(jsondict, path):
    with open(path, 'w') as f:
        json.dump(jsondict, f, sort_keys=True, indent=4)    

