import json

VERSION = "1.0"

def get_config():
    with open(f'configs/transformer_config_{VERSION}.json', 'r') as file:
        json_data = json.load(file)
    return json_data