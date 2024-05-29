import json


def load_params(file_path):
    with open(file_path, 'r') as f:
        params = json.load(f)
    return params

params = load_params('fast_calc\snowball_params.json')

