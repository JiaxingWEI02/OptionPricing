import json
import os
import torch
import numpy as np


def load_params(file_path):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(current_dir, 'opt_params.json')
    with open(file_path, 'r') as f:
        params = json.load(f)

    for key in params:
        if key != "Type" and isinstance(params[key], str):
            params[key] = eval(params[key], {"torch":torch}, {"np.float32":np.float32, "np.float64":np.float64})
    return params

params = load_params('option_pricing\opt_params.json')

