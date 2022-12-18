import pandas as pd
import numpy as np
import os

def normalize(value, bounds):
    normalized = (value - bounds[0]) / (bounds[1] - bounds[0])
    return normalized


def get_scale_coefs(file: str, end_time: float, timestep_size: float, absolute: bool = True):
    assert os.path.exists(file), "Scale coefs file not found!"
    data = pd.read_csv(file)
    coef = data["ke_reward"] / data["disper_imp"]
    coef_dict = {}
    for i, t in enumerate(np.arange(0.01, end_time + 1e-6, timestep_size)):
        time = format(t, ".3f")
        # coef_dict[time] = coef[i] if coef[i] > 0 else 1
        coef_dict[time] = abs(coef[i]) if absolute else coef[i]
    return coef_dict