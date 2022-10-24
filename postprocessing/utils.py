import os
from boiles.objective.simulation2d import Simulation2D
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import data_path

def safe_create_folder(path: str):
    """
    as the name implies ...
    """
    if os.path.exists(path):
        os.system(f"rm -rf {path}")
    os.makedirs(path)


def dispersion_to_baseline(base: Simulation2D, sim: Simulation2D):
    """
    computer anti-diffusion increase to baseline
    """
    base_trunc = base.result["highorder_dissipation_rate"]
    sim_eff = sim.result["effective_dissipation_rate"]
    trunc = base_trunc - sim_eff
    return np.where(trunc < 0, trunc, 0).sum()


def test_on_highresolution(
    name: str,
    model_path: str,
    envs: list
):
    folder = f"/media/yiqi/Elements/RL/August/{name}/"
    for env in envs:
        print("- Build environment ... ")
        env = env() # create an instance
        model = PPO.load(model_path, env=env, device="cuda")
        dones = False
        obs = env.reset(evaluate=True)
        
        print(f"- Run {env.__class__.__name__} ... ")
        while not dones:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            
        print(f"- Copy {env.__class__.__name__} ... ")
        safe_create_folder(folder + env.__class__.__name__)
        safe_create_folder(folder + env.__class__.__name__ + "/runtime_data")
        # copy to hardrive and release the space, since SSD doesn't have enough storage
        os.system(f"cp -r ./runtime_data/{env.inputfile}_* {folder}/{env.__class__.__name__}/runtime_data")
        os.system(f"rm -rf ./runtime_data/{env.inputfile}_*")  
        
        print("- Save the actions ... ")
        actions = pd.DataFrame(env.debug.action_trajectory, columns=["q", "C", "eta"])
        actions.to_csv(f"{folder}/{env.__class__.__name__}/actions.csv")


def _plot_field(
    file: str,
    state: str,
    types: list, # contour or imshow
    config: dict,
    shape: tuple = None
) -> None:
    """
    config for 
    imshow:  extent, vmin, vmax, cmap, xticks, yticks
    contour: extent, colors, linewidths, levels, xticks, yticks
    """
    sim = Simulation2D(file=file, shape=shape)
    data = sim.result[state]
    if "imshow" in types:
        plt.imshow(
            data, 
            origin="lower", 
            extent=config.get("extent", None), 
            vmin=config.get("vmin", None), 
            vmax=config.get("vmax", None),
            cmap=config.get("cmap", "viridis")
        )
    if "contour" in types:
        plt.contour(
            data, 
            origin="lower", 
            extent=config.get("extent", None), 
            colors=config.get("colors", "black"), 
            linewidths=config.get("linewidths", 0.3), 
            levels=config.get("levels", 10), 
        )
    plt.xticks(config.get("xticks", [0, 0.5, 1]))
    plt.yticks(config.get("yticks", [0, 0.5, 1]))


def plot_test_results(
    name: str,
    time: str,
    state: str,
    config: dict,
    types: list = ["imshow"],
    shape: tuple = None,
    dpi=100
):
    schemes = [r"WENO5", r"TENO5", r"TENO5SP", r"TENO5RL"]
    files = data_path(name, time)
    plt.figure(figsize=(16, 4 * len(files.keys())), dpi=dpi)
    i = 1
    for key, file in files.items():
        for data_file in file:
            plt.subplot(len(files.keys()), 4, i)
            _plot_field(data_file, state, types, config, shape)
            plt.title(schemes[(i-1) % 4])
            i += 1
    plt.tight_layout()