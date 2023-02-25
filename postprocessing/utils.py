import os, csv
from boiles.objective.simulation2d import Simulation2D
from stable_baselines3 import PPO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from .config import *
from IPython.display import clear_output
import pandas as pd
from ..envs.utils import get_scale_coefs
from ..envs.env_base import AlpacaEnv

plt.rc("font", family="Times New Roman")
plt.rcParams["mathtext.fontset"] = "stix"


def log(text: str, file: str):
    text += "\n"
    with open(file, "a+") as file:
        file.write(text)

def read_from_csv(file_name: str) -> np.array:
    """
    a helper function to read data from csv file
    :param file_name: csv file name
    :return: numpy array
    """
    with open(file_name, 'r') as csvfile:
        reader = csv.reader(csvfile)
        data_rows = np.array([row for row in reader])

    data = []
    for row in data_rows:
        if len(row) != 0:
            data.append(row)

    return np.array(data)

def safe_create_folder(path: str):
    """
    as the name implies ...
    """
    if os.path.exists(path):
        os.system(f"rm -rf {path}")
    os.makedirs(path)



def test_on_highresolution(
    name: str,
    model_path: str,
    envs: list,
):
    folder = f"/media/yiqi/Elements/RL/August/{name}/"
    for env in envs:
        print("- Build environment ... ")
        env = env() # create an instance
        model = PPO.load(model_path, env=env, device="cuda")
        dones = False
        obs = env.reset(verbose=False, evaluate=True)
        
        print(f"- Run {env.__class__.__name__} ... ")
        while not dones:
            action, _states = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
            if env.obj.is_crashed:
                print("- simulation crashed!")
                return
            
        print(f"- Copy {env.__class__.__name__} ... ")
        env_folder = os.path.join(folder, env.__class__.__name__)
        safe_create_folder(env_folder)
        safe_create_folder(os.path.join(env_folder, "runtime_data"))
        # copy to hard drive and release the space, since SSD doesn't have enough storage
        os.system(f"cp -r ./runtime_data/{env.inputfile}_* {env_folder}/runtime_data")
        os.system(f"rm -rf runtime_data")  
        
        print("- Save the actions ... ")
        actions = pd.DataFrame(env.debug.action_trajectory, columns=["q", "C", "eta"])
        actions.to_csv(f"{env_folder}/actions.csv")
        print("-------- END ----------")


def _plot_field(
    file: str,
    state: str,
    types: list, # contour or imshow
    config: dict,
    slice_: slice = None,
    shape: tuple = None
) -> None:
    """
    config for 
    imshow:  extent, vmin, vmax, cmap, xticks, yticks
    contour: extent, colors, linewidths, levels, xticks, yticks
    """
    sim = Simulation2D(file=file, shape=shape)
    # print(sim.result_exit, shape, slice_)
    if slice_ is None:
        data = sim.result[state]
    else:
        data = sim.result[state][:, slice_]
    # print(data)
    if "imshow" in types:
        plt.imshow(
            data, 
            origin="lower", 
            extent=config.get("extent", None), 
            vmin=config.get("vmin", None), 
            vmax=config.get("vmax", None),
            cmap=config.get("cmap", "viridis"),
            alpha=config.get("alpha", 1.0)
        )
        if config.get("colorbar", False):
            plt.colorbar()
    if "contour" in types:
        # print("levels:", config.get("levels", 10))
        plt.contour(
            data, 
            origin="lower", 
            extent=config.get("extent", None), 
            colors=config.get("colors", "black"), 
            linewidths=config.get("linewidths", 0.3), 
            levels=config.get("levels", 10), 
            alpha=config.get("contour_alpha", 1)
        )
    plt.xticks(config.get("xticks", [0, 0.5, 1]), fontsize=12)
    plt.yticks(config.get("yticks", [0, 0.5, 1]), fontsize=12)
    


def plot_test_results(
    name: str,
    time: str,
    state: str,
    config: dict,
    types: list = ["imshow"],
    shape: tuple = None,
    dpi: int = 100,
    path: str = None
):
    schemes = [r"WENO5", r"TENO5", r"TENO5SP", r"TENO5RL"]
    files = data_path(name, time)
    shape = shape_config(name)
    data_slice = data_slices(name)
    # if data_slice is None:
        # data_slice = slice(None, None)
    # print(data_slice)
    if name.lower() == "rti":
        plt.figure(figsize=(6, 4 * len(files.keys())), dpi=dpi)
    elif name.lower() == "viscous_shock":
        plt.figure(figsize=(16, 2 * len(files.keys())), dpi=dpi)
    else:
        plt.figure(figsize=(16, 4 * len(files.keys())), dpi=dpi)

    i = 1
    for key, file in files.items():
        for data_file in file:
            plt.subplot(len(files.keys()), 4, i)
            try:
                if data_slice is not None:
                    _plot_field(data_file, state, types, config, data_slice[key], shape[key])
                else:
                    _plot_field(data_file, state, types, config, shape=shape[key])
            except:
                pass
            plt.title(schemes[(i-1) % 4], fontsize=10)
            i += 1
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=400)


def plot_separate_test_results(
    name: str,
    time: str,
    state: str,
    config: dict,
    types: list = ["imshow"],
    shape: tuple = None,
    dpi: int = 100,
    path: str = None,
    cells: int = 64
):
    schemes = [r"WENO5", r"TENO5", r"TENO5SP", r"TENO5RL"]
    files = data_path(name, time)[cells]
    shape = shape_config(name)
    if name.lower() == "rti":
        plt.figure(figsize=(6, 4), dpi=dpi)
    else:
        plt.figure(figsize=(16, 4), dpi=dpi)

    i = 1
    for data_file in files:
        plt.subplot(1, 4, i)
        # try:
        _plot_field(data_file, state, types, config, shape=shape[cells])
        # except:
            # pass
        plt.title(schemes[(i-1) % 4], fontsize=16)
        i += 1
    plt.tight_layout()
    if path is not None:
        plt.savefig(path, dpi=400)

def dispersion_to_baseline(base: Simulation2D, sim: Simulation2D, subdomain: list):
    """
    computer anti-diffusion increase to baseline
    """
    if subdomain is not None:
        row = slice(None, subdomain[0])
        col = slice(None, subdomain[1])
        base_trunc = base.result["highorder_dissipation_rate"][row, col]
        sim_eff = sim.result["effective_dissipation_rate"][row, col]
    else:
        base_trunc = base.result["highorder_dissipation_rate"]
        sim_eff = sim.result["effective_dissipation_rate"]
    trunc = base_trunc - sim_eff
    return np.where(trunc < 0, trunc, 0).sum()


def compute_improvement(
    name: str,
    cells: int,
    test_schemes: list = ["teno5", "teno5lin", "teno5rl"],
    subdomain: list = None
):
    
    if name == "doublemach" and subdomain is None:
        raise Exception("Double mach reflection should have a subdomain!")
    baseline = "weno5"
    envs_names = envs_name(name)
    end_time, timestep_size = time_config(name)["end_time"], time_config(name)["timestep_size"]
    shape = shape_config(name)[cells]
    baseline_folder = "baseline"
    for scheme in test_schemes: 
        results = []
        timesteps = np.arange(timestep_size, end_time + 1e-6, timestep_size)
        for t in timesteps:
            print(f"{scheme} to {baseline}:")
            print(f"{round(t / end_time * 100, 2)}% ... ")
            clear_output(wait=True)
            time = format(t, ".3f")
            base = Simulation2D(f"/media/yiqi/Elements/RL/{baseline_folder}/{name}/{name}_{cells}_{baseline}/domain/data_{time}*.h5", shape=shape)
            if subdomain is not None:
                row = slice(None, subdomain[0])
                col = slice(None, subdomain[1])
                ke_base = base.result["kinetic_energy"][row, col].sum()
            else:
                ke_base = base.result["kinetic_energy"].sum()
            dissip_base, disper_base, sum_base, _ = base.truncation_errors(subdomain=subdomain)
            if scheme != "teno5rl":
                test = Simulation2D(f"/media/yiqi/Elements/RL/{baseline_folder}/{name}/{name}_{cells}_{scheme}/domain/data_{time}*.h5", shape=shape)
            else:
                test = Simulation2D(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/runtime_data/{name}_{cells}_{time}/domain/data_{time}*.h5", shape=shape)
            if subdomain is not None:
                row = slice(None, subdomain[0])
                col = slice(None, subdomain[1])
                ke_test = test.result["kinetic_energy"][row, col].sum()
            else:
                ke_test = test.result["kinetic_energy"].sum()
            ke_reward = ke_test / ke_base - 1
            disper_test = dispersion_to_baseline(base, test, subdomain)
            disper_imp = disper_test / disper_base - 1
            row = [float(time), ke_base, ke_test, ke_reward, disper_base, disper_test, disper_imp]
            results.append(row)

        results = pd.DataFrame(results, columns=[
            "t",
            f"ke_{baseline}", 
            "ke_test", 
            "ke_reward", 
            f"disper_{baseline}", 
            "disper_test", 
            "disper_imp", 
        ])
        results.to_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/{scheme}_to_{baseline}_info.csv", index=False)

from scipy.signal import savgol_filter
def plot_improvement(
    name: str,
    cells: int
):
    baseline = "weno5"
    envs_names = envs_name(name)

    teno5_to_weno5 = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5_to_{baseline}_info.csv")
    teno5lin_to_weno5 = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5lin_to_{baseline}_info.csv")
    teno5rl_to_weno5 = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5rl_to_{baseline}_info.csv")

    # df = pd.concat([teno5_to_weno5["t"], teno5_to_weno5["ke_reward"]*100, teno5lin_to_weno5["ke_reward"]*100, teno5rl_to_weno5["ke_reward"]*100], axis=1)
    # df.columns = ["t", "TENO5", "TENO5SP", "TENO5RL"]
    # ax = df.plot(x="t", xlim=(0, time_config(name)["end_time"]), figsize=(5, 4), grid=True)
    # ax.set_title(r"KE Increse, $" + str(cells) + r"\times " + str(cells) + r"$")
    # ax.set_xlabel(r"$t$")
    # ax.set_ylabel(r"$\%$")
    # ax.legend(fontsize=12)
    # plt.tight_layout()
    w = 5
    window_length = 21
    p_order = 2
    timestep = teno5_to_weno5["t"].to_numpy()
    teno5 = teno5_to_weno5["ke_reward"].to_numpy() * 100
    teno5lin = teno5lin_to_weno5["ke_reward"].to_numpy() * 100
    teno5rl = teno5rl_to_weno5["ke_reward"].to_numpy() * 100
    # teno5_m = np.convolve(teno5, np.ones(w), mode="same") / w
    teno5_sav = savgol_filter(teno5, window_length, p_order)
    teno5lin_sav = savgol_filter(teno5lin, window_length, p_order)
    teno5rl_sav = savgol_filter(teno5rl, window_length, p_order)
    # plt.plot(teno5)
    # plt.plot(teno5_m)
    plt.figure(dpi=100)
    plt.plot(timestep, teno5, color="black", linestyle="-", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5lin, color="red", linestyle="--", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5rl, color="blue", linestyle="-.", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5_sav, linestyle="-", color="black", linewidth=0.8)
    plt.plot(timestep, teno5lin_sav, linestyle="--", color="red", linewidth=0.8)
    plt.plot(timestep, teno5rl_sav, linestyle="-.", color="blue", linewidth=0.8)
    plt.show()
    # plt.savefig(f"{envs_name[cells]}_KE.jpg", dpi=400)

    # df = pd.concat([teno5_to_weno5["t"], teno5_to_weno5["disper_imp"]*100, teno5lin_to_weno5["disper_imp"]*100, teno5rl_to_weno5["disper_imp"]*100], axis=1)
    # df.columns = ["t", "TENO5", "TENO5SP", "TENO5RL"]
    # ax = df.plot(x="t", xlim=(0, time_config(name)["end_time"]), figsize=(5, 4), fontsize=12, grid=True)
    # ax.set_title(r"Anti-Diffusion Increse, $" + str(cells) + r"\times " + str(cells) + r"$", fontsize=16)
    # ax.set_xlabel(r"$t$", fontsize=12)
    # ax.set_ylabel(r"$\%$", fontsize=12)
    # ax.legend(fontsize=12)
    # plt.tight_layout()

    w = 5
    window_length = 21
    p_order = 2
    timestep = teno5_to_weno5["t"].to_numpy()
    teno5 = teno5_to_weno5["disper_imp"].to_numpy() * 100
    teno5lin = teno5lin_to_weno5["disper_imp"].to_numpy() * 100
    teno5rl = teno5rl_to_weno5["disper_imp"].to_numpy() * 100
    # teno5_m = np.convolve(teno5, np.ones(w), mode="same") / w
    teno5_sav = savgol_filter(teno5, window_length, p_order)
    teno5lin_sav = savgol_filter(teno5lin, window_length, p_order)
    teno5rl_sav = savgol_filter(teno5rl, window_length, p_order)
    # plt.plot(teno5)
    # plt.plot(teno5_m)
    plt.figure(dpi=100)
    plt.plot(timestep, teno5, color="black", linestyle="-", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5lin, color="red", linestyle="--", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5rl, color="blue", linestyle="-.", alpha=0.2, linewidth=0.5)
    plt.plot(timestep, teno5_sav, linestyle="-", color="black", linewidth=0.8)
    plt.plot(timestep, teno5lin_sav, linestyle="--", color="red", linewidth=0.8)
    plt.plot(timestep, teno5rl_sav, linestyle="-.", color="blue", linewidth=0.8)
    # plt.plot(teno5lin_s)
    # plt.plot(teno5rl_s)
    # plt.savefig(f"{envs_name[cells]}_AntiDiffusion.jpg", dpi=400)


def plot_actions(
    name: str,
    cells: int
):
    titles = {64: r"$64\times 64$", 128: r"$128\times 128$", 256: r"$256\times 256$"}
    envs_names = envs_name(name)
    timesteps = np.arange(0.01, time_config(name)["end_time"] + 1e-6, 0.01)
    actions = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/actions.csv", index_col=0)
    actions["t"] = timesteps
    subplots = actions.plot(subplots=True, x="t", legend=False, figsize=(8, 8), fontsize=12, grid=True, xlim=(0, time_config(name)["end_time"]))

    subplots[0].set_ylim(0.8, 6.2)
    subplots[0].set_ylabel("$q$", fontsize=12)
    subplots[0].set_title(titles[cells], fontsize=16)

    subplots[1].set_ylim(-1, 21)
    subplots[1].set_ylabel("$C$", fontsize=12)

    subplots[2].set_ylim(0.19, 0.41)
    subplots[2].set_xlabel("$t$", fontsize=12)
    subplots[2].set_ylabel("$\eta$", fontsize=12)

    plt.tight_layout()

def plot_training_ave_reward(name: str, path: str = "", n: int = 20, slice: slice = None, bias: float = 0, scale: float = 1) -> None:
    # end_time, timestep_size = time_config(name)["end_time"], time_config(name)["timestep_size"]
    files = reward_files(name)
    dfs = [pd.read_csv(file) for file in files ]
    rewards = np.array([df["rollout/ep_rew_mean"] for df in dfs])
    
    means = rewards.mean(axis=0) * scale - bias
    index = np.unravel_index(np.argsort(rewards, axis=None)[::-1][:n], rewards.shape)
    print(np.concatenate((((index[0] + 1) * 100).reshape(-1, 1), ((index[1] + 1) / 2 - 1).reshape(-1, 1), rewards[index].reshape(-1, 1)), axis=1))
    maximum = rewards.max(axis=0) * scale - bias 
    minimum = rewards.min(axis=0) * scale - bias
    steps = dfs[0]["time/total_timesteps"].to_numpy()
    plt.figure(figsize=(5, 4))
    plt.plot(steps[slice], means[slice], linewidth=0.8, c="black")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(4,4))
    plt.fill_between(steps[slice], maximum[slice], minimum[slice], alpha=0.2, color="black", edgecolor="none")
    plt.xlabel("Time steps")
    plt.ylabel("Total reward ($r$)")
    plt.xlim(0, steps[slice][-1])
    plt.tight_layout()
    plt.grid()
    if path != "":
        plt.savefig(path, dpi=400)

def plot_deterministic_ave_reward(name: str, path: str = "", n: int = 20, slice: slice = None, bias: float = 0) -> None:
    # end_time, timestep_size = time_config(name)["end_time"], time_config(name)["timestep_size"]
    files = deterministic_reward_files(name)
    dfs = [pd.read_csv(file) for file in files ]
    rewards = np.array([df["rewards"] for df in dfs])
    
    means = rewards.mean(axis=0)
    index = np.unravel_index(np.argsort(rewards, axis=None)[::-1][:n], rewards.shape)
    print(np.concatenate((((index[0] + 1) * 100).reshape(-1, 1), index[1].reshape(-1, 1), rewards[index].reshape(-1, 1)), axis=1))
    maximum = rewards.max(axis=0)
    minimum = rewards.min(axis=0)
    steps = dfs[0]["iteration"]
    plt.figure(figsize=(5, 4))
    plt.plot(steps[slice], means[slice], linewidth=0.8, c="black")
    plt.ticklabel_format(style='sci', axis='x', scilimits=(2,2))
    plt.fill_between(steps[slice], maximum[slice], minimum[slice], alpha=0.2, color="black", edgecolor="none")
    plt.xlabel("Time steps")
    plt.ylabel("Total reward ($r$)")
    # plt.xlim(0, 199)
    plt.tight_layout()
    plt.grid()
    if path != "":
        plt.savefig(path, dpi=400)



def test_deterministic_reward(
    model_path: str,
    env: AlpacaEnv  # initialized environment
):
    """
    Test training case and highresolution case with trained model
    :param name: case name
    :param model_path: trained model
    :param envs: environment for testing
    """
    model = PPO.load(model_path, env=env, device="cuda")
    dones = False
    obs = env.reset(evaluate=True, verbose=False)
    
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)

    return env.cumulative_reward

def compute_reward(
    name: str,
    cells: int,
    test_schemes: list = ["teno5", "teno5lin", "teno5rl"]
):
    
    baseline = "weno5"
    envs_names = envs_name(name)
    end_time, timestep_size = time_config(name)["end_time"], time_config(name)["timestep_size"]
    shape = shape_config(name)[cells]
    
    for scheme in test_schemes: 
        if cells == 64:
            scale_coef = get_scale_coefs(
                f"scheme_rl/data/{name}_teno5_to_weno5.csv", 
                end_time, 
                timestep_size, 
                absolute=True if scheme == "teno5" else True
            )
        else:
            scale_coef = get_scale_coefs(f"scheme_rl/data/{name}_teno5_to_weno5_{cells}.csv", end_time, timestep_size)
        results = []
        timesteps = np.arange(0.01, end_time + 1e-6, timestep_size)
        for t in timesteps:
            print(f"{scheme} to {baseline}:")
            print(f"{round(t / end_time * 100, 2)}% ... ")
            clear_output(wait=True)
            time = format(t, ".3f")
            base = Simulation2D(f"/media/yiqi/Elements/RL/baseline/{name}/{name}_{cells}_{baseline}/domain/data_{time}*.h5", shape=shape)
            if scheme != "teno5rl":
                test = Simulation2D(f"/media/yiqi/Elements/RL/baseline/{name}/{name}_{cells}_{scheme}/domain/data_{time}*.h5", shape=shape)
            else:
                test = Simulation2D(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/runtime_data/{name}_{cells}_{time}/domain/data_{time}*.h5", shape=shape)
            
            
            ke_base = base.result["kinetic_energy"].sum()
            ke_test = test.result["kinetic_energy"].sum()
            ke_reward = ke_test / ke_base - 1


            dissip_base, disper_base, sum_base, _ = base.truncation_errors()   
            disper_test = dispersion_to_baseline(base, test)
            disper_imp = abs(disper_test) / abs(disper_base) - 1
            total_reward = ke_reward - np.max(disper_imp, 0) * scale_coef[time]
            row = [float(time), total_reward]
            results.append(row)

        results = pd.DataFrame(results, columns=[
            "t",
            "reward"
        ])
        results.to_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/{scheme}_to_{baseline}_reward.csv", index=False)


def plot_discounted_reward(
    name: str,
    cells: int,
    path: str = ""
):
    envs_names = envs_name(name)
    end_time, timestep_size = time_config(name)["end_time"], time_config(name)["timestep_size"]
    timesteps = np.arange(0.01, end_time + 1e-6, timestep_size)
    def discounted(df):
        rews = df["reward"]
        n = len(rews)
        rtgs = np.zeros_like(rews)
        for i in reversed(range(n)):
            rtgs[i] = rews[i] + (rtgs[i+1] if i+1 < n else 0)
        return rtgs
    df = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5_to_weno5_reward.csv")
    plt.figure(figsize=(5, 4))
    plt.plot(timesteps, discounted(df), label="TENO5")
    df = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5lin_to_weno5_reward.csv")
    plt.plot(timesteps, discounted(df), label="TENO5SP")
    df = pd.read_csv(f"/media/yiqi/Elements/RL/August/{name}/{envs_names[cells]}/teno5rl_to_weno5_reward.csv")
    plt.plot(timesteps, discounted(df), label="TENO5RL")
    plt.xlim(0, end_time)
    plt.legend()
    plt.xlabel("$t$")
    plt.ylabel("Reward to go $V^\pi(s_t)$")
    plt.grid()
    plt.tight_layout()
    if path != "":
        plt.savefig(path, dpi=400)