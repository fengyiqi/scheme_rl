import numpy as np
from boiles.objective.simulation2d import Simulation2D
import matplotlib.pyplot as plt

def plot_states(env, end_times, states, shape=None):
    cols = len(end_times)
    y_size = 4
    for state in states:
        plt.figure(figsize=(cols * y_size, y_size), dpi=100)
        for i, end_time in enumerate(end_times):
            try:
                obj = Simulation2D(f"runtime_data/{env.inputfile}_{end_time}/domain/data_{end_time}*.h5", shape) 
                data = obj.result[state]
                plt.subplot(1, cols, i+1)
                plt.imshow(data, origin="lower")
                plt.title(f"{state} ({end_time[:-1]}s)")
            except:
                continue
        plt.show()

def plot_reward_and_quality(value):

    value_array = np.array(value)
    plt.figure(figsize=(8, 3), dpi=100)
    plt.subplot(1, 2, 1)
    plt.plot(value_array[:, 0], color="black", linewidth=0.8)
    plt.title("reward")
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.plot(value_array[:, 1], color="black", linewidth=0.8)
    plt.title("quality")
    plt.grid()
    plt.show()

def plot_action_trajectory(action_trajectory):

    action_traj = np.array(action_trajectory)
    label = [r"$q$", r"$C$", r"$\eta$"]
    a_range = [(1-0.5, 6+0.5), (1-1, 20+1), (0.2-0.1, 0.4+0.1)]
    plt.figure(figsize=(len(label)*4, 3))
    for i in range(len(label)):
        plt.subplot(1, len(label), i+1)
        plt.plot(action_traj[:, i])
        plt.ylim(a_range[i][0], a_range[i][1])
        plt.grid()
        plt.title(label[i])

    plt.show()