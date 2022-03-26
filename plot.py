import numpy as np
from boiles.objective.simulation2d import Simulation2D
from .envs.env_base import AlpacaEnv
import matplotlib.pyplot as plt


def timesteps(env: AlpacaEnv):
    return np.arange(env.timestep_size, env.end_time+1e-6, env.timestep_size)


def runtime_rate(env: AlpacaEnv):
    rate = []
    for end_time in timesteps(env):
        end_time = format(end_time, ".3f")
        baseline_case = Simulation2D(
            results_folder=env.baseline_data_loc,
            result_filename=f"data_{end_time}*.h5"
        )
        _, _, baseline_rate = baseline_case.truncation_errors()
        simulation_case = Simulation2D(
            results_folder=f"runtime_data/{env.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}*.h5"
        )
        _, _, simulation_rate = simulation_case.truncation_errors()
        rate.append([baseline_rate, simulation_rate])
    return np.array(rate)


def plot(env: AlpacaEnv):
    rate = runtime_rate(env)
    plt.plot(figsize=(5, 5), dpi=200)
    plt.plot(rate[:, 0], c="black", linewidth=0.6, label="$TENO5$")
    plt.plot(rate[:, 1], c="blue", linewidth=0.6, label="$TENO5_{RL}$")
    plt.legend()
    plt.show()


def draw_subcontour(
        value: np.array,
        figsize=None,
        dpi=100,
        range=None,
        levels=21,
        linewidths=0.5,
        colors="black",
        cmap=None,
        extent=None,
        title=None,
        plot_shape=None,
        save=None,
        ticks_fontsize=10,
        title_fontsize=10,
        grid=False,
        colorbar=False,
        colorbar_fontsize=None,
        xticks=None,
        yticks=None
):
    value = np.array(value)
#     plt.rcParams["font.family"] = "Times New Roman"
    if range is None:
        vmin = vmax = None
    else:
        vmin = range[0]
        vmax = range[1]
    if cmap is not None:
        colors = None

    if value.ndim == 2:
        print("Only one figure.")
        figsize = (5, 4) if figsize is None else figsize
        plt.figure(figsize=figsize, dpi=dpi)
        plt.contour(
            value, origin="lower", vmin=vmin, vmax=vmax, levels=levels, linewidths=linewidths,
            colors=colors, cmap=cmap, extent=extent
        )
        if title is not None:
            plt.title(title)
        plt.xticks(xticks, fontsize=ticks_fontsize)
        plt.yticks(yticks, fontsize=ticks_fontsize)
        if colorbar:
            cbar = plt.colorbar()
            if colorbar_fontsize is not None:
                cbar.ax.tick_params(labelsize=colorbar_fontsize)
        if grid:
            plt.grid()
    elif value.ndim == 3:
        print("Plot subplots")
        if plot_shape is None:
            ncols = 2
            nrows = np.ceil(value.shape[0] / ncols)
        else:
            nrows, ncols = plot_shape
        figsize = (5*ncols, 4*nrows) if figsize is None else figsize
        plt.figure(figsize=figsize, dpi=dpi)
        for i, data in enumerate(value):
            plt.subplot(nrows, ncols, i+1)
            print(vmin, vmax)
            plt.contour(
                data, origin="lower", vmin=vmin, vmax=vmax, levels=np.linspace(vmin, vmax, levels),
                linewidths=linewidths, colors=colors, cmap=cmap, extent=extent
            )
            if title is not None:
                if len(title) == len(value):
                    plt.title(title[i], fontsize=title_fontsize)
            plt.xticks(xticks, fontsize=ticks_fontsize)
            plt.yticks(yticks, fontsize=ticks_fontsize)
            if colorbar:
                cbar = plt.colorbar()
                if colorbar_fontsize is not None:
                    cbar.ax.tick_params(labelsize=colorbar_fontsize)
            if grid:
                plt.grid()
    if save is not None:
        plt.savefig(save)
    plt.show()
# ke_dict, dissipation_dict, dispersion_dict, abs_numerical_dict, rel_numerical_dict = {}, {}, {}, {}, {}
#
# for i, folder in enumerate(folder_list):
#     ke_dict[i], dissipation_dict[i], dispersion_dict[i], rel_numerical_dict[i], abs_numerical_dict[
#         i] = [], [], [], [], []
#
#     for t in timesteps:
#         end_time = format(t, ".3f")
#         if folder == "runtime_data":
#             freeshear = Simulation2D(results_folder=f"{folder}/full_implosion_64_{end_time}/domain",
#                                       result_filename=f"data_{end_time}*.h5")
#         else:
#             freeshear = Simulation2D(results_folder=folder, result_filename=f"data_{end_time}*.h5")
#         numerical_rate = freeshear.result["numerical_dissipation_rate"]
#         dissipation = numerical_rate[numerical_rate > 0]
#         dispersion = numerical_rate[numerical_rate < 0]
#         dissipation_dict[i].append(dissipation.sum())
#         dispersion_dict[i].append(dispersion.sum())
#
#         rel_numerical_dict[i].append(numerical_rate.sum())
#         abs_numerical_dict[i].append(abs(numerical_rate).sum())
#
#         ke = compute_ke_array(freeshear)
#         ke_dict[i].append(ke.sum())

#     plt.plot(timesteps, ke_dict[q], label=str(q), linewidth=0.6)