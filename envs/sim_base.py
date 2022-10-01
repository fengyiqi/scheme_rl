import os
import numpy as np

import xml.etree.ElementTree as ET
from boiles.objective.simulation2d import Simulation2D
# from boiles.objective.simulation3d import Simulation3D
from boiles.objective.tgv import TaylorGreenVortex
import torch

SM_PROP = "numerical_dissipation_rate"

paras_range = dict(q=(1, 6), cq=(1, 100), eta=(0.2, 0.4), ct_power=(3, 15))
paras_decimals = dict(q=0, cq=0, eta=4, ct_power=0)
paras_default = dict(q=6, cq=1, eta=0.4, ct_power=5)
paras_index = dict(q=0, cq=1, eta=2, ct_power=3)

action_bound = (-1, 1)


class TimeStepsController:
    def __init__(self, time_span, timestep_size):
        self._time_span = time_span
        self._timestep_size = timestep_size
        self.counter = 1

    def get_time_span(self):
        return self._time_span

    def get_timestep_size(self):
        return self._timestep_size

    def get_end_time_float(self, counter=None):
        if counter is None:
            return self.counter * self._timestep_size
        return counter * self._timestep_size

    def get_end_time_string(self, counter=None):
        if counter is None:
            return format(self.get_end_time_float(self.counter), ".3f")
        return format(self.get_end_time_float(counter), ".3f")

    def get_total_steps(self):
        return int(self._time_span / self._timestep_size)

    def get_restart_time_string(self, end_time, decimal=6):
        # restart file needs the exact file name
        return format(float(end_time) - self._timestep_size, f".{decimal}f")

    def get_time_span_string(self):
        return format(self._time_span, ".3f")



class SchemeParametersWriter:
    def __init__(self, file, parameters):
        self.file = file
        self.parameters = parameters
        self.last_net_action = (0, 0, 0)
        self.net_action = None
        self.real_action = None

    def _rescale(self, value, value_bound, decimal, action_bound=action_bound):
        clipped = np.clip(value, action_bound[0], action_bound[1])
        action_range = action_bound[1] - action_bound[0]
        value_range = value_bound[1] - value_bound[0]
        return round((clipped - action_bound[0]) / action_range * value_range + value_bound[0], decimal)
    def rescale_actions(self, action):
        # print(action)
        real_actions = {}
        for key, value in zip(self.parameters, action):
            real_actions[key] = self._rescale(value, paras_range[key], paras_decimals[key])
        return real_actions

    def configure_scheme_xml(self, action):
        self.net_action = action
        self.real_actions = self.rescale_actions(action)
        tree = ET.ElementTree(file=self.file)
        root = tree.getroot()
        for key, value in paras_default.items():
            if key in self.parameters:
                root[paras_index[key]].text = str(self.real_actions[key])
            else:
                root[paras_index[key]].text = str(value)
        root[paras_index["ct_power"]].text = str(0.1 ** self.real_actions["ct_power"]) \
                                             if "ct_power" in self.parameters else str(1e-5)
        tree.write(self.file)

class AlpacaExecutor:
    def __init__(self, executable, inputfile, cpu_num):
        self.executable = executable
        self.inputfile = inputfile
        self.cpu_num = cpu_num

    def run_alpaca(self, inputfile):
        os.system(f"cd runtime_data; mpiexec -n {self.cpu_num} {self.executable} inputfiles/{inputfile}")


# def _get_states(data_obj, layers=None, normalize_states=True, zero_mean=zero_mean, ave_pool=None):
#     if layers is None:
#         layers = ["density", "velocity", "pressure"]
#     state_matrix = []
#     for state in layers:
#         if state == "velocity":
#             value = np.sqrt(data_obj.result["velocity_x"]**2 + data_obj.result["velocity_y"]**2)
#         # only for shear
#         # elif state == "pressure":
#         #     value = data_obj.result[state] / 100
#         else:
#             value = data_obj.result[state]
#         if ave_pool is not None and value.shape != (64, 64):
#             value = torch.nn.AvgPool2d(ave_pool)(torch.tensor(np.expand_dims(value, axis=0)))[0].numpy()
#         if normalize_states:
#             # only for shear
#             if state != "density" and state != "pressure":
#                 value = normalize(value=value, bounds=(value.min(), value.max()))
#             value = value - 0.5 if zero_mean else value
#         state_matrix.append(value)
#     # print(state_matrix)
#     return state_matrix

class BaselineDataHandler:
    def __init__(
            self,
            timestep_size,
            time_span,
            data_loc,
            layers,
            high_res,
            get_state_func,
            dimension = 2,
            config: dict = None
    ):
        super(BaselineDataHandler, self).__init__()

        if layers is None:
            layers = ["density", "velocity_x", "pressure"]
        self.timestep_size = timestep_size
        self.end_time = time_span
        self.state_data_loc = os.path.join(data_loc, "domain")
        self.restart_data_loc = os.path.join(data_loc, "restart")
        self.layers = layers
        self.high_res = high_res
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)
        self.get_state_func = get_state_func
        self.simulation_reader = Simulation2D if dimension == 2 else TaylorGreenVortex
        self.states = self.get_all_states()
        self.initial_state = self.get_initial_state()
        # self.smoothness = self.get_all_baseline_smoothness_reward()
        # self.truncation = self.get_all_baseline_truncation_reward()
        self.kinetic = self.get_all_baseline_ke_reward()
        self.highorder_dissipation_rate = self.get_all_baseline_highorder_dissipation_rate()
        # self.vorticity = self.get_all_baseline_vor_reward()
        # self.cutoff_tke = self.get_all_baseline_cutoff_tke_reward()
        # self.cutoff_vor = self.get_all_baseline_cutoff_vor_reward()
        self.dispersive = self.get_all_baseline_dispersive_reward()
        # self.implosion_teno5lin_disper_upperbound = self.get_all_baseline_teno5lin_disper_upperbound()

    def get_all_states(self):
        states = {}
        for timestep in np.arange(0, self.end_time, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            states[end_time] = self.get_state_func(data_obj=data_obj, layers=self.layers, ave_pool=self.high_res[1])
            if self.high_res[0]:
                break
        return states

    def get_initial_state(self):
        state = self.states["0.000"]
        if "kinetic_energy" in self.layers:
            index = self.layers.index("kinetic_energy")
            state[index] = np.zeros((64, 64))
        # if "velocity" in self.layers:
        #     index = self.layers.index("velocity")
        #     state[index] = np.zeros((64, 64))
        return state

    def get_all_baseline_smoothness_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            _, rewards[end_time] = data_obj.smoothness(threshold=self.smoothness_threshold, property=SM_PROP)
        return rewards

    def get_all_baseline_truncation_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            _, _, rewards[end_time], _ = data_obj.truncation_errors()
        return rewards

    def get_all_baseline_dispersive_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            _, rewards[end_time], _, _ = data_obj.truncation_errors()
        return rewards

    def get_all_baseline_highorder_dissipation_rate(self):
        if self.high_res[0]:
            return None
        rate = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            rate[end_time] = data_obj.result["highorder_dissipation_rate"]
        return rate

    def get_all_baseline_ke_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            rewards[end_time] = data_obj.result["kinetic_energy"].sum()
        return rewards

    def get_all_baseline_vor_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            rewards[end_time] = data_obj.result["vorticity"].sum()
        return rewards

    def get_all_baseline_cutoff_vor_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            rewards[end_time] = np.where(data_obj.result["vorticity"] > 1, 0, data_obj.result["vorticity"]).sum()
        return rewards

    def get_all_baseline_cutoff_tke_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"))
            rewards[end_time] = data_obj._create_spectrum()[32:, 1].sum()
        return rewards

    def get_all_baseline_teno5lin_disper_upperbound(self):
        if self.high_res[0]:
            return None
        upperbound = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            teno5lin_data_obj = self.simulation_reader(file=f"/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64_teno5lin/domain/data_{end_time}*.h5")
            teno5lin_disper = abs(teno5lin_data_obj.truncation_errors()[1])
            upperbound[end_time] = teno5lin_disper / abs(self.dispersive[end_time])
        return upperbound

class SimulationHandler:
    def __init__(
            self,
            solver: AlpacaExecutor,
            time_controller: TimeStepsController,
            scheme_writer: SchemeParametersWriter,
            baseline_data_obj: BaselineDataHandler,
            linked_reset: bool,
            high_res: tuple,
            config: dict
    ):
        self.solver = solver
        self.time_controller = time_controller
        self.time_span, self.timestep_size = time_controller.get_time_span(), time_controller.get_timestep_size()
        self.scheme_writer = scheme_writer
        self.inputfile = solver.inputfile
        self.baseline_data_obj = baseline_data_obj
        self.is_crashed = False
        self.done = False
        self.layers = baseline_data_obj.layers
        self.linked_reset = linked_reset
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)
        self.high_res = high_res
        self.current_data = None
        self.current_data_obj = None

    def configure_inputfile(self, end_time):
        new_file = self.rename_inputfile(end_time)
        # starting from initial condition, no restart
        if self.time_controller.counter == 1:
            restore_mode = "Off"
            restart_file = "None"
        else:
            restore_mode = "Forced"
            restart_time = self.time_controller.get_restart_time_string(end_time, decimal=3)
            if self.is_crashed and self.linked_reset:

                self.is_crashed = False
                restart_file = os.path.join(
                    self.baseline_data_obj.restart_data_loc,
                    f"restart_{restart_time}.h5"
                )
            else:
                last_time = self.time_controller.get_restart_time_string(end_time, decimal=3)
                restart_time = self.time_controller.get_restart_time_string(end_time)
                restart_file = f"{self.inputfile}_{last_time}/restart/restart_{restart_time}.h5"
        return self.configure_inputfile_xml(
                new_file,
                endtime=end_time,
                restore_mode=restore_mode,
                restart_file=restart_file,
                snapshots_type="Stamps"
            )

    def configure_inputfile_xml(self, file: str, endtime: str, restore_mode: str, restart_file: str, snapshots_type: str):
        # All arguments should be string
        full_path = os.path.join("runtime_data/inputfiles", file)
        tree = ET.ElementTree(file=full_path)
        root = tree.getroot()

        root[4][1].text = endtime  # timeControl -> endTime
        root[6][0][0].text = restore_mode  # restart -> restore -> mode
        root[6][0][1].text = restart_file  # restart -> restore -> fileName
        root[6][1][0].text = snapshots_type  # restart -> snapshots -> type
        root[6][1][3][0].text = endtime  # restart -> snapshots -> stamps -> ts1

        tree.write(full_path)
        return file

    def get_state(self, end_time):
        self.current_data = self.baseline_data_obj.simulation_reader(file=f"runtime_data/{self.inputfile}_{end_time}/domain/data_{end_time}*.h5")
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            return self.baseline_data_obj.initial_state
        else:
            return self.baseline_data_obj.get_state_func(data_obj=self.current_data, layers=self.layers, ave_pool=self.high_res[1])

    def get_tke_reward(self):
        return self.current_data.objective_spectrum()

    def get_smoothness_reward(self, end_time):
        _, reward = self.current_data.smoothness(threshold=self.smoothness_threshold, property=SM_PROP)
        baseline_reward = self.baseline_data_obj.smoothness[end_time]
        improvement = reward / baseline_reward - 1
        return improvement

    def get_truncation_reward(self, end_time):
        _, _, reward, _ = self.current_data.truncation_errors()
        baseline_reward = self.baseline_data_obj.truncation[end_time]
        improvement = 1 - reward / baseline_reward
        return improvement

    def get_dispersive_penalty(self, end_time):
        _, reward, _, _ = self.current_data.truncation_errors()
        baseline_reward = self.baseline_data_obj.dispersive[end_time]
        improvement = 1 - reward / baseline_reward
        return improvement

    def get_dispersive_to_highorder_baseline_penalty(self, end_time):
        eff_rate = self.current_data.result["effective_dissipation_rate"]
        highorder_baseline_rate = self.baseline_data_obj.highorder_dissipation_rate[end_time]
        trunc_to_baseline = highorder_baseline_rate - eff_rate
        dispersion = np.where(trunc_to_baseline < 0, trunc_to_baseline, 0).sum()
        baseline_dispersion = self.baseline_data_obj.dispersive[end_time]
        # baseline_dispersion = 
        # print(dispersion, baseline_dispersion)
        improvement = 1 - abs(dispersion) / abs(baseline_dispersion)
        return improvement

    def get_dispersive_comparison(self, end_time):
        # compare with TENO5LIN
        current_disper = abs(self.current_data.truncation_errors()[1])
        baseline_disper = abs(self.baseline_data_obj.dispersive[end_time])
        ratio = current_disper / baseline_disper
        teno5lin_disper_ratio = self.baseline_data_obj.implosion_teno5lin_disper_upperbound[end_time]
        return ratio - teno5lin_disper_ratio

    def get_ke_reward(self, end_time):
        reward = self.current_data.result["kinetic_energy"].sum()
        baseline_reward = self.baseline_data_obj.kinetic[end_time]
        improvement = reward / baseline_reward - 1
        return improvement

    def get_vor_reward(self, end_time):
        reward = self.current_data.result["vorticity"].sum()
        baseline_reward = self.baseline_data_obj.vorticity[end_time]
        improvement = reward / baseline_reward - 1
        return improvement

    def get_cutoff_tke_reward(self, end_time):
        reward = self.current_data._create_spectrum()[32:, 1].sum()
        baseline_reward = self.baseline_data_obj.cutoff_tke[end_time]
        improvement = 1 - reward / baseline_reward
        return improvement

    def get_cutoff_vor_penalty(self, end_time):
        vor = self.current_data.result["vorticity"]
        vor = np.where(vor > 1, 0, vor).sum()
        baseline_vor = self.baseline_data_obj.cutoff_vor[end_time]
        improvement = 1 - vor / baseline_vor
        return improvement

    def get_action_penalty(self):
        if self.time_controller.get_end_time_float() > self.time_controller.get_timestep_size():
            last_action = np.array(self.scheme_writer.last_net_action)
            current_action = np.array(self.scheme_writer.net_action)
            diff = np.abs(last_action - current_action).mean()
            self.scheme_writer.last_net_action = self.scheme_writer.net_action
        else:
            self.scheme_writer.last_net_action = self.scheme_writer.net_action
            diff = 0
        return diff

    def rename_inputfile(self, end_time):
        old_file = f"runtime_data/inputfiles/{self.inputfile}*"
        new_file = f"runtime_data/inputfiles/{self.inputfile}_{end_time}.xml"
        os.system(f"mv {old_file} {new_file}")
        return f"{self.inputfile}_{end_time}.xml"

    def run(self, inputfile):
        self.solver.run_alpaca(inputfile)

class DebugProfileHandler:
    def __init__(
            self,
            objective: SimulationHandler,
            parameters=("q", "cq", "eta")
    ):
        self.objective = objective
        self.evaluation = False
        self.parameters = parameters
        self.info = ""
        self.action_trajectory = []

    def collect_scheme_paras(self):
        for key in self.objective.scheme_writer.parameters:
            self.collect_info(f"{key:<3}({self.objective.scheme_writer.real_actions[key]:<4}) ")
        self.action_trajectory.append([
            self.objective.scheme_writer.real_actions[key] for key in self.objective.scheme_writer.parameters
        ])

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation

    def collect_info(self, info: str):
        self.info += info

    def flush_info(self):
        print(self.info)
        self.info = ""
        # self.action_trajectory = []
