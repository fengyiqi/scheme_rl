import os
import numpy as np

import xml.etree.ElementTree as ET
from boiles.objective.simulation2d import Simulation2D
# from boiles.objective.simulation3d import Simulation3D
from boiles.objective.tgv import TaylorGreenVortex

SM_PROP = "numerical_dissipation_rate"
paras_range = dict(q=(1, 6), cq=(1, 20), eta=(0.2, 0.4), ct_power=(3, 15))
paras_decimals = dict(q=0, cq=0, eta=4, ct_power=0)
paras_default = dict(q=6, cq=1, eta=0.4, ct_power=5)
paras_index = dict(q=0, cq=1, eta=2, ct_power=3)

action_bound = (-1, 1)


class TimeStepsController:
    """
    Advancing the simulation step by step through ALPACA restart function.
    :param timestep_size: simulation timestep interval
    :param time_span: end time of the simulation
    """
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
    """
    TENO5 parameters writter. 
    :param file: location of TENO5 parameters interface to ALPACA
    :param scheme_parameters: optimizing teno5 parameters
    """
    def __init__(self, file, scheme_parameters):
        self.file = file
        self.scheme_parameters = scheme_parameters
        self.last_net_action = (0, 0, 0)
        self.net_action = None
        self.real_action = None

    def _rescale(self, value, value_bound, decimal, action_bound=action_bound):
        clipped = np.clip(value, action_bound[0], action_bound[1])
        action_range = action_bound[1] - action_bound[0]
        value_range = value_bound[1] - value_bound[0]
        return round((clipped - action_bound[0]) / action_range * value_range + value_bound[0], decimal)
    def rescale_actions(self, action):
        real_actions = {}
        for key, value in zip(self.scheme_parameters, action):
            real_actions[key] = self._rescale(value, paras_range[key], paras_decimals[key])
        return real_actions

    def configure_scheme_xml(self, action):
        self.net_action = action
        self.real_actions = self.rescale_actions(action)
        tree = ET.ElementTree(file=self.file)
        root = tree.getroot()
        for key, value in paras_default.items():
            if key in self.scheme_parameters:
                root[paras_index[key]].text = str(self.real_actions[key])
            else:
                root[paras_index[key]].text = str(value)
        root[paras_index["ct_power"]].text = str(0.1 ** self.real_actions["ct_power"]) if "ct_power" in self.scheme_parameters else str(1e-5)
        tree.write(self.file)

class AlpacaExecutor:
    """
    Used for executing ALPACA.
    :param executable: location of ALPACA excutable
    :param inputfile: location of ALPACA inputfile
    :param cpu_num: how many cpu will be used
    """
    def __init__(self, executable, inputfile, cpu_num):
        self.executable = executable
        self.inputfile = inputfile
        self.cpu_num = cpu_num

    def run_alpaca(self, inputfile):
        os.system(f"cd runtime_data; mpiexec -n {self.cpu_num} {self.executable} inputfiles/{inputfile}")

class BaselineDataHandler:
    """
    A class used for dealing with the baseline data. We use WENO5 simulation as the 
    baseline. 
    :param timestep_size: simulation timestep interval
    :param time_span: end time of the simulation
    :param data_loc: location of baseline simulation. Typically we use weno5 as the
                              reference
    :param layers: states for the observation
    :param high_res: a tuple that indicates if a high resolution case is running and how times
                     larger of the domain than the training simulation
    :param get_state_func: a function that defined how to get the state, which may be different 
                           for dealing with the initial state
    :param dimension: typically 2D
    :param shape: (y, x), for non-square domain the shape shall be indicated.
    :param config: other configuration
    """
    def __init__(
            self,
            timestep_size,
            time_span,
            data_loc,
            layers,
            high_res,
            get_state_func,
            dimension = 2,
            shape = None,
            config: dict = None
    ):
        self.timestep_size = timestep_size
        self.end_time = time_span
        self.state_data_loc = os.path.join(data_loc, "domain")
        self.restart_data_loc = os.path.join(data_loc, "restart")
        self.layers = layers
        self.high_res = high_res
        self.get_state_func = get_state_func
        self.simulation_reader = Simulation2D if dimension == 2 else TaylorGreenVortex
        self.shape = shape

        self.initial_state = self.get_initial_state()
        # cached in memory. 
        self.kinetic = self.get_all_baseline_ke_reward()
        self.highorder_dissipation_rate = self.get_all_baseline_highorder_dissipation_rate()
        self.dispersive = self.get_all_baseline_dispersive_reward()

    def get_all_states(self):
        states = {}
        for timestep in np.arange(0, self.end_time, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"), shape=self.shape)
            states[end_time] = self.get_state_func(data_obj=data_obj, layers=self.layers, ave_pool=self.high_res[1])
            if self.high_res[0]:
                break
        return states

    def get_initial_state(self):
        data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_0.000000.h5"), shape=self.shape)
        states = self.get_state_func(data_obj=data_obj, layers=self.layers, ave_pool=self.high_res[1])
        return states

    def get_all_baseline_dispersive_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"), shape=self.shape)
            _, rewards[end_time], _, _ = data_obj.truncation_errors()
        return rewards

    def get_all_baseline_highorder_dissipation_rate(self):
        if self.high_res[0]:
            return None
        rate = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"), shape=self.shape)
            rate[end_time] = data_obj.result["highorder_dissipation_rate"]
        return rate

    def get_all_baseline_ke_reward(self):
        if self.high_res[0]:
            return None
        rewards = {}
        for timestep in np.arange(0, self.end_time + 1e-6, self.timestep_size):
            end_time = format(timestep, ".3f")
            data_obj = self.simulation_reader(file=os.path.join(self.state_data_loc, f"data_{end_time}*.h5"), shape=self.shape)
            rewards[end_time] = data_obj.result["kinetic_energy"].sum()
        return rewards

class SimulationHandler:
    """
    A class to deal with the simulation
    :param solver: ALPACA excutor class
    :param time_controller: TimeStepsController
    :param scheme_writer: SchemeParametersWriter
    :param baseline_data_obj: BaselineDataHandler
    :param high_res: a tuple that indicates if a high resolution case is running and how times
                     larger of the domain than the training simulation
    :param config: other configuration
    """
    def __init__(
            self,
            solver: AlpacaExecutor,
            time_controller: TimeStepsController,
            scheme_writer: SchemeParametersWriter,
            baseline_data_obj: BaselineDataHandler,
            high_res: tuple,
            config: dict
    ):
        self.solver = solver
        self.time_controller = time_controller
        self.time_span, self.timestep_size = time_controller.get_time_span(), time_controller.get_timestep_size()
        self.scheme_writer = scheme_writer
        self.inputfile = solver.inputfile
        self.baseline_data_obj = baseline_data_obj
        self.simulation_reader = self.baseline_data_obj.simulation_reader
        self.shape = self.baseline_data_obj.shape
        self.get_state_func = self.baseline_data_obj.get_state_func
        self.done = False
        self.is_crashed = False
        self.layers = baseline_data_obj.layers
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
            last_time = self.time_controller.get_restart_time_string(end_time, decimal=3)
            # restart file needs the exact file name with decimal of 6
            restart_time = self.time_controller.get_restart_time_string(end_time, decimal=6)
            restart_file = f"{self.inputfile}_{last_time}/restart/restart_{restart_time}.h5"
        return self._do_configure_inputfile(
                new_file,
                endtime=end_time,
                restore_mode=restore_mode,
                restart_file=restart_file,
                snapshots_type="Stamps"
            )

    def _do_configure_inputfile(self, file: str, endtime: str, restore_mode: str, restart_file: str, snapshots_type: str):
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

    def rename_inputfile(self, end_time):
        old_file = f"runtime_data/inputfiles/{self.inputfile}*"
        new_file = f"runtime_data/inputfiles/{self.inputfile}_{end_time}.xml"
        os.system(f"mv {old_file} {new_file}")
        return f"{self.inputfile}_{end_time}.xml"

    def run(self, inputfile):
        self.solver.run_alpaca(inputfile)

    def get_state(self, end_time):
        self.current_data = self.simulation_reader(file=f"runtime_data/{self.inputfile}_{end_time}/domain/data_{end_time}*.h5", shape=self.shape)
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            return self.baseline_data_obj.initial_state
        else:
            state = self.get_state_func(data_obj=self.current_data, layers=self.layers, ave_pool=self.high_res[1])
            return state

    def get_dispersive_to_highorder_baseline_penalty(self, end_time):
        eff_rate = self.current_data.result["effective_dissipation_rate"]
        highorder_baseline_rate = self.baseline_data_obj.highorder_dissipation_rate[end_time]
        adi_to_baseline = highorder_baseline_rate - eff_rate
        dispersion = np.where(adi_to_baseline < 0, adi_to_baseline, 0).sum()
        baseline_dispersion = self.baseline_data_obj.dispersive[end_time]
        adi = dispersion / baseline_dispersion - 1
        return adi

    def get_ke_reward(self, end_time):
        reward = self.current_data.result["kinetic_energy"].sum()
        baseline_reward = self.baseline_data_obj.kinetic[end_time]
        improvement = reward / baseline_reward - 1
        return improvement


class DebugProfileHandler:
    def __init__(
            self,
            objective: SimulationHandler,
            scheme_parameters: tuple
    ):
        self.objective = objective
        self.evaluation = False
        self.parameters = scheme_parameters
        self.info = ""
        self.action_trajectory = []

    def collect_scheme_paras(self):
        for key in self.parameters:
            self.collect_info(f"{key:<3}({self.objective.scheme_writer.real_actions[key]:<4}) ")
        self.action_trajectory.append([
            self.objective.scheme_writer.real_actions[key] for key in self.parameters
        ])

    def set_evaluation(self, evaluation):
        self.evaluation = evaluation

    def collect_info(self, info: str):
        self.info += info

    def flush_info(self):
        print(self.info)
        with open("log.txt", "a") as file:
            file.write(self.info + "\n")
        self.info = ""
