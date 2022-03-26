import os
import numpy as np
import xml.etree.ElementTree as ET
from boiles.objective.simulation2d import Simulation2D
from .env_base import q_bound, cq_bound, eta_bound, ct_power_bound, action_bound

if __name__ == '__main__':
    from .data_handler import normalize, BaselineDataHandler
else:
    from .data_handler import normalize, BaselineDataHandler




class TimeStepsController:
    def __init__(self, time_span, timestep_size):
        self._time_span = time_span
        self._timestep_size = timestep_size
        self.counter = 1

    def get_time_span(self):
        return self._time_span

    def get_timestep_size(self):
        return self._timestep_size

    def get_end_time_float(self, counter):
        return counter * self._timestep_size

    def get_end_time_string(self, counter):
        return format(self.get_end_time_float(counter), ".3f")

    def get_total_steps(self):
        return int(self._time_span / self._timestep_size)

    def get_restart_time_string(self, end_time):
        # restart file needs the exact file name
        return format(float(end_time) - self._timestep_size, ".6f")



class SchemeParametersWriter:
    def __init__(self, file):
        self.file = file
        self.para_index = {key: value for value, key in enumerate(("q", "cq", "d1", "d2", "ct"), 2)}

    @staticmethod
    def rescale_actions(action):
        q, cq, eta, ct_power = action[0], action[1], action[2], action[3]
        action_range = action_bound[1] - action_bound[0]
        eta = round((eta - action_bound[0]) / action_range * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = round((q - action_bound[0]) / action_range * (q_bound[1] - q_bound[0]) + q_bound[0])
        cq = round((cq - action_bound[0]) / action_range * (cq_bound[1] - cq_bound[0]) + cq_bound[0])
        ct_power = round((ct_power - action_bound[0]) / action_range * \
                         (ct_power_bound[1] - ct_power_bound[0]) + ct_power_bound[0])
        ct = 0.1 ** ct_power
        return q, cq, d1, d2, ct

    def configure_scheme_xml(self, action):
        q, cq, d1, d2, ct = self.rescale_actions(action)
        tree = ET.ElementTree(file=self.file)
        root = tree.getroot()
        root[0].text = "0"
        for i, para in enumerate([q, cq, d1, d2, ct], 2):
            root[i].text = str(para)

        tree.write(self.file)
        # if self.evaluation:
        #     self.action_trajectory.append((q, cq, d1, d2, eta))
        #     self.runtime_info += f"q, cq, d1, d2, eta: ({q:<2}, {cq:<3}, {fmt(d1)}, {fmt(d2)}, {fmt(eta)})   "

class DebugProfileHandler:
    def __init__(self):
        pass

class AlpacaExecutor:
    def __init__(self, executable, inputfile, cpu_num):
        self.executable = executable
        self.inputfile = inputfile
        self.cpu_num = cpu_num

class SimulaitonHandler:
    def __init__(
            self,
            solver: AlpacaExecutor,
            time_controller: TimeStepsController,
            baseline_data_obj: BaselineDataHandler,
            linked_reset: bool
    ):
        self.solver = solver
        self.time_controller = time_controller
        self.inputfile = solver.inputfile
        self.baseline_data_obj = baseline_data_obj
        self.is_crashed = False
        self.done = False
        self.layers = baseline_data_obj.layers
        self.linked_reset = linked_reset

    def configure_input_file(self, end_time):
        new_file = self.rename_inputfile(end_time)
        # starting from initial condition, no restart
        if self.time_controller.counter == 1:
            restore_mode = "Off"
            restart_file = "None"
        elif self.is_crashed and self.linked_reset:
            self.is_crashed = False
            restore_mode = "Forced"
            restart_time = self.time_controller.get_restart_time_string(end_time)
            restart_file = os.path.join(
                self.baseline_data_obj.restart_data_loc,
                f"restart_{restart_time}.h5"
            )
        else:
            restore_mode = "Forced"
            restart_time = self.time_controller.get_restart_time_string(end_time)
            if self.is_crashed and self.linked_reset:
                self.is_crashed = False
                restart_file = os.path.join(
                    self.baseline_data_obj.restart_data_loc,
                    f"restart_{restart_time}.h5"
                )
            else:
                restart_file = f"{self.inputfile}_{restart_time}/restart/restart_{restart_time}.h5"
        return self.configure_input_xml(
                new_file,
                endtime=end_time,
                restore_mode=restore_mode,
                restart_file=restart_file,
                snapshots_type="Stamps"
            )

    def configure_input_xml(self, file: str, endtime: str, restore_mode: str, restart_file: str, snapshots_type: str):
        # All arguments should be string
        tree = ET.ElementTree(file=file)
        root = tree.getroot()

        root[4][1].text = endtime  # timeControl -> endTime
        root[6][0][0].text = restore_mode  # restart -> restore -> mode
        root[6][0][1].text = restart_file  # restart -> restore -> fileName
        root[6][1][0].text = snapshots_type  # restart -> snapshots -> type
        root[6][1][3][0].text = endtime  # restart -> snapshots -> stamps -> ts1

        tree.write(file)
        return True

    def get_states(self, end_time):
        current_data = Simulation2D(file=f"runtime_data/{self.inputfile}_{end_time}/domain/data_{end_time}0*.h5")
        if not current_data.result_exit:
            self.is_crashed = True
            self.done = True
            return self.baseline_data_obj.get_initial_state()
        else:
            states = []
            for i, layer in enumerate(self.layers):
                value = current_data.result[layer]
                value = normalize(
                    value=value,
                    bounds=self.baseline_data_obj.bounds[self.layers[i]],
                )
                states.append(value)
            return np.array(states)

    def rename_inputfile(self, end_time):
        old_file = f"runtime_data/inputfiles/{self.inputfile}_*.xml"
        new_file = f"runtime_data/inputfiles/{self.inputfile}_{end_time}.xml"

        os.system(f"mv {old_file} {new_file}")
        return new_file



class GymIOSpaceHandler:
    def __init__(self, observation_space, action_space):
        self._observation_space = observation_space
        self._action_space = action_space

    def get_io_space(self):
        return self._observation_space, self._action_space
