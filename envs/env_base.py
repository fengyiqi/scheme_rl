import os
from abc import ABC, abstractmethod

import gym
from gym import spaces
import numpy as np
from .sim_base import (
    SchemeParametersWriter,
    TimeStepsController,
    AlpacaExecutor,
    SimulationHandler,
    DebugProfileHandler,
    BaselineDataHandler
)

q_bound = (1, 10)
cq_bound = (1, 100)
eta_bound = (0.4, 0.9)
ct_power_bound = (3, 15)
action_bound = (-1, 1)

def fmt(value, dig=".3f"):
    return format(value, dig)

class AlpacaEnv(gym.Env, ABC):

    def __init__(
            self,
            executable: str,
            inputfile: str,
            observation_space: spaces.Box,
            action_space: spaces.Box,
            timestep_size: float,
            time_span: float,
            baseline_data_loc: str,
            cpu_num: int = 1,
            schemefile: str = "/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            layers: list = None,
            config: dict = None
    ):
        if layers is None:
            layers = ["density", "kinetic_energy", "pressure"]
        if config is None:
            config = dict()
        self.observation_space = observation_space
        self.action_space = action_space
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)
        self.done, self.si_improve, self.ke_improve, self.evaluation = False, False, False, False
        self.quality = 0
        self.linked_reset = config.get("linked_reset", False)
        self.obj = self._build_objective(
            executable=executable,
            inputfile=inputfile,
            timestep_size=timestep_size,
            time_span=time_span,
            baseline_data_loc=baseline_data_loc,
            cpu_num=cpu_num,
            schemefile=schemefile,
            layers=layers,
            config=config
        )
        self.debug = DebugProfileHandler(objective=self.obj)
        self._build_folders()

    def _build_objective(
            self,
            executable: str,
            inputfile: str,
            timestep_size: float,
            time_span: float,
            baseline_data_loc: str,
            cpu_num: int,
            schemefile: str,
            layers: list,
            config: dict
    ) -> SimulationHandler:
        schemefile = SchemeParametersWriter(schemefile)
        alpaca = AlpacaExecutor(
            executable=executable,
            inputfile=inputfile,
            cpu_num=cpu_num
        )
        timestep_controller = TimeStepsController(
            time_span=time_span,
            timestep_size=timestep_size
        )
        baseline_data_obj = BaselineDataHandler(
            timestep_size=timestep_size,
            time_span=time_span,
            data_loc=baseline_data_loc,
            layers=layers,
            config=config
        )
        objective = SimulationHandler(
            solver=alpaca,
            time_controller=timestep_controller,
            baseline_data_obj=baseline_data_obj,
            scheme_writer=schemefile,
            linked_reset=True
        )
        return objective


    def _build_folders(self):
        if not os.path.exists("runtime/inputfiles"):
            os.makedirs("runtime/inputfiles")
        else:
            os.system("rm -rf runtime/inputfiles/*")
        os.system(f"mv scheme_rl/xml/{self.obj.inputfile}.xml runtime_data/inputfiles/{self.obj.inputfile}")
        os.system(f"mv scheme_rl/xml/scheme.xml runtime_data/scheme.xml")

    def _reset_flags_and_buffers(self):
        # reset flags and buffers, e.g. self.counter, self.is_crashed
        if not self.linked_reset:
            # the is_crashed flag is used in configuring inputfile and will be reset there
            self.is_crashed = False
            self.obj.time_controller.counter = 1
        self.done = False
        self.quality = 0
        os.system(f"rm -rf runtime_data/{self.obj.inputfile}_*")

    def _if_reset_from_crashed(self):
        # "and" function
        conditions = (
            self.linked_reset,
            self.obj.is_crashed,
            self.obj.time_controller.counter < self.obj.time_controller.get_total_steps() - 1,
        )
        return False not in conditions

    def reset(self, print_info=False, evaluate=False):
        if self._if_reset_from_crashed():
            self.obj.time_controller.counter += 0
            end_time = self.obj.time_controller.get_end_time_string()
            states = self.obj.baseline_data_obj.states[end_time]
            self._reset_flags_and_buffers()
            return np.array(states)
        self._reset_flags_and_buffers()
        if evaluate:
            self.debug.collect_scheme_paras()
            self.debug.flush_info()
        return self.obj.baseline_data_obj.initial_state

    def step(self, action: list):
        assert self.action_space.contains(action), f"Invalid action! {action}"
        # action = [np.tanh(a) for a in action]
        self.obj.time_controller.counter += 1
        self.obj.scheme_writer.configure_scheme_xml(action)
        end_time = self.obj.time_controller.get_end_time_string()
        inputfile = self.obj.configure_inputfile(end_time=end_time)
        self.obj.run(inputfile=inputfile)
        current_state = self.obj.get_state(end_time=end_time)
        reward = self.get_reward(end_time=end_time)

        if end_time == self.obj.time_controller.get_time_span_string():
            self.done = True
        return current_state, reward, self.done, {}

    @abstractmethod
    def get_reward(self, end_time):
        return NotImplemented


    def render(self, mode="human"):
        return NotImplemented

    def __str__(self):
        info = self.__class__.__name__ + " Summary:\n\n"
        info += f"\tInputfile: {self.obj.inputfile}\n"
        info += f"\tTimespan: (0, {self.obj.time_controller.get_time_span_string()}); " \
                f"Timestep size: {self.obj.time_controller.get_timestep_size()}\n"
        info += f"\tParameters: q {q_bound}; Cq {cq_bound}; Eta {eta_bound}; Ct(power) {ct_power_bound}\n"
        info += f"\tLayers: {self.obj.layers}\n"
        info += f"\tSmoothness: {self.smoothness_threshold}\n"
        info += f"\tBaseline data: {self.obj.baseline_data_obj.state_data_loc}\n"
        info += "\n"
        info += f"\tExecutable: {self.obj.solver.executable}\n"
        info += f"\tCore num: {self.obj.solver.cpu_num}"
        # print("inputfile: ", self.inputfile)
        return info


