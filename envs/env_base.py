import os
from abc import ABC, abstractmethod
from typing import Callable
import gym
from gym import spaces
import numpy as np
from .sim_base import (
    SchemeParametersWriter,
    TimeStepsController,
    AlpacaExecutor,
    SimulationHandler,
    DebugProfileHandler,
    BaselineDataHandler,
)
# from .sim_base import q_bound, cq_bound, eta_bound, ct_power_bound
PRINT_VERBOSE = True

def fmt(value, dig=".3f"):
    return format(value, dig)

class AlpacaEnv(gym.Env, ABC):
    """
    Base environment of ALPACA simulation, provides interface to gym-env such as reset, step
    :param executable: location of ALPACA excutable
    :param inputfile: location of ALPACA inputfile
    :param schemefile: location of TENO5 parameters interface to ALPACA
    :param observation_space: observation space
    :param action_space: action space
    :param timestep_size: simulation timestep interval
    :param time_span: end time of the simulation
    :param baseline_data_loc: location of baseline simulation. Typically we use weno5 as the
                              reference
    :param get_state_func: a function that defined how to get the state, which may be different 
                           for dealing with the initial state
    :param high_res: a tuple that indicates if a high resolution case is running and how times
                     larger of the domain than the training simulation
    :param cpu_num: how many cpu will be used
    :param dimension: typically 2D
    :param shape: (y, x), for non-square domain the shape shall be indicated.
    :param scheme_parameters: optimizing teno5 parameters
    :param layers: states for the observation
    :param config: other configuration
    """
    def __init__(
            self,
            executable: str,
            inputfile: str,
            schemefile: str,
            observation_space: spaces.Box,
            action_space: spaces.Box,
            timestep_size: float,
            time_span: float,
            baseline_data_loc: str,
            get_state_func: Callable,
            high_res: tuple = (False, None),
            cpu_num: int = 1,
            dimension: int = 2,
            shape: tuple = None,
            scheme_parameters: tuple = ("q", "cq", "eta"),
            layers: tuple = ("density", "velocity_x", "velocity_y", "pressure"),
            config: dict = None
    ):
        if config is None:
            config = {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.inputfile = inputfile
        self.config = config
        self.shape = shape
        self.obj = self._build_objective(
            executable=executable,
            inputfile=inputfile,
            scheme_parameters=scheme_parameters,
            timestep_size=timestep_size,
            time_span=time_span,
            baseline_data_loc=baseline_data_loc,
            high_res=high_res,
            get_state_func=get_state_func,
            cpu_num=cpu_num,
            dimension=dimension,
            shape=shape,
            schemefile=schemefile,
            layers=layers,
            config=config
        )
        self.debug = DebugProfileHandler(objective=self.obj, scheme_parameters=scheme_parameters)
        self._build_folders()
        self.evaluation = False
        self.iteration = None
        self.cumulative_reward = 0

    def _build_objective(
            self,
            executable: str,
            inputfile: str,
            scheme_parameters: tuple,
            timestep_size: float,
            time_span: float,
            baseline_data_loc: str,
            high_res: tuple,
            get_state_func: Callable,
            cpu_num: int,
            dimension: int,
            shape: tuple,
            schemefile: str,
            layers: list,
            config: dict
    ) -> SimulationHandler:
        schemefile = SchemeParametersWriter(schemefile, scheme_parameters)
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
            high_res=high_res,
            get_state_func=get_state_func,
            dimension=dimension,
            shape=shape,
            config=config
        )
        objective = SimulationHandler(
            solver=alpaca,
            time_controller=timestep_controller,
            baseline_data_obj=baseline_data_obj,
            scheme_writer=schemefile,
            high_res=high_res,
            config=config
        )
        return objective

    def _build_folders(self):
        if not os.path.exists("runtime_data/inputfiles"):
            os.makedirs("runtime_data/inputfiles")
        else:
            os.system("rm -rf runtime_data/inputfiles/*")
        os.system(f"cp scheme_rl/xml/{self.obj.inputfile}.xml runtime_data/inputfiles/{self.obj.inputfile}.xml")
        os.system(f"cp scheme_rl/xml/scheme.xml runtime_data/scheme.xml")

    def _reset_flags_and_buffers(self):
        self.obj.done = False
        self.cumulative_quality, self.cumulative_reward = 0, 0
        self.debug.action_trajectory = []
        os.system(f"rm -rf runtime_data/{self.obj.inputfile}_*")

    def reset(self, print_info=False, evaluate=False, iteration=-1):
        self.evaluation = evaluate
        self._reset_flags_and_buffers()
        self.obj.time_controller.counter = 0
        self.obj.is_crashed = False
        state = self.obj.baseline_data_obj.initial_state
        return state

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
        infos = self.get_infos(end_time=end_time)

        if end_time == self.obj.time_controller.get_time_span_string():
            self.obj.done = True
        if self.evaluation:
            self.debug.collect_scheme_paras()
            if PRINT_VERBOSE:
                self.debug.flush_info()
        return current_state, reward, self.obj.done, infos

    @abstractmethod
    def get_reward(self, end_time):
        return NotImplemented

    def get_infos(self, end_time):
        return {}

    def render(self, mode="human"):
        return NotImplemented

    def __str__(self):
        info = self.__class__.__name__ + " Summary:\n\n"
        info += f"\tInputfile: {self.obj.inputfile}\n"
        info += f"\tTimespan: (0, {self.obj.time_controller.get_time_span_string()}); " \
                f"Timestep size: {self.obj.time_controller.get_timestep_size()}\n"
        info += f"\tLayers: {self.obj.layers}\n"
        info += f"\tBaseline data: {self.obj.baseline_data_obj.state_data_loc}\n"
        info += "\n"
        info += f"\tExecutable: {self.obj.solver.executable}\n"
        info += f"\tCore num: {self.obj.solver.cpu_num}"
        return info


