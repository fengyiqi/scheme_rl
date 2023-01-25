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
            scheme_parameters: list = ["q", "cq", "eta"],
            layers: tuple = ("density", "velocity_x", "velocity_y", "pressure"),
            config: dict = None
    ):
        if config is None:
            config = {}
        self.observation_space = observation_space
        self.action_space = action_space
        self.inputfile = inputfile
        self.timestep_size = timestep_size
        self.end_time = time_span
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
        self.verbose = True

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

    def reset(self, verbose=True, evaluate=False, iteration=-1):
        self.verbose = verbose
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
        self.obj.run(inputfile=inputfile, evaluation=self.evaluation)
        current_state = self.obj.get_state(end_time=end_time)
        if self.inputfile == "doublemach_32" and (np.any(current_state[0] > 25) or np.any(current_state[3] > 600)):
            # raise Exception("Simulation crashed!")
            self.obj.is_crashed = True
            self.obj.done = True
        reward = self.get_reward(end_time=end_time)
        infos = self.get_infos(end_time=end_time)

        if end_time == self.obj.time_controller.get_time_span_string():
            self.obj.done = True
        if self.evaluation:
        # if self.obj.is_crashed:
            self.debug.collect_scheme_paras()
            if self.verbose:
                self.debug.flush_info()
        return current_state, reward, self.obj.done, infos

    def compute_reward(self, end_time, coef_dict, bias=0.0, scale=1):
        if self.obj.is_crashed:
            # self.cumulative_reward += -50
            return -50
        else:
            # compute the kinetic energy improvement
            reward_ke = self.obj.get_ke_reward(end_time=end_time)
            # compute the anti-diffusion improvement
            reward_si = self.obj.get_dispersive_to_highorder_baseline_penalty(end_time=end_time)
            si_penalty = np.max((reward_si, 0)) * coef_dict[end_time]
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            reward = reward_ke - si_penalty
            total_reward = (reward + bias) * scale
            self.cumulative_reward += reward
            # print(reward_ke, si_penalty)
            if self.evaluation:
            # if True:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"disper: {round(reward_si, 3):<6} ")
                self.debug.collect_info(f"coef: {round(coef_dict[end_time], 3):<5} ")
                self.debug.collect_info(f"disper_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
            return total_reward

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
        info += f"\tParameters: {self.obj.scheme_writer.scheme_parameters}\n"
        info += f"\tBaseline data: {self.obj.baseline_data_obj.state_data_loc}\n"
        info += "\n"
        info += f"\tExecutable: {self.obj.solver.executable}\n"
        info += f"\tCore num: {self.obj.solver.cpu_num}"
        return info


