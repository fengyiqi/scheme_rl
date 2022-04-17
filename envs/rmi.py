import numpy as np
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound

class RMIEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.1
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RMIEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA_ROEM",
            inputfile="rmi_64",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.02,
            time_span=0.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/rmi_64_weno5_roem",
            linked_reset=True,
            high_res=(False, None),
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -10
        else:
            # truncation error improvement
            reward_vor = self.obj.get_ke_reward(end_time=end_time) * 50
            vor_improve = True if reward_vor > 0 else False
            # smoothness improvement
            penalty_si = self.obj.get_dispersive_penalty(end_time)
            si_improve = True if penalty_si > 0 else False
            # smoothness indicator adaptive weight
            si_penalty = abs(np.min((penalty_si, 0))) ** 1.0

            # penalty_disper = self.obj.get_dispersive_penalty(end_time)
            # si_penalty += abs(np.min((penalty_disper, 0))) ** 1.3
            if self.obj.time_controller.get_end_time_float() < 0.12:
                si_penalty *= 10
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            quality = (reward_vor - si_penalty)
            self.cumulative_quality += quality
            total_reward = 10 * quality
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"vor_reward: {round(reward_vor, 3):<6} ")
                self.debug.collect_info(f"improve (si, vor): {si_improve:<1}, {vor_improve:<1} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
                self.debug.collect_info(f"quality: {round(quality, 3):<6}  ")
            return total_reward

class RMIHighRes128Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.1
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RMIHighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="rmi_128",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.02,
            time_span=0.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/rmi_64_teno5_roem",
            linked_reset=False,
            high_res=(True, 2),
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0

class RMIHighRes256Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.1
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RMIHighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="rmi_256",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.02,
            time_span=0.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/rmi_64_teno5_roem",
            linked_reset=False,
            high_res=(True, 4),
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0

class RMIHighRes512Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.1
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RMIHighRes512Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="rmi_512",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.02,
            time_span=0.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/rmi_64_teno5_roem",
            linked_reset=False,
            high_res=(True, 8),
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0
