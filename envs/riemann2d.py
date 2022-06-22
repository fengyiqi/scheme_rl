import numpy as np
from .env_base import AlpacaEnv
from gym import spaces
from .sim_base import action_bound


class RiemannConfig3Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.15
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RiemannConfig3Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            schemefile="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            inputfile="config3_64",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5",
            linked_reset=False,
            high_res=(False, None),
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if self.obj.is_crashed:
            return -10
        else:

            reward_ke = self.obj.get_ke_reward(end_time)
            ke_improve = True if reward_ke > 0 else False
            # smoothness improvement
            reward_si = self.obj.get_dispersive_penalty(end_time)
            si_improve = True if reward_si > 0 else False
            si_penalty = abs(np.min((reward_si, 0))) ** 1.9
            # since we modify Gaussian to SquashedGaussian, we don't need action penalty anymore.
            # modify sb3/common/distributions/line 661, DiagGaussianDistribution to SquashedDiagGaussianDistribution
            quality = (reward_ke - si_penalty)
            self.cumulative_quality += quality
            total_reward = 10 * (quality + .02)
            # if total_reward < 0:
            #     self.obj.done = True
            #     return -10
            self.cumulative_reward += total_reward
            if self.evaluation:
                end_time = self.obj.time_controller.get_end_time_string()
                self.debug.collect_info(f"{self.obj.time_controller.get_restart_time_string(end_time, decimal=3)} -> ")
                self.debug.collect_info(f"{end_time}: ")
                self.debug.collect_info(f"disper: {round(reward_si, 3):<6} ")
                self.debug.collect_info(f"disper_penalty: {round(si_penalty, 3):<5} ")
                self.debug.collect_info(f"ke_reward: {round(reward_ke, 3):<6} ")
                self.debug.collect_info(f"reward: {round(total_reward, 3):<6} ")
                self.debug.collect_info(f"improve (si, vor): {si_improve:<1}, {ke_improve:<1} ")
                self.debug.collect_info(f"quality: {round(quality, 3):<6}  ")
            return total_reward


class RiemannConfig3HighRes128Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.15
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RiemannConfig3HighRes128Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="config3_128",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5",
            linked_reset=False,
            high_res=(True, 2),
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0


class RiemannConfig3HighRes256Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.15
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RiemannConfig3HighRes256Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="config3_256",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5",
            linked_reset=False,
            high_res=(True, 4),
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0


class RiemannConfig3HighRes512Env(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.15
        }
        layers = ["density", "kinetic_energy", "pressure"]
        paras = ("q", "cq", "eta")
        super(RiemannConfig3HighRes512Env, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="config3_512",
            parameters=paras,
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=action_bound[0], high=action_bound[1], shape=(len(paras), ), dtype=np.float32),
            timestep_size=0.01,
            time_span=1.0,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/config3_64_weno5",
            linked_reset=False,
            high_res=(True, 8),
            cpu_num=6,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        return 0
