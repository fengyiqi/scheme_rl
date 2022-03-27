import numpy as np
import torch
from .env_base import AlpacaEnv, q_bound, cq_bound, eta_bound, ct_power_bound
from gym import spaces
import xml.etree.ElementTree as ET



class ImplosionEnv(AlpacaEnv):

    def __init__(self):
        config = {
            "smoothness_threshold": 0.33
        }
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionEnv, self).__init__(
            executable="/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA",
            inputfile="implosion_64",
            observation_space=spaces.Box(low=-1.0, high=1.0, shape=(len(layers), 64, 64), dtype=np.float32),
            action_space=spaces.Box(low=-1.0, high=1.0, shape=(2, ), dtype=np.float32),
            timestep_size=0.1,
            time_span=2.5,
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64",
            cpu_num=4,
            layers=layers,
            config=config
        )

    def get_reward(self, end_time):
        if self.is_crashed:
            return 0
        else:
            # truncation error improvement
            reward_tr = self.obj.get_truncation_reward(end_time=end_time)
            tr_improve = True if reward_tr > 0 else False
            # smoothness improvement
            reward_si = self.obj.get_smoothness_reward(end_time=end_time)
            si_improve = True if reward_si > 0 else False
            # smoothness indicator adaptive weight
            si_penalty = abs(np.min((reward_si, 0))) ** 1.1

            self.quality += (reward_tr - si_penalty)
            total_reward = 10 * self.quality
            if self.evaluation:
                self.debug.collect_info(f"si_penalty: {round(si_penalty, 3):<3} ")
                self.debug.collect_info(f"improvement (tr, si): {tr_improve:<2}, {si_improve:<2} ")
                self.debug.collect_info(f"quality: {self.quality}")
            return total_reward


class ImplosionHighResEnv(AlpacaEnv):
    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64/domain"
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionHighResEnv, self).__init__(
            executable=executable,
            inputfile="implosion_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.1,
            end_time=2.5,
            layers=layers,
            config=config
        )

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            self.current_data = self.objective(
                results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
                result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
            )

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )

            state.append(value)
        return np.array(state)

    def get_reward(self, end_time):
        return 1


class ImplosionOutflowEnv(AlpacaEnv):

    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_outflow_64/domain",
            smoothness_threshold=0.33
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionOutflowEnv, self).__init__(
            executable=executable,
            inputfile="implosion_outflow_64",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.04,
            end_time=0.8,
            layers=layers,
            config=config
        )
        if self.baseline_data_loc is not None:
            self.ke_baseline = self.baseline_data_handler.get_baseline_reward(prop="kinetic_energy")
            self.si_baseline = self.baseline_data_handler.get_baseline_reward(prop="smoothness_indicator")

    def get_reward(self, end_time):
        if self.is_crashed:
            return -100
        else:
            # smoothness improvement
            _, reward = self.current_data.smoothness(threshold=self.smoothness_threshold)
            reward_si = reward / self.si_baseline[end_time] - 1
            self.si_improve = True if reward_si > 0 else False

            # kinetic energy improvement (dissipation reduce)
            ke = self.current_data.result["kinetic_energy"]
            reward_ke = ke.sum() / self.ke_baseline[end_time] - 1
            self.ke_improve = True if reward_ke > 0 else False

            si_penalty = abs(np.min((reward_si, 0))) ** 1.5
            # smoothness indicator adaptive weight
            self.quality += (reward_ke - si_penalty)
            total_reward = (reward_ke - si_penalty)
            self.runtime_info += f"si_penalty: {fmt(si_penalty)}   "
            self.runtime_info += f"Improve (si, ke)={self.si_improve, self.ke_improve}   Reward: {fmt(total_reward)}   "
            self.runtime_info += f"Quality: {self.quality}"
            return total_reward


class ImplosionOutflowHighResEnv(AlpacaEnv):
    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_outflow_128/domain"
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionOutflowHighResEnv, self).__init__(
            executable=executable,
            inputfile="implosion_outflow_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.04,
            end_time=0.8,
            layers=layers,
            config=config
        )

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            self.current_data = self.objective(
                results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
                result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
            )

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )

            state.append(value)
        return np.array(state)

    def get_reward(self, end_time):
        return 1


class ImplosionCTEnv(AlpacaEnv):

    def __init__(self, config=None):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        class_config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64/domain",
            scheme_file="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            smoothness_threshold=0.33
        )
        if config is not None:
            class_config.update(config)
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionCTEnv, self).__init__(
            executable=executable,
            inputfile="implosion_64",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.1,
            end_time=2.5,
            layers=layers,
            config=class_config
        )
        if self.baseline_data_loc is not None:
            self.ke_baseline = self.baseline_data_handler.get_baseline_reward(prop="kinetic_energy")
            self.si_baseline = self.baseline_data_handler.get_baseline_reward(prop="smoothness_indicator")
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def get_reward(self, end_time):
        if self.is_crashed:
            return -100
        else:
            # smoothness improvement
            _, reward = self.current_data.smoothness(threshold=self.smoothness_threshold)
            reward_si = reward / self.si_baseline[end_time] - 1
            self.si_improve = True if reward_si > 0 else False

            # kinetic energy improvement (dissipation reduce)
            ke = self.current_data.result["kinetic_energy"]
            reward_ke = ke.sum() / self.ke_baseline[end_time] - 1
            self.ke_improve = True if reward_ke > 0 else False

            si_penalty = abs(np.min((reward_si, 0))) ** 1.1
            # smoothness indicator adaptive weight
            self.quality += (reward_ke - si_penalty)
            total_reward = 10 * (reward_ke - si_penalty)
            self.runtime_info += f"si_penalty: {fmt(si_penalty)}   "
            self.runtime_info += f"Improve (si, ke)={self.si_improve:<2}, {self.ke_improve:<2}   Reward: {fmt(total_reward):<6}   "
            self.runtime_info += f"Quality: {fmt(self.quality)}"
            return total_reward

    def configure_scheme_xml(self, action):
        tree = ET.ElementTree(file=self.schemefile)
        root = tree.getroot()
        root[0].text = "0"
        ct, eta = action[0], action[1]

        eta = np.round((eta + 1) / 2 * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = np.round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = 6
        cq = 1
        ct_power = np.round((ct + 1) / 2 * (ct_bound[1] - ct_bound[0]) + ct_bound[0], 2)
        ct = 0.1 ** ct_power

        for i, para in enumerate([q, cq, d1, d2, ct]):
            root[i + 2].text = str(para)

        tree.write(self.schemefile)
        if self.evaluation:
            self.action_trajectory.append((d1, d2, eta, ct))
            self.runtime_info += f"d1, d2, eta, ct: ({fmt(d1)}, {fmt(d2)}, {fmt(eta)}, 1.e-{ct_power:<5})  "


class ImplosionHighResCTEnv(AlpacaEnv):
    def __init__(self, config=None):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        class_config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_64/domain",
            scheme_file="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            smoothness_threshold=0.33
        )
        if config is not None:
            class_config.update(config)
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionHighResCTEnv, self).__init__(
            executable=executable,
            inputfile="implosion_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.1,
            end_time=2.5,
            layers=layers,
            config=class_config
        )
        if self.baseline_data_loc is not None:
            self.ke_baseline = self.baseline_data_handler.get_baseline_reward(prop="kinetic_energy")
            self.si_baseline = self.baseline_data_handler.get_baseline_reward(prop="smoothness_indicator")
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            self.current_data = self.objective(
                results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
                result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
            )

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )

            state.append(value)
        return np.array(state)

    def configure_scheme_xml(self, action):
        tree = ET.ElementTree(file=self.schemefile)
        root = tree.getroot()
        root[0].text = "0"
        ct, eta = action[0], action[1]

        eta = np.round((eta + 1) / 2 * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = np.round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = 6
        cq = 1
        ct_power = round((ct + 1) / 2 * (ct_bound[1] - ct_bound[0]) + ct_bound[0])
        ct = 0.1 ** ct_power

        for i, para in enumerate([q, cq, d1, d2, ct]):
            root[i + 2].text = str(para)

        tree.write(self.schemefile)
        if self.evaluation:
            self.action_trajectory.append((d1, d2, eta, ct))
            self.runtime_info += f"d1, d2, eta, ct: ({fmt(d1)}, {fmt(d2)}, {fmt(eta)}, 1.e-{ct_power})  "

    def get_reward(self, end_time):
        return 1


class ImplosionOutflowCTEnv(AlpacaEnv):

    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_outflow_64/domain",
            scheme_file="/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml",
            smoothness_threshold=0.33
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionOutflowCTEnv, self).__init__(
            executable=executable,
            inputfile="implosion_outflow_64",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.04,
            end_time=0.8,
            layers=layers,
            config=config
        )
        if self.baseline_data_loc is not None:
            self.ke_baseline = self.baseline_data_handler.get_baseline_reward(prop="kinetic_energy")
            self.si_baseline = self.baseline_data_handler.get_baseline_reward(prop="smoothness_indicator")
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def get_reward(self, end_time):
        if self.is_crashed:
            return -100
        else:
            # smoothness improvement
            _, reward = self.current_data.smoothness(threshold=self.smoothness_threshold)
            reward_si = reward / self.si_baseline[end_time] - 1
            self.si_improve = True if reward_si > 0 else False

            # kinetic energy improvement (dissipation reduce)
            ke = self.current_data.result["kinetic_energy"]
            reward_ke = ke.sum() / self.ke_baseline[end_time] - 1
            self.ke_improve = True if reward_ke > 0 else False

            si_penalty = abs(np.min((reward_si, 0))) ** 1.5
            # smoothness indicator adaptive weight
            self.quality += (reward_ke - si_penalty)
            total_reward = 10 * (reward_ke - si_penalty)
            self.runtime_info += f"si_penalty: {fmt(si_penalty)}   "
            self.runtime_info += f"Improve (si, ke)={self.si_improve:<2}, {self.ke_improve:<2}   Reward: {fmt(total_reward):<6}   "
            self.runtime_info += f"Quality: {fmt(self.quality)}"
            return total_reward

    def configure_scheme_xml(self, action):
        tree = ET.ElementTree(file=self.schemefile)
        root = tree.getroot()
        root[0].text = "0"
        ct, eta = action[0], action[1]

        eta = np.round((eta + 1) / 2 * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = np.round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = 6
        cq = 1
        ct_power = round((ct + 1) / 2 * (ct_bound[1] - ct_bound[0]) + ct_bound[0])
        ct = 0.1 ** ct_power

        for i, para in enumerate([q, cq, d1, d2, ct]):
            root[i + 2].text = str(para)

        tree.write(self.schemefile)
        if self.evaluation:
            self.action_trajectory.append((d1, d2, eta, ct))
            self.runtime_info += f"d1, d2, eta, ct: ({fmt(d1)}, {fmt(d2)}, {fmt(eta)}, 1.e-{ct_power})  "


class ImplosionOutflowHighResCTEnv(AlpacaEnv):
    def __init__(self):
        executable = "/home/yiqi/PycharmProjects/RL2D/solvers/ALPACA_32_TENO5RL_ETA"
        config = dict(
            baseline_data_loc="/home/yiqi/PycharmProjects/RL2D/baseline/implosion_outflow_128/domain",
            scheme_file = "/home/yiqi/PycharmProjects/RL2D/runtime_data/scheme.xml"
        )
        layers = ["density", "kinetic_energy", "pressure"]
        super(ImplosionOutflowHighResCTEnv, self).__init__(
            executable=executable,
            inputfile="implosion_outflow_128",
            observation_space=spaces.Box(low=-np.inf, high=np.inf, shape=(len(layers), 64, 64), dtype=np.float32),
            timestep_size=0.04,
            end_time=0.8,
            layers=layers,
            config=config
        )
        self.action_space = spaces.Box(
            low=np.array([-1, -1]),
            high=np.array([1, 1]),
            dtype=np.float32
        )

    def get_state(self, end_time):
        self.current_data = self.objective(
            results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
            result_filename=f"data_{end_time}0*.h5"
        )
        if not self.current_data.result_exit:
            self.is_crashed = True
            self.done = True
            self.current_data = self.objective(
                results_folder=f"runtime_data/{self.inputfile}_{end_time}/domain",
                result_filename=f"data_{format(float(end_time) - self.timestep_size, '.3f')}0*.h5"
            )

        state = []
        for i, layer in enumerate(self.layers):
            value = self.current_data.result[layer]
            value = torch.nn.AvgPool2d(2)(torch.tensor([value]))[0].numpy()
            value = normalize(
                value=value,
                bounds=self.bounds[self.layers[i]],
            )

            state.append(value)
        return np.array(state)

    def configure_scheme_xml(self, action):
        tree = ET.ElementTree(file=self.schemefile)
        root = tree.getroot()
        root[0].text = "0"
        ct, eta = action[0], action[1]

        eta = np.round((eta + 1) / 2 * (eta_bound[1] - eta_bound[0]) + eta_bound[0], 6)
        d1, d2 = np.round((2 + eta) / 4, 4), np.round((1 - eta) / 2, 4)
        q = 6
        cq = 1
        ct_power = round((ct + 1) / 2 * (ct_bound[1] - ct_bound[0]) + ct_bound[0])
        ct = 0.1 ** ct_power

        for i, para in enumerate([q, cq, d1, d2, ct]):
            root[i + 2].text = str(para)

        tree.write(self.schemefile)
        if self.evaluation:
            self.action_trajectory.append((d1, d2, eta, ct))
            self.runtime_info += f"d1, d2, eta, ct: ({fmt(d1)}, {fmt(d2)}, {fmt(eta)}, 1.e-{ct_power})  "

    def get_reward(self, end_time):
        return 1
