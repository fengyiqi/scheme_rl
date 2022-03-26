import os.path

import numpy as np
import matplotlib.pyplot as plt
from boiles.objective.simulation2d import Simulation2D


def normalize(value, bounds):
    normalized = (value - bounds[0]) / (bounds[1] - bounds[0]) - 0.5
    # zero_mean = normalized - np.mean(normalized)
    return normalized


class DataHandler:
    def __init__(self):
        pass


class BaselineDataHandler(DataHandler):
    def __init__(
            self,
            timestep_size,
            time_span,
            data_loc,
            layers,
            config: dict = None
    ):
        super(BaselineDataHandler, self).__init__()

        if layers is None:
            layers = ["density", "kinetic_energy", "pressure"]
        self.timestep_size = timestep_size
        self.end_time = time_span
        self.state_data_loc = os.path.join(data_loc, "domain")
        self.restart_data_loc = os.path.join(data_loc, "restart")
        self.layers = layers
        self.smoothness_threshold = config.get("smoothness_threshold", 0.33)

        self.baseline_state = self.get_baseline_state()
        self.bounds = self.get_bounds()

    # def get_baseline_state(self) -> dict:
    #     timesteps = np.arange(0, self.end_time, self.timestep_size)
    #     baseline = {}
    #     for end_time in timesteps:
    #         state_matrix = []
    #         end_time = format(end_time, ".3f")
    #         freeshear = self.objective(results_folder=self.state_data_loc, result_filename=f"data_{end_time}*.h5")
    #         for state in self.layers:
    #             value = freeshear.result[state]
    #             state_matrix.append(value)
    #         baseline[end_time] = np.array(state_matrix)
    #     return baseline

    def get_baseline_state(self, end_time):
        data_obj = Simulation2D(file=os.path.join(self.state_data_loc, f"data_{end_time}.h5"))
        return self.get_states(data_obj=data_obj)

    def get_states(self, data_obj, normalize_states=True):
        state_matrix = []
        for state in self.layers:
            value = data_obj.result[state]
            state_matrix.append(value)
        if normalize_states:
            normalized_states = []
            for i, state in enumerate(state_matrix):
                normalized_states.append(
                    normalize(
                        value=state,
                        bounds=self.bounds[self.layers[i]]
                    )
                )
            return normalized_states
        return state_matrix

    def get_bounds(self):
        baseline_data = self.baseline_state
        baseline_flat = []
        for i in range(len(self.layers)):
            baseline_flat.append([])
        for key, value in baseline_data.items():
            for i, state in enumerate(value):
                baseline_flat[i] += np.array(state).flatten().tolist()
        bounds = {}
        for i, key in enumerate(self.layers):
            bounds[key] = (min(baseline_flat[i]), max(baseline_flat[i]))
        return bounds

    def get_initial_state(self):
        initial_state = self.baseline_state["0.000"]
        normalized_initial_state = []
        for i, value in enumerate(initial_state):
            normalized_initial_state.append(
                normalize(
                    value=value,
                    bounds=self.bounds[self.layers[i]],
                )
            )
        return np.array(normalized_initial_state)

    def get_baseline_reward(self, prop):
        timesteps = np.arange(self.timestep_size, self.end_time + self.timestep_size, self.timestep_size)
        baseline = {}
        for end_time in timesteps:
            end_time = format(end_time, ".3f")
            freeshear = self.objective(results_folder=self.data_loc, result_filename=f"data_{end_time}*.h5")
            if prop == "kinetic_energy":
                data = freeshear.result["kinetic_energy"]
                data = data.sum().round(4)
            if prop == "numerical_dissipation_rate":
                _, _, _, data = freeshear.truncation_errors()
                data = round(data, 4)
            if prop == "smoothness_indicator":
                _, data = freeshear.smoothness(threshold=self.smoothness_threshold)
                data = round(data, 4)

            baseline[end_time] = data
        return baseline

    def test_functionality(self):
        baseline_data = self.baseline_state
        timesteps = np.arange(self.timestep_size, self.end_time, self.timestep_size)
        time = np.random.choice(timesteps)
        data = baseline_data[format(time, ".3f")]

        print("Running test function")
        print("Dimension: \n", data.shape)
        print("Bounds: \n", self.get_bounds())
        print("Baseline reward (kinetic_energy): \n", self.get_baseline_reward(prop="kinetic_energy"))
        print("Baseline reward (numerical_dissipation_rate): \n",
              self.get_baseline_reward(prop="numerical_dissipation_rate"))
        print("Baseline reward (smoothness_indicator): \n", self.get_baseline_reward(prop="smoothness_indicator"))

        plt.figure(figsize=(4 * len(self.layers), 3))
        plt.suptitle(f"t={format(time, '.3f')}")
        for i in range(len(self.layers)):
            plt.subplot(1, len(self.layers), i + 1)
            plt.imshow(data[i])
            plt.title(self.layers[i])
            plt.colorbar()

        plt.figure(figsize=(4 * len(self.layers), 3))
        plt.suptitle(f"t={format(time, '.3f')}, normalized")
        for i in range(len(self.layers)):
            normalized_data = normalize(
                value=data[i],
                bounds=(data[i].min(), data[i].max()),
            )
            plt.subplot(1, len(self.layers), i + 1)
            plt.imshow(normalized_data)
            plt.title(self.layers[i])
            plt.colorbar()

        plt.figure(figsize=(4 * len(self.layers), 3))
        plt.suptitle("Initial state")
        initial_state = self.get_initial_state()
        for i in range(len(self.layers)):
            plt.subplot(1, len(self.layers), i + 1)
            plt.imshow(initial_state[i])
            plt.title(self.layers[i])
            plt.colorbar()
        plt.show()
        plt.close()



# def main():
#     baseline_handler = FullImplosionBaselineData(
#         2.5,
#         0.1,
#         data_loc="/home/yiqi/PycharmProjects/RL2D/full_implosion/full_implosion_teno5/domain"
#     )
#     baseline_handler.test_functionality()
    # baseline_handler = RTIBaselineData(
    #     end_time=2.0,
    #     timestep_size=0.1,
    #     data_loc="/home/yiqi/PycharmProjects/RL2D/rti/rti_teno5/domain"
    # )
    # baseline_handler.test_functionality()


# if __name__ == '__main__':
#     main()
