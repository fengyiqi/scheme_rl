import os
import numpy as np
from .envs.env_base import AlpacaEnv
from .networks import CustomCNN
from IPython.display import clear_output
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from .plot import plot_states, plot_quality

SEED = 100


def back_scale(a):
    a1 = (a[0] - 1) / 9 * 2 - 1
    a2 = (a[1] - 1) / 99 * 2 - 1
    a3 = (a[2] - 1) / 0.8 * 2 - 1
    return np.array([a1, a2, a3])


class PPOTrain:
    def __init__(
            self,
            env: AlpacaEnv,
            n_episode_to_collect: int = 16,
            n_episode_to_train: int = 4,
            folder_name: str = None,
            features_extractor=CustomCNN,
            features_dim: int = 256,
            random_seed: int = None,
            last_iteration: int = None,
    ):
        self.env = env
        self.n_episode_to_collect = n_episode_to_collect
        self.n_episode_to_train = n_episode_to_train
        self.folder_name = self.env.inputfile if folder_name is None else folder_name
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        self.seed = SEED if random_seed is None else random_seed
        set_random_seed(self.seed)
        self.reward_his, self.quality_list = [], [["iteration", "quality"]]
        self.steps = int(env.obj.time_span / env.obj.timestep_size)

        self.iteration = last_iteration if last_iteration is not None else 0
        self.continue_training = False

    def _build_folders(self):
        if not os.path.exists(os.path.join("ppo_models", self.folder_name)):
            os.makedirs(os.path.join("ppo_models", self.folder_name))
        else:
            os.system(f"rm -rf ppo_models/{self.folder_name}/*")

    def initialize_model(self, continue_training=False):
        self.continue_training = continue_training
        if not continue_training:
            self._build_folders()
        tmp_path = f"./ppo_models/{self.folder_name}/"
        new_logger = configure(tmp_path, ["csv", "tensorboard"])

        policy_kwargs = dict(
            features_extractor_class=self.features_extractor,
            features_extractor_kwargs=dict(features_dim=self.features_dim),
            normalize_images=False
        )
        model = PPO(
            "CnnPolicy",
            self.env,
            n_steps=self.n_episode_to_collect * self.steps,
            batch_size=self.n_episode_to_train * self.steps,
            n_epochs=20,
            device='cpu',
            gamma=0.99,
            seed=self.seed,
            policy_kwargs=policy_kwargs
        )
        model.set_logger(new_logger)
        return model

    def train(
            self,
            model: PPO,
            iteration,
            eval_interval=None,
            plot_time=None,
            plot_state=None,
            continue_training=False,
            model_path=None
    ):
        # assert self.iteration == 0, "You "
        if plot_time is None:
            plot_time = []
        if plot_state is None:
            plot_state = ["density", "vorticity", "numerical_dissipation_rate"]
        # quality_list_plot = []
        if continue_training:
            assert model_path is not None, "You must specify a 'model_path'"
            model.load(model_path, device="cpu")
            print(f"load model from {model_path}")
        model.set_env(self.env)
        for i in range(self.iteration, self.iteration + iteration):
            self.env.reset(evaluate=False, iteration=i)
            model.policy.float()
            model.learn(
                total_timesteps=self.n_episode_to_collect * self.steps,
                log_interval=1,
                reset_num_timesteps=False
            )
            if eval_interval is not None and (i+1) % eval_interval == 0:
                obs = self.env.reset(evaluate=True)
                dones = False
                while not dones:
                    action, _states = model.predict(obs, deterministic=True)
                    obs, rewards, dones, info = self.env.step(action)
                print(f"{i + 1}: reward, quality =", format(self.env.cumulative_reward, ".4f"),
                      format(self.env.cumulative_quality, ".4f"))
                # quality_list_plot.append(self.env.cumulative_quality)
                # plot_quality(quality_list_plot)
                plot_states(self.env, end_times=plot_time, states=plot_state)
                clear_output(wait=True)
            model.save(f"ppo_models/{self.folder_name}/seed{self.seed}_{i}.zip")
            # print("iteration: ", i)
        self.iteration = iteration

    @staticmethod
    def eval(env, model_path, plot_time, plot_state=None, actions=None):
        if plot_state is None:
            plot_state = ["density", "vorticity", "numerical_dissipation_rate"]
        model = PPO.load(model_path, env=env, device="cpu")
        model.policy.float()
        obs = env.reset(evaluate=True)
        dones = False
        i = 0
        while not dones:
            if actions is None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = back_scale(actions[i])
            obs, rewards, dones, info = env.step(action)
        plot_states(
            env,
            end_times=plot_time,
            states=plot_state
        )
        clear_output(wait=True)

