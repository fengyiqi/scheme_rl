import os
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure

from ..envs.implosion import ImplosionEnv
from ..models.networks import CustomCNN

seed = 100
set_random_seed(seed)
os.system("rm -rf ppo_models/implosion/events* ppo_models/implosion/*.csv")
reward_his = []
quality_list = [["iteration", "quality"]]

env = ImplosionEnv()
env.reset(evaluate=False)
steps = int(env.obj.time_span / env.obj.timestep_size)
print(env)

buffer_length = 4 * steps
batch_length = 2 * steps

tmp_path = "./ppo_models/implosion/"
new_logger = configure(tmp_path, ["csv", "tensorboard"])

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False
)
model = PPO("CnnPolicy", env, n_steps=buffer_length, batch_size=batch_length, n_epochs=50, device='cpu', gamma=0.99, seed=seed, policy_kwargs=policy_kwargs)
model.set_logger(new_logger)
reward_and_quality = []

model.set_env(env)

for i in range(1000):
    env.reset(evaluate=False)
    model.policy.float()
    model.learn(total_timesteps=2 * buffer_length, n_eval_episodes=2, eval_freq=5, log_interval=1,
                reset_num_timesteps=False)

    obs = env.reset(evaluate=True)
    dones = False
    while not dones:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, info = env.step(action)
    model.save(f"ppo_models/implosion/{env.inputfile}_{seed}_{i}.zip")

