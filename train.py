import argparse

parser = argparse.ArgumentParser(description='A script to trigger the training of the RL model.')
# implosion, rti, config3, shear
parser.add_argument("-case_name", type=str, required=True, help="type a test case name: implosion, rti, config3")
# workstation, desktop
parser.add_argument("-computer", type=str, required=True, help="type which computer is used: workstation, desktop")
parser.add_argument("-seed", type=int, required=True, help="type an integer random seed")
parser.add_argument("-buffer_length", type=int, default=1024, help="type a buffer length")
parser.add_argument("-batch_length", type=int, default=64, help="type a batch length")
parser.add_argument("-iteration", type=int, default=400, help="type a number of iteration")

args = parser.parse_args()
print(args)

if args.computer.lower() == "workstation":
    import torch
    print("4 threads are used in workstation")
    torch.set_num_interop_threads(4)

import os
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure
from scheme_rl.networks import CustomCNN
from scheme_rl.plot import plot_action_trajectory, plot_states, plot_reward_and_quality

os.system("rm -rf ppo_models runtime_data log.txt")
seed = args.seed
set_random_seed(seed)

if args.case_name.lower() == "implosion":
    from scheme_rl.envs.implosion import ImplosionEnv
    env = ImplosionEnv()
if args.case_name.lower() == "rti":
    from scheme_rl.envs.rti import RTIEnv
    env = RTIEnv()
if args.case_name.lower() == "config3":
    from scheme_rl.envs.riemann2d import RiemannConfig3Env
    env = RiemannConfig3Env()
if args.case_name.lower() == "shear":
    from scheme_rl.envs.shear_flow import FreeShearEnv
    env = FreeShearEnv()
if args.case_name.lower() == "viscous_shock":
    from scheme_rl.envs.viscous_shock import ViscousShockTubeEnv
    env = ViscousShockTubeEnv()
if args.case_name.lower() == "moving_gresho":
    from scheme_rl.envs.gresho import MovingGreshoEnv
    env = MovingGreshoEnv()

env.reset()

policy_kwargs = dict(
    features_extractor_class=CustomCNN,
    features_extractor_kwargs=dict(features_dim=256),
    normalize_images=False
)
model = PPO(
    "CnnPolicy", 
    env, 
    n_steps=args.buffer_length, 
    batch_size=args.batch_length, 
    n_epochs=10, 
    device='cpu' if args.computer.lower() == "workstation" else "cuda", 
    gamma=0.99, 
    clip_range=0.1,
    seed=seed, 
    policy_kwargs=policy_kwargs
)

tmp_path = f"./ppo_models/"
new_logger = configure(tmp_path, ["csv", "tensorboard"])
model.set_logger(new_logger)
model.set_env(env)

reward_and_quality = []
for i in range(args.iteration):
    env.reset()
    model.policy.float()
    model.learn(
        total_timesteps=args.buffer_length, 
        n_eval_episodes=2, 
        eval_freq=5, 
        log_interval=1,
        reset_num_timesteps=False
    )
    model.save(f"ppo_models/{env.inputfile}_{seed}_{i}.zip")
    if args.computer.lower() == "desktop" and i % 10 == 0:
        obs = env.reset(evaluate=True)
        dones = False
        while not dones:
            action, _ = model.predict(obs, deterministic=True)
            obs, rewards, dones, info = env.step(action)
        plot_action_trajectory(env.debug.action_trajectory, path=f"./figures/actions.jpg")
        reward_and_quality.append([env.cumulative_reward, env.cumulative_quality])
        plot_reward_and_quality(reward_and_quality, path=f"./figures/reward.jpg")
        plot_states(
            env,
            end_times=["0.500", "1.000", "1.500", "2.000", "2.500", "3.000"], 
            states=["density", "velocity_x", "velocity_y", "pressure", "vorticity"],
            path=f"./figures/states_{i}.jpg",
            shape=env.shape
        )
