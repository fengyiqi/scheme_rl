# to use this file, copy it to the folder parallel with scheme_rl

from scheme_rl.envs.rmi import RMIEnv, RMIHighRes128Env, RMIHighRes256Env, RMIHighRes512Env
from scheme_rl.envs.riemann2d import RiemannConfig3Env, RiemannConfig3HighResEnv
from scheme_rl.train import PPOTrain

for seed in (100, 200, 300):
    env = RiemannConfig3Env()
    ppo = PPOTrain(
        env=env,
        n_episode_to_collect=32,
        n_episode_to_train=8,
        folder_name=env.inputfile+str(seed),
        random_seed=seed
    )
    model = ppo.initialize_model()
    ppo.train(model, iteration=100)
