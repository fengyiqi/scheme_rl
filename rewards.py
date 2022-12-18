import time
from scheme_rl.envs.implosion import ImplosionEnv
from scheme_rl.envs.riemann2d import RiemannConfig3Env
from scheme_rl.postprocessing.utils import test_deterministic_reward
import pandas as pd
env = RiemannConfig3Env()

for seed in [100]:
    rewards = []
    for it in range(200, 400):
        print(f"test {it}")
        reward = test_deterministic_reward(
            model_path=f"./ppo_models/config3_64_{seed}_{it}.zip",
            env=env
        )
        rewards.append(reward)
    data = dict(iteration=range(200, 400), rewards=rewards)
    df = pd.DataFrame(data)
    df.to_csv(f"./config3_{seed}_rewards.csv", index=False)