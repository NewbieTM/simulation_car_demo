from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from circle_env import CircleDriveEnv

def make_env(reward_type):
    return lambda: CircleDriveEnv(radius=10.0, reward_type=reward_type)

env = make_vec_env(make_env("aligned"), n_envs=4)
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./logs/ppo_circle")
model.learn(total_timesteps=200_000)
model.save("ppo_circle_aligned")
