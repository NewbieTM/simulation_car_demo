import torch
import numpy as np
from circle_env import CircleDriveEnv
from bc_train_raw import BCNet

from stable_baselines3.common.logger import configure

bc_model = BCNet()
bc_model.load_state_dict(torch.load("bc_model.pth"))
bc_model.eval()

log_path = "./logs/bc_baseline/"
new_logger = configure(log_path, ["stdout", "tensorboard"])

env = CircleDriveEnv()
episodes = 20
all_cte = []
all_rewards = []

for ep in range(episodes):
    obs, _ = env.reset()
    ep_cte = []
    ep_rewards = []
    for t in range(500):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
        action = bc_model(obs_tensor).detach().numpy()[0]
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        ep_cte.append(abs(info["cross_track"]))
        ep_rewards.append(reward)
        if done:
            break
    all_cte.append(np.mean(ep_cte))
    all_rewards.append(np.sum(ep_rewards))

new_logger.record("bc/mean_cte", np.mean(all_cte))
new_logger.record("bc/mean_reward", np.mean(all_rewards))
new_logger.dump(0)
