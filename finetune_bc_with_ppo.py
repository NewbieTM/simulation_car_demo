import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor

from circle_env import CircleDriveEnv
import shutil


LOG_DIR = "./logs"
if os.path.exists(LOG_DIR):
    shutil.rmtree(LOG_DIR)
os.makedirs(LOG_DIR, exist_ok=True)


class SafetyCallback(BaseCallback):
    def __init__(self, eval_env, eval_every_steps=5000, episode_len=500, verbose=0):
        super().__init__(verbose)
        self.eval_env = eval_env
        self.eval_every_steps = eval_every_steps
        self.episode_len = episode_len

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_every_steps == 0:
            obs, _ = self.eval_env.reset()
            ctes = []
            rew_sum = 0.0
            for _ in range(self.episode_len):
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = self.eval_env.step(action)
                ctes.append(abs(info["cross_track"]))
                rew_sum += float(reward)
                if terminated or truncated:
                    obs, _ = self.eval_env.reset()
            self.logger.record("safety/mean_cte", float(np.mean(ctes)))
            self.logger.record("eval/mean_reward_one_ep", rew_sum)
        return True


def policy_mean_actions(policy, obs_tensor):
    features = policy.extract_features(obs_tensor)
    latent_pi, _ = policy.mlp_extractor(features)

    try:
        dist_tuple = policy._get_action_dist_from_latent(latent_pi)
        dist = dist_tuple[0] if isinstance(dist_tuple, tuple) else dist_tuple
        if hasattr(dist.distribution, "mean"):
            mu = dist.distribution.mean
        elif hasattr(dist.distribution, "loc"):
            mu = dist.distribution.loc
        else:
            mu = policy.action_net(latent_pi)
    except Exception:
        mu = policy.action_net(latent_pi)

    if getattr(policy, "squash_output", False):
        mu = torch.tanh(mu)
    return mu


def bc_pretrain_policy(
    ppo_model,
    obs_np: np.ndarray,
    acts_np: np.ndarray,
    epochs: int = 5,
    batch_size: int = 256,
    lr: float = 1e-3,
    log_dir: str = "./logs/bc_pretrain"
):
    device = ppo_model.device
    policy = ppo_model.policy
    policy.train()

    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)

    ds = TensorDataset(
        torch.tensor(obs_np, dtype=torch.float32),
        torch.tensor(acts_np, dtype=torch.float32),
    )
    loader = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=False)

    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)

    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            mu = policy_mean_actions(policy, x)
            loss = F.mse_loss(mu, y)
            loss.backward()
            optimizer.step()

            running += loss.item()
            writer.add_scalar("bc_pretrain/loss_iter", loss.item(), global_step)
            global_step += 1

        writer.add_scalar("bc_pretrain/loss_epoch", running / len(loader), epoch)
    writer.close()


def evaluate_policy_bc_baseline(ppo_model, episodes=20, episode_len=500, log_dir="./logs/bc_baseline"):
    writer = SummaryWriter(log_dir)
    env = CircleDriveEnv()
    all_cte, all_rewards = [], []

    for _ in range(episodes):
        obs, _ = env.reset()
        ctes = []
        R = 0.0
        for _ in range(episode_len):
            action, _ = ppo_model.predict(obs, deterministic=True)
            obs, reward, term, trunc, info = env.step(action)
            ctes.append(abs(info["cross_track"]))
            R += float(reward)
            if term or trunc:
                break
        all_cte.append(float(np.mean(ctes)))
        all_rewards.append(R)

    writer.add_scalar("bc_baseline/mean_cte", float(np.mean(all_cte)), 0)
    writer.add_scalar("bc_baseline/mean_reward", float(np.mean(all_rewards)), 0)
    writer.close()


def load_demos_flat(demos_path="demos_circle.npy", bc_obs_path="bc_obs.npy", bc_act_path="bc_actions.npy"):
    if os.path.exists(bc_obs_path) and os.path.exists(bc_act_path):
        print(f"[load] Found flat BC files: {bc_obs_path}, {bc_act_path}")
        obs_np = np.load(bc_obs_path).astype(np.float32)
        acts_np = np.load(bc_act_path).astype(np.float32)
        return obs_np, acts_np

    if not os.path.exists(demos_path):
        raise FileNotFoundError(f"Не найден ни {bc_obs_path}/{bc_act_path} ни {demos_path}")

    demos = np.load(demos_path, allow_pickle=True)
    print(f"[load] Loaded demos file {demos_path}, episodes: {len(demos)}")
    obs_list = []
    act_list = []

    for i, ep in enumerate(demos):
        if isinstance(ep, dict):
            obs_ep = np.asarray(ep.get("obs"))
            acts_ep = np.asarray(ep.get("actions"))
        else:
            try:
                obs_ep, acts_ep = ep
                obs_ep = np.asarray(obs_ep)
                acts_ep = np.asarray(acts_ep)
            except Exception as e:
                raise ValueError(f"Неизвестный формат эпизода в demos[{i}]: {type(ep)}") from e

        if obs_ep.size == 0 or acts_ep.size == 0:
            continue

        if obs_ep.shape[0] != acts_ep.shape[0]:
            L = min(obs_ep.shape[0], acts_ep.shape[0])
            obs_ep = obs_ep[:L]
            acts_ep = acts_ep[:L]

        obs_list.append(obs_ep)
        act_list.append(acts_ep)

    if len(obs_list) == 0:
        raise ValueError("Ни одного валидного шага не найдено в demos")

    obs_np = np.concatenate(obs_list, axis=0).astype(np.float32)
    acts_np = np.concatenate(act_list, axis=0).astype(np.float32)

    print(f"[load] Flattened demos -> obs: {obs_np.shape}, acts: {acts_np.shape}")
    return obs_np, acts_np



def main():
    seed = 42
    np.random.seed(seed)
    torch.manual_seed(seed)

    obs_np, acts_np = load_demos_flat(demos_path="demos_circle.npy", bc_obs_path="bc_obs.npy", bc_act_path="bc_actions.npy")

    assert obs_np.ndim == 2 and obs_np.shape[1] == 5, f"obs_np must be (N,5), got {obs_np.shape}"
    assert acts_np.ndim == 2 and acts_np.shape[1] == 2, f"acts_np must be (N,2), got {acts_np.shape}"

    n_envs = 8
    venv = DummyVecEnv([lambda: Monitor(CircleDriveEnv(reward_type='distance_sq')) for _ in range(n_envs)])
    print(f"[env] DummyVecEnv with {n_envs} envs created")

    model = PPO("MlpPolicy", venv, verbose=1, tensorboard_log=LOG_DIR)

    bc_pretrain_policy(
        model,
        obs_np,
        acts_np,
        epochs=3,
        batch_size=256,
        lr=1e-3,
        log_dir=os.path.join(LOG_DIR, "bc_pretrain")
    )

    model.save("ppo_bc_pretrained")
    print("[save] Saved BC-pretrained model: ppo_bc_pretrained.zip")

    evaluate_policy_bc_baseline(
        model,
        episodes=20,
        episode_len=500,
        log_dir=os.path.join(LOG_DIR, "bc_baseline")
    )

    eval_env = CircleDriveEnv()
    callback = SafetyCallback(eval_env, eval_every_steps=5000, episode_len=500)
    model.learn(total_timesteps=100_000, tb_log_name="ppo_finetune", callback=callback)

    model.save("ppo_finetuned_from_bc")
    print("[save] Saved finetuned model: ppo_finetuned_from_bc.zip")


if __name__ == "__main__":
    main()

