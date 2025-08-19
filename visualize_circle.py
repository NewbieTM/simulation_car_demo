import argparse
import time
import numpy as np
from stable_baselines3 import PPO
from circle_env import CircleDriveEnv

def play_expert(env, demos_path="demos_circle.npy", speed=0.02):
    demos = np.load(demos_path, allow_pickle=True)
    print(f"[INFO] Воспроизведение {len(demos)} демонстраций.")
    for ep_idx, ep in enumerate(demos):
        init_state = ep["initial_state"]
        actions = ep["actions"]
        env.set_state(init_state)
        obs = env._get_obs()  #
        env.render()
        time.sleep(0.05)
        for action in actions:
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(speed)
            if terminated or truncated:
                break
        time.sleep(0.2)

def play_model(env, model_path, speed=0.02):
    model = PPO.load(model_path)
    for ep in range(5):
        obs, _ = env.reset()
        env.render()
        time.sleep(0.05)
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            env.render()
            time.sleep(speed)
            if terminated or truncated:
                break
        time.sleep(0.2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, choices=["expert", "bc", "ppo"], required=True)
    args = parser.parse_args()

    env = CircleDriveEnv()
    if args.mode == "expert":
        play_expert(env)
    elif args.mode == "bc":
        play_model(env, "ppo_bc_pretrained.zip")
    elif args.mode == "ppo":
        play_model(env, "ppo_finetuned_from_bc.zip")

    env.close()
    print("[INFO] Визуализация завершена.")

if __name__ == "__main__":
    main()
