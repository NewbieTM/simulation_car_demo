import numpy as np
from circle_env import CircleDriveEnv
import math

def expert_action(obs, radius=10.0, target_speed=2.0, L=1.0):
    x, y, cos_t, sin_t, v = obs
    r = math.hypot(x, y)
    theta = math.atan2(sin_t, cos_t)
    ang_radial = math.atan2(y, x)
    ang_tangent = ang_radial + math.pi/2

    cross_track = r - radius
    heading_err = (ang_tangent - theta + math.pi) % (2*math.pi) - math.pi

    base_steer = math.atan(L / radius)
    Kp_cte = 1.0
    steer = np.clip(base_steer + Kp_cte * cross_track + 0.5 * heading_err, -1.0, 1.0)

    throttle = np.clip(target_speed - v, -1.0, 1.0)
    return np.array([steer, throttle], dtype=np.float32)



def collect(save_path="demos_circle.npy", bc_obs_path="bc_obs.npy", bc_act_path="bc_actions.npy",
            n_eps=200, max_steps=500):
    env = CircleDriveEnv()
    demos = []
    bc_obs = []
    bc_acts = []

    for ep in range(n_eps):
        obs, _ = env.reset()
        init_state = env.get_state()
        obs_list = []
        act_list = []

        for t in range(max_steps):
            action = expert_action(obs)
            obs_list.append(obs.copy())
            act_list.append(action.copy())

            bc_obs.append(obs.copy())
            bc_acts.append(action.copy())

            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                break

        demos.append({
            "initial_state": init_state,
            "obs": np.stack(obs_list),
            "actions": np.stack(act_list),
        })

    np.save(save_path, demos, allow_pickle=True)
    np.save(bc_obs_path, np.stack(bc_obs), allow_pickle=False)
    np.save(bc_act_path, np.stack(bc_acts), allow_pickle=False)
    print(f"[collect_demos] Saved {len(demos)} episodes to {save_path}")
    print(f"[collect_demos] Saved BC pairs: {bc_obs_path} {bc_act_path}")

if __name__ == "__main__":
    collect()
