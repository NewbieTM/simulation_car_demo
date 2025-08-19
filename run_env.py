from circle_env import CircleDriveEnv
import time

env = CircleDriveEnv()
obs = env.reset()
for t in range(500):
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.02)
env.close()
