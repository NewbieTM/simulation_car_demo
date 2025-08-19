import math, numpy as np
import gymnasium as gym
from gymnasium import spaces
import pygame

class CircleDriveEnv(gym.Env):
    metadata = {"render_fps": 60}

    def __init__(self, radius=10.0, dt=0.05, max_speed=5.0, reward_type="distance"):
        super().__init__()
        self.radius = radius
        self.dt = dt
        self.max_speed = max_speed
        self.observation_space = spaces.Box(low=-100.0, high=100.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.reward_type = reward_type

        self.screen = None
        self.clock = None

        self.np_random = np.random.default_rng()

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.v = 0.0

    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.np_random = np.random.default_rng(seed)
        ang = float(self.np_random.uniform(0, 2*math.pi))
        self.x = (self.radius + float(self.np_random.uniform(-0.2, 0.2))) * math.cos(ang)
        self.y = (self.radius + float(self.np_random.uniform(-0.2, 0.2))) * math.sin(ang)
        self.theta = ang + math.pi/2 + float(self.np_random.uniform(-0.2, 0.2))
        self.v = float(self.np_random.uniform(0.5*self.max_speed, 0.8*self.max_speed))
        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        action = np.array(action, dtype=np.float32).flatten()
        if action.size == 1:
            steer, throttle = float(action[0]), 0.0
        else:
            steer, throttle = float(action[0]), float(action[1])
        L = 1.0
        self.theta += (self.v / L) * math.tan(steer*0.5) * self.dt
        self.v = np.clip(self.v + throttle * 1.0 * self.dt, 0.0, self.max_speed)
        self.x += self.v * math.cos(self.theta) * self.dt
        self.y += self.v * math.sin(self.theta) * self.dt

        obs = self._get_obs()
        reward = self._compute_reward(obs, action)
        terminated = False
        truncated = False
        info = {"cross_track": self._cross_track_error()}
        return obs, float(reward), terminated, truncated, info

    def _get_obs(self):
        return np.array([self.x, self.y, math.cos(self.theta), math.sin(self.theta), self.v], dtype=np.float32)

    def _cross_track_error(self):
        r = math.hypot(self.x, self.y)
        return r - self.radius

    def _compute_reward(self, obs, action):
        """
        Варианты reward.
        'distance_abs'  : -|cte|
        'distance_sq'   : -(cte**2)
        'aligned'       : -alpha*|cte| - beta*|heading_err| + gamma*(v/target_v)
        'sparse'
        """
        x, y, cos_t, sin_t, v = obs
        r = math.hypot(x, y)
        cte = r - self.radius
        ang_radial = math.atan2(y, x)
        ang_tangent = ang_radial + math.pi / 2
        theta = math.atan2(sin_t, cos_t)
        heading_err = (theta - ang_tangent + math.pi) % (2 * math.pi) - math.pi

        if self.reward_type == "distance_abs":
            return -abs(cte)
        elif self.reward_type == "distance_sq":
            return -(cte ** 2)
        elif self.reward_type == "aligned":
            alpha = 1.0
            beta = 0.5
            gamma = 0.01
            return -alpha * abs(cte) - beta * abs(heading_err) + gamma * (v / (self.max_speed + 1e-8))
        elif self.reward_type == "sparse":
            return 1.0 if abs(cte) < 0.2 and abs(heading_err) < 0.5 else -0.01
        else:
            return -abs(cte)

    def get_state(self):
        return {
            "x": float(self.x),
            "y": float(self.y),
            "theta": float(self.theta),
            "v": float(self.v),
        }

    def set_state(self, state: dict):
        self.x = float(state["x"])
        self.y = float(state["y"])
        self.theta = float(state["theta"])
        self.v = float(state["v"])

    def render(self):
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((600,600))
            self.clock = pygame.time.Clock()
        self.screen.fill((255,255,255))
        cx, cy = 300, 300
        scale = 20
        pygame.draw.circle(self.screen, (220,220,220), (cx,cy), int(self.radius*scale), 2)
        car_x = int(cx + self.x*scale)
        car_y = int(cy - self.y*scale)
        rect = pygame.Rect(0,0,14,8)
        rect.center = (car_x, car_y)
        car_surf = pygame.Surface(rect.size, pygame.SRCALPHA)
        pygame.draw.polygon(car_surf, (255,0,0), [(0,rect.height//2),(rect.width,0),(rect.width,rect.height)])
        rotated = pygame.transform.rotate(car_surf, -math.degrees(self.theta))
        rrect = rotated.get_rect(center=(car_x,car_y))
        self.screen.blit(rotated, rrect.topleft)
        pygame.display.flip()
        self.clock.tick(50)

    def close(self):
        if self.screen:
            pygame.quit()
            self.screen = None
