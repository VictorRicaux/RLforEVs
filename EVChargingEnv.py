import gymnasium as gym
from gymnasium import spaces
import numpy as np
from datetime import timedelta

class EVChargingEnv(gym.Env):
    def __init__(self, price_data):
        super(EVChargingEnv, self).__init__()
        self.price_data = price_data
        self.battery_level = 1.0  # Start with a full battery for simplicity

        # Define action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: charge, 2: discharge
        self.observation_space = spaces.Box(low=np.array([0.2, 0.0], dtype=np.float32),
                                            high=np.array([1.0, np.inf], dtype=np.float32),
                                            dtype=np.float32)
        self.np_random = None  # For random number generation

        # Time parameters for the daily simulation
        self.start_time = price_data['Timestamp'].iloc[0]
        self.current_time = self.start_time
        self.time_step = 0
        self.charge_intervals = [(timedelta(hours=7), timedelta(hours=10)),
                                 (timedelta(hours=16), timedelta(hours=18))]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)
        
        # Reset battery level and time
        self.battery_level = 1.0
        self.time_step = 0
        self.current_time = self.start_time
        
        # Return the initial observation
        initial_observation = np.array([self.battery_level, self.price_data['Price'].iloc[self.time_step]], dtype=np.float32)
        return initial_observation, {}

    def step(self, action):
        # Update battery based on action, but only if the car is not in use
        in_use = any(start <= self.current_time.time() < end for start, end in self.charge_intervals)
        if not in_use:
            if action == 1:  # charge
                self.battery_level = min(self.battery_level + 0.1, 1.0)
            elif action == 2:  # discharge
                self.battery_level = max(self.battery_level - 0.1, 0.2)

        # Update time step and check for end of day
        self.time_step += 1
        self.current_time += timedelta(minutes=5)
        terminated = self.time_step >= len(self.price_data) - 1

        # Reward: encourage charging when prices are low, penalize discharging when prices are high
        reward = -self.price_data['Price'].iloc[self.time_step] if action == 1 else self.price_data['Price'].iloc[self.time_step]

        # Construct next observation
        observation = np.array([self.battery_level, self.price_data['Price'].iloc[self.time_step]], dtype=np.float32)

        # Info for debugging or additional logging
        info = {}

        return observation, reward, terminated, False, info

    def render(self, mode='human'):
        print(f"Time: {self.current_time}, Battery Level: {self.battery_level:.2f}, Price: {self.price_data['Price'].iloc[self.time_step]}")
