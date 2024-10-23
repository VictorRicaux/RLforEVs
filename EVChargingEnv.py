import gymnasium as gym
from gymnasium import spaces
import numpy as np

class EVChargingEnv(gym.Env):
    def __init__(self, price_data, usage_prob_data):
        super(EVChargingEnv, self).__init__()
        self.price_data = price_data
        self.usage_prob_data = usage_prob_data
        self.battery_level = 0.5

        # Define the action and observation spaces
        self.action_space = spaces.Discrete(3)  # 0: hold, 1: charge, 2: discharge
        self.observation_space = spaces.Box(low=np.array([0.2, 0.0, 0.0], dtype=np.float32),
                                            high=np.array([1.0, np.inf, 1.0], dtype=np.float32),
                                            dtype=np.float32)
        self.np_random = None  # For random number generation

    def reset(self, seed=None, options=None):
        # Set the random seed
        super().reset(seed=seed)
        self.np_random, _ = gym.utils.seeding.np_random(seed)

        # Reset the environment's state
        self.battery_level = 0.5
        self.time_step = 0
        
        # Return the initial observation and cast it to float32
        initial_observation = np.array([self.battery_level, self.price_data[self.time_step], self.usage_prob_data[self.time_step]], dtype=np.float32)
        return initial_observation, {}

    def step(self, action):
        # Example logic for updating the environment based on the action
        if action == 1:  # charge
            self.battery_level = min(self.battery_level + 0.1, 1.0)
        elif action == 2:  # discharge
            self.battery_level = max(self.battery_level - 0.1, 0.2)
        
        self.time_step += 1

        # Check if we reached the end of the price data
        terminated = self.time_step >= len(self.price_data) - 1
        truncated = False  # No truncation logic in this example

        # Calculate reward (example reward function)
        reward = -self.price_data[self.time_step] if action == 1 else self.price_data[self.time_step]

        # Construct the observation (next state)
        observation = np.array([self.battery_level, self.price_data[self.time_step], self.usage_prob_data[self.time_step]], dtype=np.float32)

        # Info can be an empty dictionary or contain useful debugging info
        info = {}

        return observation, reward, terminated, truncated, info                                                 


    def render(self, mode='human'):
        print(f"Time: {self.time_step}, Battery Level: {self.battery_level}, Price: {self.price_data[self.time_step]}")
