import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd


class EVChargingEnv(gym.Env):
    def __init__(self, price_data, charging_rate=0.1, battery_capacity=1.0):
        """
        Custom Environment for EV Charging and Discharging
        :param price_data: A pandas DataFrame containing 'Timestamp' and 'Price' columns.
        :param charging_rate: Maximum amount of energy the battery can charge or discharge per step.
        :param battery_capacity: Total capacity of the battery in arbitrary units.
        """
        super(EVChargingEnv, self).__init__()
        
        # Load and validate price data
        self.price_data = price_data
        self.num_steps = len(price_data)
        if 'Price' not in price_data.columns:
            raise ValueError("Price data must contain a 'Price' column.")
        
        # Battery parameters
        self.charging_rate = charging_rate
        self.battery_capacity = battery_capacity
        self.battery_level = 0.5 * battery_capacity  # Start at 50% capacity
        
        # Step counter
        self.current_step = 0
        
        # Action space: -1 (discharge), 0 (hold), 1 (charge)
        self.action_space = spaces.Discrete(3)
        
        # Observation space: [battery level, current price]
        self.observation_space = spaces.Box(
            low=np.array([0.0, price_data['Price'].min()]),
            high=np.array([battery_capacity, price_data['Price'].max()]),
            dtype=np.float32
        )
        
        # Initial observation
        self.state = np.array([self.battery_level, self.price_data['Price'].iloc[self.current_step]])

    def step(self, action):
        """
        Execute one time step within the environment.
        :param action: Action taken by the agent (0: discharge, 1: do nothing, 2: charge).
        :return: Tuple (obs, reward, terminated, truncated, info)
        """
        assert self.action_space.contains(action), f"Invalid action {action}"
        price = self.price_data['Price'].iloc[self.current_step]
        energy_change = self.charging_rate * (action - 1)  # action: 0 (discharge), 1 (do nothing), 2 (charge)
        self.battery_level = np.clip(self.battery_level + energy_change, 0.0, self.battery_capacity)
        reward = -energy_change * price
        self.current_step += 1
        
        # Check if the episode is terminated
        terminated = self.current_step >= self.num_steps
        truncated = False  # No time limits applied here

        # Update state if not terminated
        if not terminated:
            self.state = np.array([self.battery_level, self.price_data['Price'].iloc[self.current_step]], dtype=np.float32)
        else:
            self.state = None
        
        info = {}  # Additional information (can be expanded later if needed)
        return self.state, reward, terminated, truncated, info


    def reset(self, seed=None, options=None):
        """
        Reset the environment to the initial state.
        :param seed: Random seed for reproducibility.
        :param options: Additional options for the reset (not used here).
        :return: Tuple (initial observation, info)
        """
        if seed is not None:
            self.np_random, _ = gym.utils.seeding.np_random(seed)
        self.current_step = 0
        self.battery_level = 0.5 * self.battery_capacity
        # Ensure state is in float32
        self.state = np.array([self.battery_level, self.price_data['Price'].iloc[self.current_step]], dtype=np.float32)
        info = {}
        return self.state, info

    def render(self, mode='human'):
        """
        Render the environment (optional).
        """
        print(f"Step: {self.current_step}, Battery Level: {self.battery_level:.2f}, "
              f"Price: {self.price_data['Price'].iloc[self.current_step]:.2f}")

    def close(self):
        """
        Clean up resources (optional).
        """
        pass

