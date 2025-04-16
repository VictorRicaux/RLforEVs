import gymnasium as gym
from gymnasium import spaces
import numpy as np
from stable_baselines3 import PPO, DQN
from enum import Enum
import matplotlib.pyplot as plt

class Actions(Enum):
    DO_NOTHING = 0
    CHARGE = 1
    SELL = 2

class ElectricVehicleEnv(gym.Env):
    def __init__(self, battery_capacity, price_data, day = None):
        super(ElectricVehicleEnv, self).__init__()

        self.battery_capacity = battery_capacity 
        self.price_data = price_data
        self.action_space = spaces.Discrete(3)  # 0: do nothing, 1: charge, 2: sell
        self.observation_space = spaces.Dict({
            "battery_level": spaces.Box(low=0, high=self.battery_capacity, shape=(1,), dtype=float),
            "price": spaces.Box(low=-1.0, high=1.0, shape=(24 * 12,), dtype=float),
            "time_of_day": spaces.Discrete(24 * 12)
        })

        self.current_day = np.random.randint(0, len(price_data)) if day is None else day
        self.daily_price_data = price_data[self.current_day]
        self.current_step = 0
        self.battery_level = battery_capacity / 2  # Start at 50% capacity
        self.money = 0

        self._action_to_decision = {
            Actions.DO_NOTHING.value: 0,
            Actions.CHARGE.value: 1,
            Actions.SELL.value: -1
        }

    def _get_obs(self):
        # Use -1 as a sentinel for unobserved future prices
        sentinel = -1.0  
        price_obs = np.full(24 * 12, sentinel, dtype=float)
        price_obs[:self.current_step + 1] = self.daily_price_data[:self.current_step + 1]
        
        return {
            "battery_level": np.array([self.battery_level]),
            "price": price_obs,
            "time_of_day": self.current_step
        }
    
    def _get_info(self):
        return {
            "current_price": self.daily_price_data[self.current_step],
            "time_of_day": self.current_step
        }

    def reset(self, seed=None, day=None):
        super().reset(seed=seed)
        self.current_day = np.random.randint(0, len(self.price_data)) if day is None else day
        self.daily_price_data = self.price_data[self.current_day]
        self.current_step = 0
        self.battery_level = self.battery_capacity / 2
        self.money = 0
        return self._get_obs(), self._get_info()

    def step(self, action):
        action = int(action)
        decision = self._action_to_decision[action]
        current_price = self.daily_price_data[self.current_step]
        
        # Calculate the potential (net worth) before taking the action.
        net_worth_before = self.money + self.battery_level * current_price
        
        # Update the battery and money based on the action.
        if decision > 0 and self.battery_level < self.battery_capacity:
            self.money -= current_price
        elif decision < 0 and self.battery_level > 0:
            self.money += current_price
        self.battery_level = np.clip(self.battery_level + decision, 0, self.battery_capacity)
        
        # Move to the next time step.
        self.current_step += 1
        terminated = self.current_step == len(self.daily_price_data)
        
        # Use the price at the next time step if available, otherwise keep current price.
        if not terminated:
            next_price = self.daily_price_data[self.current_step]
        else:
            next_price = current_price
        
        # Calculate the potential (net worth) after taking the action.
        net_worth_after = self.money + self.battery_level * next_price
        
        # The shaped reward is the change in net worth.
        reward = net_worth_after - net_worth_before
        
        observation = self._get_obs()
        info = {"current_price": current_price, "time_of_day": self.current_step}
        
        return observation, reward, terminated, False, info

