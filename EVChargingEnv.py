import numpy as np
import gym
from gym import spaces

class EVChargingEnv(gym.Env):
    def __init__(self, price_data, usage_prob_data, initial_battery=0.5, battery_min=0.2, battery_max=1.0):
        super(EVChargingEnv, self).__init__()
        
        # Load data (electricity prices and usage probabilities)
        self.price_data = price_data  # Price for each time step (c_t)
        self.usage_prob_data = usage_prob_data  # Probability of EV usage at each time step (p_t)
        self.time_step = 0
        
        # Battery parameters
        self.initial_battery = initial_battery
        self.battery_min = battery_min
        self.battery_max = battery_max
        self.battery_level = initial_battery
        
        # Action space: 0 = hold, 1 = charge, 2 = discharge
        self.action_space = spaces.Discrete(3)
        
        # State space: [battery_level, electricity_price, usage_probability]
        self.observation_space = spaces.Box(low=np.array([battery_min, 0.0, 0.0]), 
                                            high=np.array([battery_max, np.inf, 1.0]), dtype=np.float32)
        
    def reset(self):
        """ Reset the environment to the initial state """
        self.time_step = 0
        self.battery_level = self.initial_battery
        return np.array([self.battery_level, self.price_data[self.time_step], self.usage_prob_data[self.time_step]])
    
    def step(self, action):
        """ Execute the action and return new state, reward, done """
        price = self.price_data[self.time_step]
        usage_prob = self.usage_prob_data[self.time_step]
        
        # Reward components
        cost = 0
        penalty = 0
        
        # Actions: 0 = hold, 1 = charge, 2 = discharge
        if action == 1:  # Charge
            if self.battery_level < self.battery_max:
                self.battery_level = min(self.battery_level + 0.1, self.battery_max)
                cost = price  # Pay for charging
        elif action == 2:  # Discharge
            if self.battery_level > self.battery_min:
                self.battery_level = max(self.battery_level - 0.1, self.battery_min)
                cost = -price  # Gain from selling electricity
        
        # Penalty if battery is below minimum during high usage probability
        if self.battery_level < self.battery_min and usage_prob > 0.5:
            penalty = -10
        
        # Reward: minimize cost and avoid low battery penalties
        reward = -cost + penalty
        
        # Move to next time step
        self.time_step += 1
        done = self.time_step >= len(self.price_data) - 1
        
        # Next state
        next_state = np.array([self.battery_level, self.price_data[self.time_step], self.usage_prob_data[self.time_step]])
        
        return next_state, reward, done, {}
    
    def render(self, mode='human'):
        print(f"Time: {self.time_step}, Battery Level: {self.battery_level}, Price: {self.price_data[self.time_step]}")