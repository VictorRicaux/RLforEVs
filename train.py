from environment import ElectricVehicleEnv
from stable_baselines3 import PPO
import pandas as pd


prices = pd.read_csv('2023combined_np15_lmp_prices.csv')
prices['Timestamp'] = pd.to_datetime(prices['Timestamp'])
prices = prices.set_index('Timestamp')

# Resample the data to daily frequency and extract the price data for each day
daily_prices = prices['Price'].resample('D').apply(lambda x: x.values)

# Convert the daily prices to a list of arrays
price_data = [day_prices for day_prices in daily_prices if len(day_prices) == 24 * 12]

battery_capacity = 20

env = ElectricVehicleEnv(battery_capacity, price_data)
model = PPO("MultiInputPolicy", env, verbose=1)
model.learn(total_timesteps=100000)
model.save("ppo_electric_vehicle_daily")


