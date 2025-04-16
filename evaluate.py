from environment import ElectricVehicleEnv
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import pandas as pd

prices = pd.read_csv('2023combined_np15_lmp_prices.csv')
prices['Timestamp'] = pd.to_datetime(prices['Timestamp'])
prices = prices.set_index('Timestamp')

daily_prices = prices['Price'].resample('D').apply(lambda x: x.values)

price_data = [day_prices for day_prices in daily_prices if len(day_prices) == 24 * 12]
battery_capacity = 20
test_day_index = 30
env = ElectricVehicleEnv(battery_capacity=battery_capacity, price_data=price_data, day=test_day_index)
model = PPO.load("ppo_electric_vehicle_daily")

obs, info = env.reset(day=test_day_index)
money, battery_levels, money_current = [], [], 0

for _ in range(24 * 12):
    action, _states = model.predict(obs)
    obs, reward, done, _, info = env.step(action)

    current_price = info.get("current_price", None)
    current_battery_level = obs["battery_level"][0]
   
    if action == 1 and current_battery_level < battery_capacity:
        money_current -= current_price
    elif action == 2 and current_battery_level > 0:
        money_current += current_price
    money.append(money_current)
    battery_levels.append(obs["battery_level"][0])
    if done:
        obs, info = env.reset(day=test_day_index)

test_day_prices = price_data[test_day_index]
time_index = pd.date_range(start='00:00', periods=24*12, freq='5min')
plt.figure(figsize=(18, 6))

plt.subplot(1, 3, 1)
plt.plot(time_index, battery_levels)
plt.title("Battery Level")
plt.xlabel("Time")
plt.ylabel("Battery Level")
plt.xticks(rotation=45)

plt.subplot(1, 3, 2)
plt.plot(time_index, money)
plt.title("Rewards")
plt.xlabel("Time")
plt.ylabel("Reward")
plt.xticks(rotation=45)

plt.subplot(1, 3, 3)
plt.plot(time_index, test_day_prices)
plt.title("Price Data")
plt.xlabel("Time")
plt.ylabel("Price")
plt.xticks(rotation=45)

plt.tight_layout()
plt.show()

print(f"Accumulated Reward: {money_current:.2f}")
