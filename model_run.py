import pandas as pd
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_checker import check_env
from EVChargingEnv import EVChargingEnv
from stable_baselines3.common.monitor import Monitor


# Load price data
data = pd.read_csv('2023combined_np15_lmp_prices.csv')
data['Timestamp'] = pd.to_datetime(data['Timestamp'])


# Create the environment
env = EVChargingEnv(price_data=data)
monitored_env = Monitor(env)

# Check the environment for errors
check_env(monitored_env, warn=True)

# Define the DQN model
model = DQN(
    "MlpPolicy",  # Use a Multi-Layer Perceptron policy
    monitored_env,
    learning_rate=1e-3,
    buffer_size=50000,
    learning_starts=1000,
    batch_size=64,
    gamma=0.99,  # Discount factor
    tau=0.005,  # Soft update coefficient
    target_update_interval=10,  # Update the target network every 10 steps
    train_freq=4,
    gradient_steps=1,
    verbose=1,
    seed=42,
)

# Train the model
timesteps = 10000
print(f"Training the DQN agent for {timesteps} timesteps...")
model.learn(total_timesteps=timesteps)

# Save the trained model
model.save("dqn_ev_charging")

# Load the model (optional, for testing)
model = DQN.load("dqn_ev_charging", env=monitored_env)

# Evaluate the trained model
mean_reward, std_reward = evaluate_policy(model, monitored_env, n_eval_episodes=10, deterministic=True)
print(f"Mean reward: {mean_reward:.2f}, Std reward: {std_reward:.2f}")

# Test the trained agent
obs = monitored_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = monitored_env.step(action)
    env.render()

print("Testing completed!")
