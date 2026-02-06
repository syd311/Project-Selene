import gymnasium as gym
from stable_baselines3 import PPO
import os
# Creating the environment 
# Using LunarLander-v2. It handles all the physics (gravity, thrust, inertia).
env = gym.make("LunarLander-v3", render_mode="rgb_array")
# Creating the brain by using th PPO algorithm 
model = PPO("MlpPolicy", env, verbose=1)
# Training the model 
print("Training started... ")
model.learn(total_timesteps=100000)
# Saving this brain for later use. 
model.save("neural_lander_brain")
print("Training Complete. Brain saved.")