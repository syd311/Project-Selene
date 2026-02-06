import gymnasium as gym
from stable_baselines3 import PPO
import os
import glob
print(" Looking for the dumb brain...")
found_files = glob.glob("neural_lander_brain*.zip")
if not found_files:
    found_files = glob.glob("project_selene/neural_lander_brain*.zip")

if not found_files:
    print("Can't find the old brain to improve.")
    exit()

brain_file = found_files[0].replace(".zip", "")
print(f"Loading: {brain_file}")

env = gym.make("LunarLander-v3", render_mode=None) # No render = faster training

# Load the model and Train MORE
model = PPO.load(brain_file, env=env)

print(" Training for 200,000 more steps...")
model.learn(total_timesteps=200000)
model.save("neural_lander_brain_SMART")
print("Training Complete!")