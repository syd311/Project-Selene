import gymnasium as gym
from stable_baselines3 import PPO
import os
import glob

subfolder = "project_selene"
brain_files = glob.glob(os.path.join(subfolder, "neural_lander_ACE*.zip"))

if not brain_files:
    print("ERROR: Could not find a brain to polish.")
    exit()

latest_brain = max(brain_files, key=os.path.getctime)
brain_path = latest_brain.replace(".zip", "")
env = gym.make("LunarLander-v3", render_mode=None) # No render = faster training
model = PPO.load(brain_path, env=env)

model.learning_rate = 0.00005 
model.ent_coef = 0.0  

print("STARTING POLISHING PHASE...")

model.learn(total_timesteps=100000)

model.save("neural_lander_FINAL")
print(" DONE. ")