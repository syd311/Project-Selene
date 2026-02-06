import gymnasium as gym
from stable_baselines3 import PPO
from gymnasium.wrappers import RecordVideo
import matplotlib.pyplot as plt
import mplcyberpunk 
import numpy as np
import os
import glob
subfolder = "project_selene"
brain_name_pattern = "neural_lander_FINAL*.zip"

print(f"Looking...")

found_files = glob.glob(brain_name_pattern)
if not found_files:
    search_path = os.path.join(subfolder, brain_name_pattern)
    found_files = glob.glob(search_path)

if not found_files:
    print("\n ERROR: Could not find 'neural_lander_FINAL.zip'.")
    exit()

actual_filename = found_files[0]
print(f"Found: '{actual_filename}'")
brain_path = actual_filename.replace(".zip", "")
if brain_path.endswith(".zip"):
    brain_path = brain_path.replace(".zip", "")

os.makedirs("./mission_footage", exist_ok=True)
env = gym.make("LunarLander-v3", render_mode="rgb_array")

env = RecordVideo(
    env, 
    video_folder="./mission_footage", 
    name_prefix="project_selene_ace_landing",
    episode_trigger=lambda x: x == 0
)

print(" Loading Neural Network...")
model = PPO.load(brain_path)

print(" Rolling camera... (Simulating landing)")
obs, _ = env.reset()

velocity_data = []
altitude_data = []

while True:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Telemetry
    altitude_data.append(obs[1])
    total_speed = np.sqrt(obs[2]**2 + obs[3]**2)
    velocity_data.append(total_speed)

    if terminated or truncated:
        break

env.close()
print(" Generating Final Dashboard...")
plt.style.use("cyberpunk")
fig, ax = plt.subplots(figsize=(10, 5))

ax.plot(velocity_data, label='Velocity (Mach)', color='#00ff41') # Matrix Green
ax.plot(altitude_data, label='Altitude', color='#00d9ff') # Cyan
ax.legend(loc='upper right', fontsize=10, frameon=True, facecolor='#2A3459', edgecolor='white')
mplcyberpunk.add_glow_effects()

plt.title("PROJECT SELENE", fontsize=16, color='white', fontweight='bold')
plt.xlabel("Mission Time")
plt.ylabel("Telemetry")
plt.grid(color='#2A3459', linestyle='--', linewidth=0.5)

output_path = "./mission_footage/final_dashboard.png"
plt.savefig(output_path, dpi=300)
print(f" Done!")