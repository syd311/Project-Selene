import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import numpy as np
import os
import glob
subfolder = "project_selene"
brain_files = glob.glob(os.path.join(subfolder, "neural_lander_FINAL*.zip"))
if not brain_files:
    brain_files = glob.glob("neural_lander_FINAL*.zip")

if not brain_files:
    print(" Error: Can't find neural_lander_FINAL.zip")
    exit()

brain_path = brain_files[0].replace(".zip", "")
print(f" Testing Brain: {brain_path}")


env = gym.make("LunarLander-v3", render_mode=None)
model = PPO.load(brain_path, env=env)
print("Running 100 test flights to get real accuracy... (Wait ~10 secs)")
rewards, episode_lengths = evaluate_policy(
    model, 
    env, 
    n_eval_episodes=100, 
    return_episode_rewards=True
)

mean_reward = np.mean(rewards)
std_reward = np.std(rewards)
success_rate = np.sum(np.array(rewards) >= 200) / 100 * 100 
print("\n" + "="*30)
print("RESULTS:")
print("="*30)
print(f"| Metric | Result |")
print(f"| :--- | :--- |")
print(f"| **Success Rate** | **{success_rate:.1f}%** |")
print(f"| **Average Reward** | **{mean_reward:.0f} +/- {std_reward:.0f}** |")
print("="*30)