import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import os
import glob
subfolder = "project_selene"
brain_files = glob.glob("neural_lander_brain*.zip")
if not brain_files:
    print(f"Not found in root. Checking inside '{subfolder}'...")
    search_path = os.path.join(subfolder, "neural_lander_brain*.zip")
    brain_files = glob.glob(search_path)

if not brain_files:
    print("\n CRITICAL ERROR: No brain file found anywhere.")
    exit()
latest_brain = max(brain_files, key=os.path.getctime)
print(f"Loading brain: {latest_brain}")
brain_name_clean = latest_brain.replace(".zip", "")
if brain_name_clean.endswith(".zip"): 
    brain_name_clean = brain_name_clean.replace(".zip", "")
try:
    env = gym.make("LunarLander-v3", render_mode=None)
except:
    print("⚠️ v3 not found, using v2.")
    env = gym.make("LunarLander-v2", render_mode=None)

model = PPO.load(brain_name_clean, env=env)

print("Starting Mastery Training...")
iteration = 0

while True:
    iteration += 1
    
    # Train for 50k steps
    print(f"\n--- Round {iteration}: Training for 50,000 steps ---")
    model.learn(total_timesteps=50000)
    
    # Test (Exam)
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Exam Score: {mean_reward:.2f}")
    
    # Check Pass/Fail
    if mean_reward >= 200:
        print(" SUCCESS! The AI has mastered landing!")
        model.save("neural_lander_MASTER")
        break
    elif mean_reward > 0:
        print("Progress! It's hovering/surviving. Keep going.")
    else:
        print("Still crashing. Needs more time.")
        
    # Save checkpoint
    checkpoint_name = f"neural_lander_checkpoint_{iteration}"
    model.save(checkpoint_name)
    print(f"   (Saved checkpoint: {checkpoint_name})")

print("DONE. Master Brain saved as 'neural_lander_MASTER.zip'")