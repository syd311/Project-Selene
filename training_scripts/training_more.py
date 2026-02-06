import gymnasium as gym
from stable_baselines3 import PPO
import os


env = gym.make("LunarLander-v3", render_mode=None)
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=0.0003,
    n_steps=1024,        # Updating the brain more often
    batch_size=64,       # Learn from smaller batches of experience
    n_epochs=4,
    gamma=0.999,       
    gae_lambda=0.98,
    ent_coef=0.01,      
    verbose=1
)

print("TRAINING PROTOCOL...")

model.learn(total_timesteps=500000)

model.save("neural_lander_ACE")
print("Training Complete. Saved as 'neural_lander_ACE.zip'")