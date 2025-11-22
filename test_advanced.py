import gymnasium as gym
import panda_gym
from stable_baselines3 import SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize
import time

# 1. Load Environment
env = make_vec_env("PandaPickAndPlace-v3", n_envs=1, env_kwargs={"render_mode": "human"})

# 2. Load Normalization Stats
stats_path = "models/SAC_HER/vec_normalize_final.pkl"
env = VecNormalize.load(stats_path, env)
env.training = False
env.norm_reward = False

# 3. Load the Model
model_path = "models/SAC_HER/SAC_HER_final.zip"
model = SAC.load(model_path, env=env)

print("---------------------------------------")
print("Loaded SAC + HER Model.")
print("Task: Pick Up the Cube and Place it.")
print("---------------------------------------")

obs = env.reset()
total_episodes = 0
success_episodes = 0

try:
    while True:
        # Predict action
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)
        
        # Slower visualization for complex tasks so you can see the grasping
        time.sleep(0.005)

        if done[0]:
            total_episodes += 1
            is_success = info[0].get("is_success", False)
            
            if is_success:
                success_episodes += 1
                print(f"Episode {total_episodes}: ✅ GRASP SUCCESSFUL!")
            else:
                print(f"Episode {total_episodes}: ❌ FAILED (Dropped or Missed)")
            
            obs = env.reset()

except KeyboardInterrupt:
    print("\nStopped.")

env.close()