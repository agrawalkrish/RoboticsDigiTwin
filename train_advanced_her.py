import gymnasium as gym
import panda_gym
import numpy as np  # Added numpy for the calculation
from stable_baselines3 import SAC, HerReplayBuffer
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os
import multiprocessing

def main():
    # 1. Setup Directories
    models_dir = "models/SAC_HER"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # 2. Configuration
    # Determine CPU cores: uses mostly all cores but leaves 2 free for your OS
    num_cpu = max(1, multiprocessing.cpu_count() - 2) 
    print(f"Using {num_cpu} parallel environments.")
    
    TOTAL_TIMESTEPS = 800000
    SAVE_FREQ = 20000

    # 3. Create Parallel Environments
    # We use SubprocVecEnv to run physics on different CPU cores simultaneously
    env = make_vec_env(
        "PandaPickAndPlace-v3", 
        n_envs=num_cpu, 
        vec_env_cls=SubprocVecEnv
    )

    # 4. Normalize Inputs
    # Critical for convergence in robotics
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # --- UPDATED SECTION: CALCULATE TARGET ENTROPY ---
    # Get the number of actions (Panda robot usually has 4: x, y, z, gripper)
    n_actions = env.action_space.shape[0]
    
    # Default is usually -1 * n_actions (which would be -4).
    # We multiply by 0.5 to force it to keep more randomness (-2).
    target_entropy_val = -0.5 * n_actions
    
    print(f"Action space dimension: {n_actions}")
    print(f"Setting Target Entropy to: {target_entropy_val} (Prevents early collapse)")
    # -------------------------------------------------

    # 5. Define the Advanced Model (SAC + HER)
    model = SAC(
        "MultiInputPolicy",
        env,
        # Reduced buffer size to 100k to prevent RAM overflow with parallel envs.
        buffer_size=100000,  
        learning_starts=1000, 
        batch_size=256,
        
        # --- ENTROPY FIX APPLIED HERE ---
        ent_coef='auto',              # Keep auto-tuning ON
        target_entropy=target_entropy_val, # Pass our calculated target
        # --------------------------------
        
        gamma=0.95,
        replay_buffer_class=HerReplayBuffer,
        # HER Strategy: "future" replaying
        replay_buffer_kwargs=dict(
            n_sampled_goal=4,
            goal_selection_strategy="future",
        ),
        verbose=1,
        tensorboard_log=log_dir,
    )

    print("Starting Advanced Training (SAC + HER)...")
    print(f"Target: {TOTAL_TIMESTEPS} steps.")

    # Callback to save model periodically
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=models_dir,
        name_prefix="SAC_HER_model"
    )

    # 6. Train
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS, 
            callback=checkpoint_callback,
            log_interval=4
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted manually.")
        # We save even if interrupted
        model.save(f"{models_dir}/SAC_HER_interrupted")
        env.save(f"{models_dir}/vec_normalize_interrupted.pkl")
        print("Interrupted model saved.")
        exit()

    # 7. Save Final Model
    model.save(f"{models_dir}/SAC_HER_final")
    env.save(f"{models_dir}/vec_normalize_final.pkl")

    print("Training Complete!")
    env.close()

if __name__ == '__main__':
    # This check is REQUIRED for multiprocessing
    main()