"""
PPO Training Script for Adaptive E-Learning Environment

Proximal Policy Optimization with multiple hyperparameter configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AdaptiveELearningEnv
import torch
import pandas as pd


class TrainingCallback(BaseCallback):
    """Callback for tracking training progress"""
    
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.current_episode_reward = 0
        self.current_episode_length = 0
    
    def _on_step(self) -> bool:
        self.current_episode_reward += self.locals['rewards'][0]
        self.current_episode_length += 1
        
        if self.locals['dones'][0]:
            self.episode_rewards.append(self.current_episode_reward)
            self.episode_lengths.append(self.current_episode_length)
            self.current_episode_reward = 0
            self.current_episode_length = 0
        
        return True


def train_ppo(
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    clip_range: float,
    ent_coef: float,
    n_steps: int,
    batch_size: int,
    total_timesteps: int,
    config_name: str,
    save_dir: str = "models/ppo"
):
    """Train a PPO agent with specified hyperparameters"""
    
    print(f"\n{'='*60}")
    print(f"Training PPO: {config_name}")
    print(f"{'='*60}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"GAE Lambda: {gae_lambda}")
    print(f"Clip Range: {clip_range}")
    print(f"Entropy Coef: {ent_coef}")
    print(f"N Steps: {n_steps}")
    print(f"Batch Size: {batch_size}")
    
    # Create environment
    env = AdaptiveELearningEnv()
    env = Monitor(env)
    
    # Create callback
    callback = TrainingCallback()
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        clip_range=clip_range,
        ent_coef=ent_coef,
        n_steps=n_steps,
        batch_size=batch_size,
        n_epochs=10,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train
    print("\nTraining...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{config_name}.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Calculate statistics
    episode_rewards = callback.episode_rewards
    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards[-100:])
        std_reward = np.std(episode_rewards[-100:])
        max_reward = np.max(episode_rewards)
        convergence_episode = np.argmax(np.convolve(
            episode_rewards, np.ones(10)/10, mode='valid'
        )) if len(episode_rewards) > 10 else len(episode_rewards)
    else:
        mean_reward = std_reward = max_reward = convergence_episode = 0
    
    results = {
        'config_name': config_name,
        'learning_rate': learning_rate,
        'gamma': gamma,
        'gae_lambda': gae_lambda,
        'clip_range': clip_range,
        'ent_coef': ent_coef,
        'n_steps': n_steps,
        'batch_size': batch_size,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'num_episodes': len(episode_rewards),
        'convergence_episode': convergence_episode,
        'episode_rewards': episode_rewards,
        'episode_lengths': callback.episode_lengths
    }
    
    print(f"\nResults:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    
    env.close()
    return results


def main():
    """Main training function"""
    
    print("="*60)
    print("PPO Training for Adaptive E-Learning Platform")
    print("="*60)
    
    configs = [
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_baseline'},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_high_lr'},
        {'learning_rate': 3e-4, 'gamma': 0.95, 'gae_lambda': 0.95, 'clip_range': 0.2, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_low_gamma'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.9, 'clip_range': 0.2, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_low_gae'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.3, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_high_clip'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.1, 
         'ent_coef': 0.01, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_low_clip'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 
         'ent_coef': 0.05, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_high_ent'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'clip_range': 0.2, 
         'ent_coef': 0.001, 'n_steps': 2048, 'batch_size': 64, 'name': 'ppo_low_ent'},
        {'learning_rate': 5e-4, 'gamma': 0.98, 'gae_lambda': 0.92, 'clip_range': 0.25, 
         'ent_coef': 0.02, 'n_steps': 1024, 'batch_size': 128, 'name': 'ppo_balanced'},
        {'learning_rate': 1e-4, 'gamma': 0.99, 'gae_lambda': 0.98, 'clip_range': 0.15, 
         'ent_coef': 0.005, 'n_steps': 4096, 'batch_size': 256, 'name': 'ppo_conservative'}
    ]
    
    total_timesteps = 50000
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\nTraining Configuration {i}/10")
        
        results = train_ppo(
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            clip_range=config['clip_range'],
            ent_coef=config['ent_coef'],
            n_steps=config['n_steps'],
            batch_size=config['batch_size'],
            total_timesteps=total_timesteps,
            config_name=config['name']
        )
        
        all_results.append(results)
    
    # Find best
    best_config = max(all_results, key=lambda x: x['mean_reward'])
    print(f"\n\n{'='*60}")
    print("BEST PPO CONFIGURATION")
    print(f"{'='*60}")
    print(f"Name: {best_config['config_name']}")
    print(f"Mean Reward: {best_config['mean_reward']:.2f}")
    
    # Save results
    results_df = pd.DataFrame([{
        'Config': r['config_name'],
        'Learning Rate': r['learning_rate'],
        'Gamma': r['gamma'],
        'GAE Lambda': r['gae_lambda'],
        'Clip Range': r['clip_range'],
        'Entropy Coef': r['ent_coef'],
        'Mean Reward': r['mean_reward']
    } for r in all_results])
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/ppo_results.csv', index=False)
    
    # Plot
    plot_training_results(all_results, best_config)
    
    print("\n" + "="*60)
    print("PPO Training Completed Successfully!")
    print("="*60)


def plot_training_results(all_results, best_config):
    """Plot training results"""
    
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for result in all_results:
        rewards = result['episode_rewards']
        if len(rewards) > 0:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed, alpha=0.6, label=result['config_name'])
    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title('PPO: Cumulative Rewards')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    rewards = best_config['episode_rewards']
    if len(rewards) > 0:
        plt.plot(rewards, alpha=0.3, color='blue', label='Raw')
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, color='red', linewidth=2, label='Smoothed')
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Best Config: {best_config["config_name"]}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    configs = [r['config_name'] for r in all_results]
    means = [r['mean_reward'] for r in all_results]
    x = np.arange(len(configs))
    plt.bar(x, means, alpha=0.7)
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean Reward')
    plt.title('PPO: Mean Reward Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 4)
    convergence = [r['convergence_episode'] for r in all_results]
    plt.bar(x, convergence, alpha=0.7, color='green')
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Episodes to Converge')
    plt.title('PPO: Convergence Speed')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/ppo_training_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: results/plots/ppo_training_results.png")
    plt.close()


if __name__ == "__main__":
    main()