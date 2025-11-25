"""
DQN Training Script for Adaptive E-Learning Environment

This script trains Deep Q-Network agents with different hyperparameter configurations
to find optimal policies for adapting the e-learning interface and content.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AdaptiveELearningEnv
import torch
from tqdm import tqdm
import pandas as pd
from datetime import datetime


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


def train_dqn(
    learning_rate: float,
    gamma: float,
    buffer_size: int,
    batch_size: int,
    exploration_fraction: float,
    total_timesteps: int,
    config_name: str,
    save_dir: str = "models/dqn"
):
    """
    Train a DQN agent with specified hyperparameters
    
    Args:
        learning_rate: Learning rate for optimizer
        gamma: Discount factor
        buffer_size: Size of replay buffer
        batch_size: Batch size for training
        exploration_fraction: Fraction of training for exploration
        total_timesteps: Total training timesteps
        config_name: Name for this configuration
        save_dir: Directory to save models
    
    Returns:
        Dictionary with training results
    """
    print(f"\n{'='*60}")
    print(f"Training DQN: {config_name}")
    print(f"{'='*60}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Buffer Size: {buffer_size}")
    print(f"Batch Size: {batch_size}")
    print(f"Exploration Fraction: {exploration_fraction}")
    print(f"Total Timesteps: {total_timesteps}")
    
    # Create environment
    env = AdaptiveELearningEnv()
    env = Monitor(env)
    
    # Create callback
    callback = TrainingCallback()
    
    # Create DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        buffer_size=buffer_size,
        batch_size=batch_size,
        exploration_fraction=exploration_fraction,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        learning_starts=1000,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Train the model
    print("\nTraining...")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True
    )
    
    # Save the model
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, f"{config_name}.zip")
    model.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Calculate statistics
    episode_rewards = callback.episode_rewards
    if len(episode_rewards) > 0:
        mean_reward = np.mean(episode_rewards[-100:])  # Last 100 episodes
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
        'buffer_size': buffer_size,
        'batch_size': batch_size,
        'exploration_fraction': exploration_fraction,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'num_episodes': len(episode_rewards),
        'convergence_episode': convergence_episode,
        'episode_rewards': episode_rewards,
        'episode_lengths': callback.episode_lengths
    }
    
    print(f"\nResults:")
    print(f"  Mean Reward (last 100 episodes): {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"  Max Reward: {max_reward:.2f}")
    print(f"  Total Episodes: {len(episode_rewards)}")
    print(f"  Convergence Episode: {convergence_episode}")
    
    env.close()
    
    return results


def main():
    """Main training function with hyperparameter configurations"""
    
    print("="*60)
    print("DQN Training for Adaptive E-Learning Platform")
    print("="*60)
    
    # Define hyperparameter configurations
    configs = [
        # Config 1: Baseline
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 32,
            'exploration_fraction': 0.3,
            'name': 'dqn_baseline'
        },
        # Config 2: Higher learning rate
        {
            'learning_rate': 5e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 32,
            'exploration_fraction': 0.3,
            'name': 'dqn_high_lr'
        },
        # Config 3: Lower gamma
        {
            'learning_rate': 1e-4,
            'gamma': 0.95,
            'buffer_size': 10000,
            'batch_size': 32,
            'exploration_fraction': 0.3,
            'name': 'dqn_low_gamma'
        },
        # Config 4: Larger buffer
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 50000,
            'batch_size': 64,
            'exploration_fraction': 0.3,
            'name': 'dqn_large_buffer'
        },
        # Config 5: Larger batch
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 128,
            'exploration_fraction': 0.3,
            'name': 'dqn_large_batch'
        },
        # Config 6: More exploration
        {
            'learning_rate': 1e-4,
            'gamma': 0.99,
            'buffer_size': 10000,
            'batch_size': 32,
            'exploration_fraction': 0.5,
            'name': 'dqn_more_explore'
        },
        # Config 7: Aggressive learning
        {
            'learning_rate': 1e-3,
            'gamma': 0.99,
            'buffer_size': 20000,
            'batch_size': 64,
            'exploration_fraction': 0.4,
            'name': 'dqn_aggressive'
        },
        # Config 8: Conservative
        {
            'learning_rate': 5e-5,
            'gamma': 0.99,
            'buffer_size': 50000,
            'batch_size': 128,
            'exploration_fraction': 0.2,
            'name': 'dqn_conservative'
        },
        # Config 9: Balanced
        {
            'learning_rate': 3e-4,
            'gamma': 0.98,
            'buffer_size': 20000,
            'batch_size': 64,
            'exploration_fraction': 0.35,
            'name': 'dqn_balanced'
        },
        # Config 10: High capacity
        {
            'learning_rate': 2e-4,
            'gamma': 0.99,
            'buffer_size': 100000,
            'batch_size': 256,
            'exploration_fraction': 0.25,
            'name': 'dqn_high_capacity'
        }
    ]
    
    # Training parameters
    total_timesteps = 50000  # Reduced for faster training, increase for better results
    
    # Train all configurations
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\nTraining Configuration {i}/10")
        
        results = train_dqn(
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            buffer_size=config['buffer_size'],
            batch_size=config['batch_size'],
            exploration_fraction=config['exploration_fraction'],
            total_timesteps=total_timesteps,
            config_name=config['name']
        )
        
        all_results.append(results)
    
    # Find best configuration
    best_config = max(all_results, key=lambda x: x['mean_reward'])
    print(f"\n\n{'='*60}")
    print("BEST CONFIGURATION")
    print(f"{'='*60}")
    print(f"Name: {best_config['config_name']}")
    print(f"Mean Reward: {best_config['mean_reward']:.2f}")
    print(f"Learning Rate: {best_config['learning_rate']}")
    print(f"Gamma: {best_config['gamma']}")
    print(f"Buffer Size: {best_config['buffer_size']}")
    print(f"Batch Size: {best_config['batch_size']}")
    
    # Save results to CSV
    results_df = pd.DataFrame([{
        'Config': r['config_name'],
        'Learning Rate': r['learning_rate'],
        'Gamma': r['gamma'],
        'Buffer Size': r['buffer_size'],
        'Batch Size': r['batch_size'],
        'Exploration Fraction': r['exploration_fraction'],
        'Mean Reward': r['mean_reward'],
        'Std Reward': r['std_reward'],
        'Max Reward': r['max_reward'],
        'Episodes': r['num_episodes'],
        'Convergence': r['convergence_episode']
    } for r in all_results])
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/dqn_results.csv', index=False)
    print(f"\nResults saved to: results/dqn_results.csv")
    
    # Plot training curves
    plot_training_results(all_results, best_config)
    
    print("\n" + "="*60)
    print("DQN Training Completed Successfully!")
    print("="*60)


def plot_training_results(all_results, best_config):
    """Plot training results for all configurations"""
    
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot 1: Cumulative rewards for all configs
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for result in all_results:
        rewards = result['episode_rewards']
        if len(rewards) > 0:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed, alpha=0.6, label=result['config_name'])
    plt.xlabel('Episode')
    plt.ylabel('Reward (smoothed)')
    plt.title('DQN: Cumulative Rewards')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Best configuration detailed
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
    
    # Plot 3: Mean rewards comparison
    plt.subplot(2, 2, 3)
    configs = [r['config_name'] for r in all_results]
    means = [r['mean_reward'] for r in all_results]
    stds = [r['std_reward'] for r in all_results]
    
    x = np.arange(len(configs))
    plt.bar(x, means, yerr=stds, alpha=0.7, capsize=5)
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean Reward')
    plt.title('DQN: Mean Reward Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Convergence comparison
    plt.subplot(2, 2, 4)
    convergence = [r['convergence_episode'] for r in all_results]
    plt.bar(x, convergence, alpha=0.7, color='green')
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Episodes to Converge')
    plt.title('DQN: Convergence Speed')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/dqn_training_results.png', dpi=300, bbox_inches='tight')
    print("Plot saved to: results/plots/dqn_training_results.png")
    plt.close()


if __name__ == "__main__":
    main()