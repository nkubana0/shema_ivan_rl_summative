"""
A2C Training Script for Adaptive E-Learning Environment

Advantage Actor-Critic with multiple hyperparameter configurations
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
from environment.custom_env import AdaptiveELearningEnv
import torch
import pandas as pd


class TrainingCallback(BaseCallback):
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


def train_a2c(
    learning_rate: float,
    gamma: float,
    gae_lambda: float,
    ent_coef: float,
    vf_coef: float,
    n_steps: int,
    total_timesteps: int,
    config_name: str,
    save_dir: str = "models/a2c"
):
    """Train an A2C agent"""
    
    print(f"\n{'='*60}")
    print(f"Training A2C: {config_name}")
    print(f"{'='*60}")
    
    env = AdaptiveELearningEnv()
    env = Monitor(env)
    callback = TrainingCallback()
    
    model = A2C(
        "MlpPolicy",
        env,
        learning_rate=learning_rate,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        n_steps=n_steps,
        verbose=0,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    print("\nTraining...")
    model.learn(total_timesteps=total_timesteps, callback=callback, progress_bar=True)
    
    os.makedirs(save_dir, exist_ok=True)
    model.save(os.path.join(save_dir, f"{config_name}.zip"))
    
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
        'ent_coef': ent_coef,
        'vf_coef': vf_coef,
        'n_steps': n_steps,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'num_episodes': len(episode_rewards),
        'convergence_episode': convergence_episode,
        'episode_rewards': episode_rewards
    }
    
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    env.close()
    return results


def main():
    print("="*60)
    print("A2C Training for Adaptive E-Learning Platform")
    print("="*60)
    
    configs = [
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_baseline'},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_high_lr'},
        {'learning_rate': 7e-4, 'gamma': 0.95, 'gae_lambda': 1.0, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_low_gamma'},
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 0.95, 'ent_coef': 0.01, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_gae'},
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.05, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_high_ent'},
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.001, 
         'vf_coef': 0.5, 'n_steps': 5, 'name': 'a2c_low_ent'},
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.01, 
         'vf_coef': 0.25, 'n_steps': 5, 'name': 'a2c_low_vf'},
        {'learning_rate': 7e-4, 'gamma': 0.99, 'gae_lambda': 1.0, 'ent_coef': 0.01, 
         'vf_coef': 1.0, 'n_steps': 5, 'name': 'a2c_high_vf'},
        {'learning_rate': 5e-4, 'gamma': 0.98, 'gae_lambda': 0.95, 'ent_coef': 0.02, 
         'vf_coef': 0.4, 'n_steps': 8, 'name': 'a2c_balanced'},
        {'learning_rate': 3e-4, 'gamma': 0.99, 'gae_lambda': 0.98, 'ent_coef': 0.005, 
         'vf_coef': 0.6, 'n_steps': 10, 'name': 'a2c_conservative'}
    ]
    
    all_results = []
    for i, config in enumerate(configs, 1):
        print(f"\n\nTraining Configuration {i}/10")
        results = train_a2c(
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            gae_lambda=config['gae_lambda'],
            ent_coef=config['ent_coef'],
            vf_coef=config['vf_coef'],
            n_steps=config['n_steps'],
            total_timesteps=50000,
            config_name=config['name']
        )
        all_results.append(results)
    
    best_config = max(all_results, key=lambda x: x['mean_reward'])
    print(f"\n\n{'='*60}")
    print("BEST A2C CONFIGURATION")
    print(f"{'='*60}")
    print(f"Name: {best_config['config_name']}")
    print(f"Mean Reward: {best_config['mean_reward']:.2f}")
    
    results_df = pd.DataFrame([{
        'Config': r['config_name'],
        'Learning Rate': r['learning_rate'],
        'Gamma': r['gamma'],
        'Mean Reward': r['mean_reward']
    } for r in all_results])
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/a2c_results.csv', index=False)
    
    plot_training_results(all_results, best_config)
    print("\nA2C Training Completed!")


def plot_training_results(all_results, best_config):
    os.makedirs('results/plots', exist_ok=True)
    
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 2, 1)
    for result in all_results:
        rewards = result['episode_rewards']
        if len(rewards) > 0:
            smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
            plt.plot(smoothed, alpha=0.6, label=result['config_name'])
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title('A2C: Cumulative Rewards')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 2)
    rewards = best_config['episode_rewards']
    if len(rewards) > 0:
        plt.plot(rewards, alpha=0.3, color='blue')
        smoothed = np.convolve(rewards, np.ones(10)/10, mode='valid')
        plt.plot(smoothed, color='red', linewidth=2)
    plt.xlabel('Episode')
    plt.ylabel('Reward')
    plt.title(f'Best Config: {best_config["config_name"]}')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 2, 3)
    configs = [r['config_name'] for r in all_results]
    means = [r['mean_reward'] for r in all_results]
    x = np.arange(len(configs))
    plt.bar(x, means, alpha=0.7)
    plt.xticks(x, configs, rotation=45, ha='right', fontsize=8)
    plt.ylabel('Mean Reward')
    plt.title('A2C: Mean Reward Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('results/plots/a2c_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()