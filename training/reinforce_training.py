"""
REINFORCE Training Script for Adaptive E-Learning Environment

Custom implementation of the REINFORCE algorithm (Monte Carlo Policy Gradient)
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from environment.custom_env import AdaptiveELearningEnv
import pandas as pd
from tqdm import tqdm


class PolicyNetwork(nn.Module):
    """Neural network for policy"""
    
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, state):
        return self.network(state)


class REINFORCE:
    """REINFORCE algorithm implementation"""
    
    def __init__(
        self,
        state_dim,
        action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        hidden_dim=128
    ):
        self.gamma = gamma
        self.policy = PolicyNetwork(state_dim, action_dim, hidden_dim)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        self.saved_log_probs = []
        self.rewards = []
    
    def select_action(self, state):
        """Select action using current policy"""
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.policy(state)
        m = Categorical(probs)
        action = m.sample()
        self.saved_log_probs.append(m.log_prob(action))
        return action.item()
    
    def update_policy(self):
        """Update policy using collected episode"""
        R = 0
        returns = []
        
        # Calculate returns
        for r in self.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns)
        returns = (returns - returns.mean()) / (returns.std() + 1e-9)
        
        # Calculate loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)
        
        policy_loss = torch.cat(policy_loss).sum()
        
        # Update
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()
        
        # Clear episode data
        self.saved_log_probs = []
        self.rewards = []
        
        return policy_loss.item()
    
    def save(self, path):
        """Save model"""
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load(self, path):
        """Load model"""
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint['policy_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


def train_reinforce(
    learning_rate: float,
    gamma: float,
    hidden_dim: int,
    num_episodes: int,
    config_name: str,
    save_dir: str = "models/reinforce"
):
    """Train REINFORCE agent"""
    
    print(f"\n{'='*60}")
    print(f"Training REINFORCE: {config_name}")
    print(f"{'='*60}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Gamma: {gamma}")
    print(f"Hidden Dim: {hidden_dim}")
    
    env = AdaptiveELearningEnv()
    
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    agent = REINFORCE(state_dim, action_dim, learning_rate, gamma, hidden_dim)
    
    episode_rewards = []
    episode_lengths = []
    losses = []
    
    print("\nTraining...")
    for episode in tqdm(range(num_episodes)):
        state, _ = env.reset()
        episode_reward = 0
        episode_length = 0
        
        done = False
        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            
            agent.rewards.append(reward)
            episode_reward += reward
            episode_length += 1
            
            state = next_state
            done = terminated or truncated
        
        loss = agent.update_policy()
        losses.append(loss)
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        if (episode + 1) % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:])
            print(f"Episode {episode + 1}: Avg Reward = {avg_reward:.2f}")
    
    # Save model
    os.makedirs(save_dir, exist_ok=True)
    agent.save(os.path.join(save_dir, f"{config_name}.pth"))
    
    # Calculate statistics
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
        'hidden_dim': hidden_dim,
        'mean_reward': mean_reward,
        'std_reward': std_reward,
        'max_reward': max_reward,
        'num_episodes': len(episode_rewards),
        'convergence_episode': convergence_episode,
        'episode_rewards': episode_rewards,
        'losses': losses
    }
    
    print(f"\nResults:")
    print(f"  Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    env.close()
    return results


def main():
    print("="*60)
    print("REINFORCE Training for Adaptive E-Learning Platform")
    print("="*60)
    
    configs = [
        {'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_dim': 128, 'name': 'reinforce_baseline'},
        {'learning_rate': 5e-3, 'gamma': 0.99, 'hidden_dim': 128, 'name': 'reinforce_high_lr'},
        {'learning_rate': 1e-4, 'gamma': 0.99, 'hidden_dim': 128, 'name': 'reinforce_low_lr'},
        {'learning_rate': 1e-3, 'gamma': 0.95, 'hidden_dim': 128, 'name': 'reinforce_low_gamma'},
        {'learning_rate': 1e-3, 'gamma': 0.98, 'hidden_dim': 128, 'name': 'reinforce_med_gamma'},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_dim': 64, 'name': 'reinforce_small'},
        {'learning_rate': 1e-3, 'gamma': 0.99, 'hidden_dim': 256, 'name': 'reinforce_large'},
        {'learning_rate': 3e-3, 'gamma': 0.97, 'hidden_dim': 128, 'name': 'reinforce_aggressive'},
        {'learning_rate': 5e-4, 'gamma': 0.99, 'hidden_dim': 256, 'name': 'reinforce_conservative'},
        {'learning_rate': 2e-3, 'gamma': 0.98, 'hidden_dim': 192, 'name': 'reinforce_balanced'}
    ]
    
    num_episodes = 500  # Adjust based on your needs
    all_results = []
    
    for i, config in enumerate(configs, 1):
        print(f"\n\nTraining Configuration {i}/10")
        results = train_reinforce(
            learning_rate=config['learning_rate'],
            gamma=config['gamma'],
            hidden_dim=config['hidden_dim'],
            num_episodes=num_episodes,
            config_name=config['name']
        )
        all_results.append(results)
    
    best_config = max(all_results, key=lambda x: x['mean_reward'])
    print(f"\n\n{'='*60}")
    print("BEST REINFORCE CONFIGURATION")
    print(f"{'='*60}")
    print(f"Name: {best_config['config_name']}")
    print(f"Mean Reward: {best_config['mean_reward']:.2f}")
    
    results_df = pd.DataFrame([{
        'Config': r['config_name'],
        'Learning Rate': r['learning_rate'],
        'Gamma': r['gamma'],
        'Hidden Dim': r['hidden_dim'],
        'Mean Reward': r['mean_reward']
    } for r in all_results])
    
    os.makedirs('results', exist_ok=True)
    results_df.to_csv('results/reinforce_results.csv', index=False)
    
    plot_training_results(all_results, best_config)
    print("\nREINFORCE Training Completed!")


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
    plt.title('REINFORCE: Cumulative Rewards')
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
    plt.title('REINFORCE: Mean Reward Comparison')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.subplot(2, 2, 4)
    losses = best_config['losses']
    if len(losses) > 0:
        smoothed = np.convolve(losses, np.ones(10)/10, mode='valid')
        plt.plot(smoothed)
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.title('REINFORCE: Training Loss')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/reinforce_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()