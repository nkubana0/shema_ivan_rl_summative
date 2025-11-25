"""
Main Entry Point for Running Trained RL Agents

This script loads trained models and runs them in the environment with visualization.
"""

import argparse
import sys
import os
import numpy as np
import pygame
from stable_baselines3 import DQN, PPO, A2C
import torch

from environment.custom_env import AdaptiveELearningEnv
from environment.rendering import ELearningVisualizer
from training.reinforce_training import REINFORCE, PolicyNetwork


def load_model(algorithm: str, model_path: str = None):
    """
    Load a trained model
    
    Args:
        algorithm: One of 'dqn', 'ppo', 'a2c', 'reinforce'
        model_path: Path to model file (if None, loads best model)
    
    Returns:
        Loaded model
    """
    if model_path is None:
        # Load best model based on algorithm
        if algorithm == 'dqn':
            model_path = 'models/dqn/dqn_baseline.zip'  # Change to your best model
        elif algorithm == 'ppo':
            model_path = 'models/ppo/ppo_baseline.zip'
        elif algorithm == 'a2c':
            model_path = 'models/a2c/a2c_baseline.zip'
        elif algorithm == 'reinforce':
            model_path = 'models/reinforce/reinforce_baseline.pth'
    
    print(f"Loading model from: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Please train the model first using the training scripts.")
        sys.exit(1)
    
    if algorithm == 'dqn':
        model = DQN.load(model_path)
    elif algorithm == 'ppo':
        model = PPO.load(model_path)
    elif algorithm == 'a2c':
        model = A2C.load(model_path)
    elif algorithm == 'reinforce':
        # Load custom REINFORCE model
        env = AdaptiveELearningEnv()
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
        model = REINFORCE(state_dim, action_dim)
        model.load(model_path)
        env.close()
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return model


def run_random_agent(num_episodes: int = 5, render: bool = True):
    """
    Run random agent for baseline comparison
    
    Args:
        num_episodes: Number of episodes to run
        render: Whether to render visualization
    """
    print("\n" + "="*60)
    print("Running Random Agent (Baseline)")
    print("="*60)
    
    env = AdaptiveELearningEnv()
    
    if render:
        visualizer = ELearningVisualizer()
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            # Random action
            action = env.action_space.sample()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if render:
                env_state = {
                    'lesson_difficulty': info['lesson_difficulty'],
                    'student_accuracy': info['student_accuracy'],
                    'engagement_level': info['engagement_level'],
                    'hints_requested': info['hints_requested'],
                    'time_on_lesson': info['time_on_lesson'],
                    'interface_efficiency': info['interface_efficiency'],
                    'aac_button_usage': state[4:10],  # AAC usage from observation
                    'step': info['step']
                }
                
                running = visualizer.render(
                    env_state, action, reward, episode + 1, episode_reward
                )
                
                if not running:
                    break
            
            state = next_state
            done = terminated or truncated
        
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {step}")
    
    if render:
        visualizer.close()
    env.close()


def run_trained_agent(
    algorithm: str,
    model_path: str = None,
    num_episodes: int = 5,
    render: bool = True
):
    """
    Run a trained agent
    
    Args:
        algorithm: Algorithm type ('dqn', 'ppo', 'a2c', 'reinforce')
        model_path: Path to model file
        num_episodes: Number of episodes to run
        render: Whether to render visualization
    """
    print("\n" + "="*60)
    print(f"Running Trained {algorithm.upper()} Agent")
    print("="*60)
    
    # Load model
    model = load_model(algorithm, model_path)
    
    # Create environment
    env = AdaptiveELearningEnv()
    
    if render:
        visualizer = ELearningVisualizer()
    
    episode_rewards = []
    
    for episode in range(num_episodes):
        state, info = env.reset()
        episode_reward = 0
        step = 0
        done = False
        
        print(f"\nEpisode {episode + 1}")
        
        while not done:
            # Get action from model
            if algorithm == 'reinforce':
                action = model.select_action(state)
            else:
                action, _ = model.predict(state, deterministic=True)
            
            next_state, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step += 1
            
            if render:
                env_state = {
                    'lesson_difficulty': info['lesson_difficulty'],
                    'student_accuracy': info['student_accuracy'],
                    'engagement_level': info['engagement_level'],
                    'hints_requested': info['hints_requested'],
                    'time_on_lesson': info['time_on_lesson'],
                    'interface_efficiency': info['interface_efficiency'],
                    'aac_button_usage': state[4:10],
                    'step': info['step']
                }
                
                running = visualizer.render(
                    env_state, action, reward, episode + 1, episode_reward
                )
                
                if not running:
                    break
            
            state = next_state
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        print(f"Episode Reward: {episode_reward:.2f}")
        print(f"Episode Length: {step}")
        print(f"Final Student Accuracy: {info['student_accuracy']:.3f}")
        print(f"Final Engagement: {info['engagement_level']:.3f}")
    
    if render:
        visualizer.close()
    env.close()
    
    # Print summary
    print("\n" + "="*60)
    print("Summary Statistics")
    print("="*60)
    print(f"Mean Reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")


def test_environment():
    """Test the environment with visualization"""
    print("\n" + "="*60)
    print("Testing Environment")
    print("="*60)
    
    env = AdaptiveELearningEnv()
    visualizer = ELearningVisualizer()
    
    state, info = env.reset()
    done = False
    total_reward = 0
    episode = 1
    
    print("\nTaking random actions to demonstrate environment...")
    print("Press ESC or Q to quit\n")
    
    while not done:
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        env_state = {
            'lesson_difficulty': info['lesson_difficulty'],
            'student_accuracy': info['student_accuracy'],
            'engagement_level': info['engagement_level'],
            'hints_requested': info['hints_requested'],
            'time_on_lesson': info['time_on_lesson'],
            'interface_efficiency': info['interface_efficiency'],
            'aac_button_usage': state[4:10],
            'step': info['step']
        }
        
        running = visualizer.render(
            env_state, action, reward, episode, total_reward
        )
        
        if not running:
            break
        
        state = next_state
        done = terminated or truncated
    
    visualizer.close()
    env.close()


def main():
    parser = argparse.ArgumentParser(
        description="Run trained RL agents for Adaptive E-Learning Platform"
    )
    parser.add_argument(
        '--algorithm',
        type=str,
        choices=['dqn', 'ppo', 'a2c', 'reinforce'],
        default='ppo',
        help='RL algorithm to use'
    )
    parser.add_argument(
        '--model-path',
        type=str,
        default=None,
        help='Path to model file (optional)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=5,
        help='Number of episodes to run'
    )
    parser.add_argument(
        '--no-render',
        action='store_true',
        help='Disable visualization'
    )
    parser.add_argument(
        '--random',
        action='store_true',
        help='Run random agent instead of trained agent'
    )
    parser.add_argument(
        '--test-env',
        action='store_true',
        help='Just test the environment'
    )
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*60)
    print("Adaptive E-Learning & AAC Platform")
    print("Reinforcement Learning Agent Demonstration")
    print("="*60)
    
    try:
        if args.test_env:
            test_environment()
        elif args.random:
            run_random_agent(args.episodes, not args.no_render)
        else:
            run_trained_agent(
                args.algorithm,
                args.model_path,
                args.episodes,
                not args.no_render
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)
    print("Program Completed")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()