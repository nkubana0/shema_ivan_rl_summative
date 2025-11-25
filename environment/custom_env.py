"""
Custom Gymnasium Environment for Adaptive E-Learning and AAC Platform

This environment simulates an educational platform for children with physical disabilities,
where an RL agent learns to:
1. Adapt interface layout (AAC button positioning)
2. Adjust lesson difficulty dynamically
3. Predict communication symbols
4. Provide appropriate hints and feedback
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any


class AdaptiveELearningEnv(gym.Env):
    """
    Custom Environment for Adaptive E-Learning with AAC Integration
    
    Observation Space:
        - Current lesson difficulty (0-1)
        - Student accuracy on current lesson (0-1)
        - Time spent on current lesson (normalized)
        - Number of hints requested (normalized)
        - AAC button usage frequency (6 buttons)
        - Student engagement level (0-1)
        - Error rate (0-1)
        - Current interface configuration (0-1)
    
    Action Space (Discrete):
        0: Keep current layout (no change)
        1: Move most-used AAC button to top-left
        2: Increase lesson difficulty
        3: Decrease lesson difficulty
        4: Provide hint
        5: Predict next word (top prediction)
        6: Adjust button sizes (increase for frequently used)
    
    Reward Structure:
        +10: High accuracy on appropriate difficulty
        +5: Optimal challenge level maintained
        +3: Successful AAC prediction
        +2: Sustained engagement
        -5: Signs of frustration
        -2: Inappropriate difficulty level
    """
    
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 4}
    
    def __init__(self, render_mode: Optional[str] = None):
        super().__init__()
        
        # Define action and observation space
        self.action_space = spaces.Discrete(7)
        
        # 13-dimensional observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(13,), dtype=np.float32
        )
        
        # Rendering
        self.render_mode = render_mode
        self.window = None
        self.clock = None
        
        # Student state variables
        self.lesson_difficulty = 0.5  # 0=easy, 1=hard
        self.student_accuracy = 0.7  # Current accuracy
        self.time_on_lesson = 0  # Time steps on current lesson
        self.hints_requested = 0
        self.aac_button_usage = np.array([0.2, 0.15, 0.25, 0.1, 0.2, 0.1])  # 6 buttons
        self.engagement_level = 0.8
        self.error_rate = 0.3
        self.interface_config = 0.5  # Current layout efficiency
        
        # Episode tracking
        self.current_step = 0
        self.max_steps = 200
        
        # Student profile (simulates individual differences)
        # Realistic disability profiles
        self.disability_profiles = {
            'cerebral_palsy': {
                'motor_control': 0.3,  # Low motor control
                'cognitive': 0.8,      # Normal cognition
                'communication': 0.4,  # Needs AAC support
                'optimal_difficulty': 0.6,
                'learning_rate': 0.12
            },
            'muscular_dystrophy': {
                'motor_control': 0.4,
                'cognitive': 0.85,
                'communication': 0.6,
                'optimal_difficulty': 0.65,
                'learning_rate': 0.13
            },
            'spinal_cord_injury': {
                'motor_control': 0.2,
                'cognitive': 0.9,
                'communication': 0.5,
                'optimal_difficulty': 0.7,
                'learning_rate': 0.15
            },
            'severe_arthritis': {
                'motor_control': 0.5,
                'cognitive': 0.8,
                'communication': 0.7,
                'optimal_difficulty': 0.6,
                'learning_rate': 0.11
            }
        }
        
        # Select random profile for this episode
        profile_name = np.random.choice(list(self.disability_profiles.keys()))
        self.current_profile = self.disability_profiles[profile_name]
        self.profile_name = profile_name
        
        self.student_learning_rate = self.current_profile['learning_rate']
        self.optimal_difficulty = self.current_profile['optimal_difficulty']
        self.frustration_threshold = 0.5 - (self.current_profile['motor_control'] * 0.3)  # Lower motor control = more frustration
        
        # AAC symbol library - Educational context
        self.aac_symbols = [
            "I want", "Help me", "Yes", "No", "I understand", "Confused",
            "Too hard", "Too easy", "Break please", "Continue", "Repeat", "Show me",
            "Read", "Write", "Math", "Science", "Art", "Play",
            "Good", "Bad", "Happy", "Sad", "Tired", "Ready"
        ]
        
        # Track which symbols are most relevant for current lesson
        self.lesson_relevant_symbols = ["I understand", "Confused", "Help me", 
                                       "Too hard", "Too easy", "Repeat"]
        
        # History for tracking
        self.action_history = []
        self.reward_history = []
        
        # Realistic lesson content types
        self.lesson_types = [
            {'name': 'Colors & Shapes', 'subject': 'Art', 'base_difficulty': 0.3},
            {'name': 'Numbers 1-10', 'subject': 'Math', 'base_difficulty': 0.4},
            {'name': 'Letter Recognition', 'subject': 'Reading', 'base_difficulty': 0.35},
            {'name': 'Simple Addition', 'subject': 'Math', 'base_difficulty': 0.5},
            {'name': 'Animal Names', 'subject': 'Science', 'base_difficulty': 0.3},
            {'name': 'Days of Week', 'subject': 'General', 'base_difficulty': 0.45},
            {'name': 'Body Parts', 'subject': 'Science', 'base_difficulty': 0.35},
            {'name': 'Weather Types', 'subject': 'Science', 'base_difficulty': 0.4}
        ]
        self.current_lesson = np.random.choice(self.lesson_types)
        
        # Time of day affects engagement (morning = higher engagement)
        self.session_time = np.random.choice(['morning', 'afternoon', 'evening'])
        self.time_multiplier = {'morning': 1.0, 'afternoon': 0.9, 'evening': 0.8}[self.session_time]
    
    def reset(
        self, 
        seed: Optional[int] = None, 
        options: Optional[dict] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the environment to initial state"""
        super().reset(seed=seed)
        
        # Reset student state with some variability
        self.lesson_difficulty = np.random.uniform(0.3, 0.7)
        self.student_accuracy = np.random.uniform(0.5, 0.8)
        self.time_on_lesson = 0
        self.hints_requested = 0
        self.aac_button_usage = np.random.dirichlet(np.ones(6))
        self.engagement_level = np.random.uniform(0.6, 0.9)
        self.error_rate = 1.0 - self.student_accuracy
        self.interface_config = 0.5
        
        # Reset tracking
        self.current_step = 0
        self.action_history = []
        self.reward_history = []
        
        # Randomize student profile for generalization
        profile_name = np.random.choice(list(self.disability_profiles.keys()))
        self.current_profile = self.disability_profiles[profile_name]
        self.profile_name = profile_name
        
        self.student_learning_rate = self.current_profile['learning_rate']
        self.optimal_difficulty = self.current_profile['optimal_difficulty']
        self.frustration_threshold = 0.5 - (self.current_profile['motor_control'] * 0.3)
        
        # Select lesson and time
        self.current_lesson = np.random.choice(self.lesson_types)
        self.session_time = np.random.choice(['morning', 'afternoon', 'evening'])
        self.time_multiplier = {'morning': 1.0, 'afternoon': 0.9, 'evening': 0.8}[self.session_time]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(
        self, 
        action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Execute one time step"""
        self.current_step += 1
        self.time_on_lesson += 1
        
        # Record action
        self.action_history.append(action)
        
        # Execute action and calculate reward
        reward = self._execute_action(action)
        
        # Update student state based on action
        self._update_student_state(action)
        
        # Simulate student progress
        self._simulate_learning_progress()
        
        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        
        # Record reward
        self.reward_history.append(reward)
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, reward, terminated, truncated, info
    
    def _execute_action(self, action: int) -> float:
        """Execute action and calculate immediate reward"""
        reward = 0.0
        
        if action == 0:  # Keep current layout
            # Small positive reward for stability if things are going well
            if self.engagement_level > 0.7 and abs(self.lesson_difficulty - self.optimal_difficulty) < 0.2:
                reward += 2.0
            else:
                reward -= 1.0  # Penalty for not adapting when needed
        
        elif action == 1:  # Optimize AAC layout
            # Reward if this improves interface efficiency
            most_used_idx = np.argmax(self.aac_button_usage)
            improvement = 0.1 * self.aac_button_usage[most_used_idx]
            self.interface_config = min(1.0, self.interface_config + improvement)
            reward += 3.0 + improvement * 10
        
        elif action == 2:  # Increase difficulty
            if self.student_accuracy > 0.8 and self.lesson_difficulty < 0.9:
                # Good decision: student is ready for more challenge
                self.lesson_difficulty = min(1.0, self.lesson_difficulty + 0.1)
                reward += 5.0
            else:
                # Bad decision: student not ready
                self.lesson_difficulty = min(1.0, self.lesson_difficulty + 0.05)
                reward -= 5.0
        
        elif action == 3:  # Decrease difficulty
            if self.student_accuracy < 0.6 or self.engagement_level < 0.5:
                # Good decision: student needs easier content
                self.lesson_difficulty = max(0.1, self.lesson_difficulty - 0.1)
                reward += 5.0
            else:
                # Bad decision: making it too easy
                self.lesson_difficulty = max(0.1, self.lesson_difficulty - 0.05)
                reward -= 2.0
        
        elif action == 4:  # Provide hint
            self.hints_requested += 1
            if self.student_accuracy < 0.7:
                # Helpful hint
                reward += 3.0
                self.student_accuracy = min(1.0, self.student_accuracy + 0.05)
            else:
                # Unnecessary hint
                reward -= 1.0
        
        elif action == 5:  # Predict next AAC word
            # Simulate prediction accuracy based on usage patterns
            prediction_accuracy = np.max(self.aac_button_usage)
            if prediction_accuracy > 0.3:
                reward += 3.0 * prediction_accuracy
            else:
                reward += 1.0
        
        elif action == 6:  # Adjust button sizes
            # Reward for accessibility optimization
            reward += 2.0
            self.interface_config = min(1.0, self.interface_config + 0.05)
        
        # Bonus rewards for maintaining optimal conditions
        difficulty_diff = abs(self.lesson_difficulty - self.optimal_difficulty)
        if difficulty_diff < 0.15:
            reward += 10.0  # Optimal challenge level
        elif difficulty_diff < 0.25:
            reward += 5.0
        
        # Engagement reward
        if self.engagement_level > 0.7:
            reward += 2.0
        elif self.engagement_level < 0.4:
            reward -= 5.0  # Frustration penalty
        
        # Accuracy reward
        if self.student_accuracy > 0.8:
            reward += 5.0
        
        return reward
    
    def _update_student_state(self, action: int):
        """Update student state based on action and natural progression"""
        # Engagement naturally decreases over time
        self.engagement_level *= 0.98
        
        # Difficulty mismatch affects engagement
        difficulty_mismatch = abs(self.lesson_difficulty - self.optimal_difficulty)
        if difficulty_mismatch > 0.3:
            self.engagement_level = max(0.1, self.engagement_level - 0.05)
        
        # Update AAC button usage (simulate communication patterns)
        # Add small random variations
        usage_change = np.random.normal(0, 0.02, 6)
        self.aac_button_usage = np.clip(
            self.aac_button_usage + usage_change, 0, 1
        )
        self.aac_button_usage /= np.sum(self.aac_button_usage)  # Normalize
        
        # Interface config degrades slightly over time without optimization
        if action not in [1, 6]:
            self.interface_config = max(0.2, self.interface_config - 0.01)
    
    def _simulate_learning_progress(self):
        """Simulate student learning over time"""
        # Student improves with appropriate difficulty
        difficulty_factor = 1.0 - abs(self.lesson_difficulty - self.optimal_difficulty)
        learning_gain = self.student_learning_rate * difficulty_factor * 0.5
        
        # Accuracy improves with learning
        self.student_accuracy = min(1.0, self.student_accuracy + learning_gain)
        
        # Add some noise to simulate variability
        self.student_accuracy = np.clip(
            self.student_accuracy + np.random.normal(0, 0.02), 0.3, 1.0
        )
        
        # Update error rate
        self.error_rate = 1.0 - self.student_accuracy
        
        # Engagement recovers with successful learning
        if learning_gain > 0:
            self.engagement_level = min(1.0, self.engagement_level + 0.02)
    
    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Terminate if student becomes too disengaged
        if self.engagement_level < 0.2:
            return True
        
        # Terminate if excellent performance achieved
        if self.student_accuracy > 0.95 and self.engagement_level > 0.8:
            return True
        
        return False
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        observation = np.array([
            self.lesson_difficulty,
            self.student_accuracy,
            min(1.0, self.time_on_lesson / 50.0),  # Normalized time
            min(1.0, self.hints_requested / 10.0),  # Normalized hints
            *self.aac_button_usage,  # 6 values
            self.engagement_level,
            self.error_rate,
            self.interface_config
        ], dtype=np.float32)
        
        return observation
    
    def _get_info(self) -> Dict[str, Any]:
        """Get additional information"""
        return {
            'lesson_difficulty': self.lesson_difficulty,
            'student_accuracy': self.student_accuracy,
            'engagement_level': self.engagement_level,
            'time_on_lesson': self.time_on_lesson,
            'hints_requested': self.hints_requested,
            'interface_efficiency': self.interface_config,
            'step': self.current_step,
            'profile_name': self.profile_name,
            'lesson_name': self.current_lesson['name'],
            'lesson_subject': self.current_lesson['subject'],
            'session_time': self.session_time,
            'motor_control': self.current_profile['motor_control'],
            'cognitive_ability': self.current_profile['cognitive']
        }
    
    def render(self):
        """Render the environment (handled by external visualization)"""
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            # Will be handled by pygame visualization
            pass
    
    def _render_frame(self):
        """Generate RGB array representation"""
        # Placeholder for RGB rendering
        return np.zeros((400, 600, 3), dtype=np.uint8)
    
    def close(self):
        """Clean up resources"""
        if self.window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()


# Test the environment
if __name__ == "__main__":
    print("Testing Adaptive E-Learning Environment...")
    
    env = AdaptiveELearningEnv()
    
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial Observation Shape: {obs.shape}")
    print(f"Initial Observation: {obs}")
    print(f"Initial Info: {info}")
    
    # Test random actions
    print("\n" + "="*50)
    print("Testing Random Actions")
    print("="*50)
    
    total_reward = 0
    for step in range(10):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        
        action_names = [
            "Keep layout", "Optimize AAC", "Increase difficulty",
            "Decrease difficulty", "Provide hint", "Predict word", "Adjust buttons"
        ]
        
        print(f"\nStep {step + 1}:")
        print(f"  Action: {action} ({action_names[action]})")
        print(f"  Reward: {reward:.2f}")
        print(f"  Accuracy: {info['student_accuracy']:.3f}")
        print(f"  Engagement: {info['engagement_level']:.3f}")
        print(f"  Difficulty: {info['lesson_difficulty']:.3f}")
        
        if terminated or truncated:
            print(f"\nEpisode ended at step {step + 1}")
            print(f"Terminated: {terminated}, Truncated: {truncated}")
            break
    
    print(f"\nTotal Reward: {total_reward:.2f}")
    print("\nEnvironment test completed successfully!")