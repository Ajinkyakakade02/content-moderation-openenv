import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Dict, Any, Optional, Tuple, Union
import json
import random
from datetime import datetime
from .models import (
    Observation, Action, Reward, ModerationAction, 
    ContentItem, ContentType, ViolationCategory
)
from .reward_functions import AdvancedRewardFunction

class ModerationEnv(gym.Env):
    """
    OpenEnv-compliant Content Moderation Environment
    
    Simulates real-world content moderation where an AI agent must decide
    to ALLOW, FLAG, or REMOVE user content based on safety policies.
    """
    
    metadata = {'render_modes': ['human', 'ansi']}
    
    def __init__(self, 
                 dataset_path: str = 'data/dataset_easy.json',
                 max_steps: int = 20,
                 render_mode: str = 'human',
                 task_difficulty: str = 'easy'):
        
        super().__init__()
        
        self.max_steps = max_steps
        self.render_mode = render_mode
        self.task_difficulty = task_difficulty
        self.current_step = 0
        self.current_index = 0
        self.current_content = None
        self.history = []
        self.total_reward = 0.0
        
        # Load dataset
        with open(dataset_path, 'r') as f:
            self.dataset = json.load(f)
        self.dataset = [ContentItem(**item) for item in self.dataset]
        
        # Define action space: 0=ALLOW, 1=FLAG, 2=REMOVE
        self.action_space = spaces.Discrete(3)
        
        # Define observation space
        self.observation_space = spaces.Dict({
            'text': spaces.Text(max_length=1000),
            'user_reputation': spaces.Box(low=0, high=1, shape=(1,)),
            'report_count': spaces.Discrete(101),
            'policy_summary': spaces.Text(max_length=500),
            'content_type': spaces.Discrete(4)
        })
        
        # Initialize reward function
        self.reward_fn = AdvancedRewardFunction(task_difficulty)
        
        self.reset()
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[Dict, Dict]:
        """Reset environment to initial state"""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.current_index = 0
        self.history = []
        self.total_reward = 0.0
        
        # Shuffle dataset if needed
        if options and options.get('shuffle', False):
            random.shuffle(self.dataset)
        
        self.current_content = self.dataset[0]
        
        observation = self._get_observation()
        info = self._get_info()
        
        return observation, info
    
    def step(self, action: int) -> Tuple[Dict, float, bool, bool, Dict]:
        """
        Execute one step in the environment
        
        Args:
            action: 0=ALLOW, 1=FLAG, 2=REMOVE
            
        Returns:
            observation: Next observation
            reward: Reward for this step
            done: Whether episode is complete
            truncated: Whether episode was truncated
            info: Additional information
        """
        
        action_enum = ModerationAction(action) if isinstance(action, int) else action
        
        # Calculate reward
        reward, reward_details = self._calculate_reward(action_enum, self.current_content)
        self.total_reward += reward
        
        # Record history
        self.history.append({
            'content': self.current_content.dict(),
            'action': action_enum.value,
            'reward': reward,
            'reward_details': reward_details,
            'step': self.current_step,
            'timestamp': datetime.now().isoformat()
        })
        
        # Move to next content
        self.current_step += 1
        self.current_index += 1
        
        # Check if episode is done
        done = self.current_step >= self.max_steps or self.current_index >= len(self.dataset)
        
        # Get next observation
        if not done:
            self.current_content = self.dataset[self.current_index]
            observation = self._get_observation()
        else:
            observation = None
        
        info = self._get_info()
        info['reward_details'] = reward_details
        info['total_reward'] = self.total_reward
        
        if self.render_mode == 'human':
            self.render()
        
        return observation, reward, done, False, info
    
    def _calculate_reward(self, action: ModerationAction, content: ContentItem) -> Tuple[float, Dict]:
        """Calculate reward with partial progress signals"""
        
        # If we have ground truth labels
        if content.true_label:
            return self._reward_with_labels(action, content)
        
        # Heuristic reward for unlabeled data
        return self.reward_fn.calculate_reward(action, content, {
            'step': self.current_step,
            'total_steps': self.max_steps,
            'history': self.history
        })
    
    def _reward_with_labels(self, action: ModerationAction, content: ContentItem) -> Tuple[float, Dict]:
        """Reward using ground truth labels"""
        
        # Define correct action mapping
        correct_mapping = {
            'safe': ModerationAction.ALLOW,
            'toxic': ModerationAction.FLAG,
            'hate_speech': ModerationAction.REMOVE,
            'nudity': ModerationAction.REMOVE,
            'violence': ModerationAction.REMOVE,
            'harassment': ModerationAction.FLAG,
            'misinformation': ModerationAction.FLAG,
            'spam': ModerationAction.FLAG
        }
        
        correct_action = correct_mapping.get(content.true_label, ModerationAction.ALLOW)
        
        components = {}
        
        # Base correctness
        if action == correct_action:
            base_reward = 1.0
            components['correctness'] = 1.0
        else:
            # Different penalties based on severity
            severity_penalties = {
                (ModerationAction.ALLOW, 'hate_speech'): -2.0,
                (ModerationAction.ALLOW, 'nudity'): -2.0,
                (ModerationAction.ALLOW, 'violence'): -1.5,
                (ModerationAction.ALLOW, 'toxic'): -1.0,
                (ModerationAction.REMOVE, 'safe'): -0.5,
                (ModerationAction.FLAG, 'safe'): -0.3,
                (ModerationAction.REMOVE, 'toxic'): -0.2,
            }
            base_reward = severity_penalties.get((action, content.true_label), -0.5)
            components['correctness'] = 0.0
            components['penalty'] = base_reward
        
        # Efficiency bonus (fast decisions)
        if self.current_step < self.max_steps * 0.3:
            efficiency_bonus = 0.1
            components['efficiency'] = efficiency_bonus
        else:
            efficiency_bonus = 0.0
            components['efficiency'] = 0.0
        
        # Consistency bonus (similar to previous decisions)
        if len(self.history) > 0:
            last_action = ModerationAction(self.history[-1]['action'])
            if last_action == action:
                consistency_bonus = 0.05
                components['consistency'] = consistency_bonus
            else:
                consistency_bonus = 0.0
                components['consistency'] = 0.0
        else:
            consistency_bonus = 0.0
            components['consistency'] = 0.0
        
        total_reward = base_reward + efficiency_bonus + consistency_bonus
        
        return total_reward, components
    
    def _get_observation(self) -> Dict:
        """Get current observation as dict"""
        return {
            'text': self.current_content.text,
            'user_reputation': np.array([self.current_content.user_reputation]),
            'report_count': self.current_content.report_count,
            'policy_summary': self._get_policy_summary(),
            'content_type': list(ContentType).index(self.current_content.content_type)
        }
    
    def _get_policy_summary(self) -> str:
        """Get current policy guidelines"""
        policies = {
            'easy': "Block hate speech, nudity, violence. Allow safe content.",
            'medium': "Consider context, sarcasm, cultural nuances. Flag ambiguous content.",
            'hard': "Balance speed vs accuracy. Prioritize critical violations."
        }
        return policies.get(self.task_difficulty, policies['easy'])
    
    def _get_info(self) -> Dict:
        """Get additional environment info"""
        return {
            'step': self.current_step,
            'total_steps': self.max_steps,
            'current_index': self.current_index,
            'total_posts': len(self.dataset),
            'history_length': len(self.history),
            'task_difficulty': self.task_difficulty,
            'total_reward': self.total_reward
        }
    
    def state(self) -> Dict:
        """OpenEnv state() method - returns full internal state"""
        return {
            'current_step': self.current_step,
            'current_index': self.current_index,
            'history': self.history,
            'total_reward': self.total_reward,
            'current_content': self.current_content.dict() if self.current_content else None,
            'dataset_length': len(self.dataset)
        }
    
    def render(self):
        """Render environment state"""
        if self.render_mode == 'human':
            print(f"\n{'='*50}")
            print(f"Step {self.current_step}/{self.max_steps}")
            print(f"Content: {self.current_content.text[:100]}...")
            print(f"User Rep: {self.current_content.user_reputation:.2f}")
            print(f"Reports: {self.current_content.report_count}")
            print(f"True Label: {self.current_content.true_label or 'Unknown'}")
            print(f"{'='*50}\n")