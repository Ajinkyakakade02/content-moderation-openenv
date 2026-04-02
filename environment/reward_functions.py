import numpy as np
from typing import Dict, Tuple, List
from .models import ModerationAction, ContentItem

class AdvancedRewardFunction:
    """
    Sophisticated reward function with partial progress signals
    Provides meaningful feedback throughout the trajectory
    """
    
    def __init__(self, task_difficulty: str = 'easy'):
        self.task_difficulty = task_difficulty
        self.history = []
        
        # Dynamic weights based on task difficulty
        self.weights = {
            'easy': {
                'accuracy': 0.7,
                'efficiency': 0.1,
                'consistency': 0.1,
                'exploration': 0.1
            },
            'medium': {
                'accuracy': 0.5,
                'efficiency': 0.2,
                'consistency': 0.15,
                'exploration': 0.15
            },
            'hard': {
                'accuracy': 0.4,
                'efficiency': 0.3,
                'consistency': 0.2,
                'exploration': 0.1
            }
        }
        
    def calculate_reward(self, action: ModerationAction, 
                        content: ContentItem, 
                        context: Dict) -> Tuple[float, Dict]:
        """
        Calculate multi-component reward with partial progress
        
        Returns:
            total_reward: Float reward value
            components: Dict of individual reward components
        """
        
        weights = self.weights.get(self.task_difficulty, self.weights['easy'])
        components = {}
        
        # 1. Accuracy component (primary)
        accuracy_reward = self._accuracy_component(action, content)
        components['accuracy'] = accuracy_reward
        
        # 2. Efficiency component (faster = better)
        efficiency_reward = self._efficiency_component(context)
        components['efficiency'] = efficiency_reward
        
        # 3. Consistency component (stable decisions)
        consistency_reward = self._consistency_component(action)
        components['consistency'] = consistency_reward
        
        # 4. Exploration component (for learning)
        exploration_reward = self._exploration_component(action, content)
        components['exploration'] = exploration_reward
        
        # Weighted sum
        total_reward = sum(
            weights[key] * components[key] 
            for key in components.keys()
        )
        
        # Record for history
        self.history.append({
            'action': action,
            'reward': total_reward,
            'components': components
        })
        
        # Keep last 10 for consistency
        if len(self.history) > 10:
            self.history.pop(0)
        
        return total_reward, components
    
    def _accuracy_component(self, action: ModerationAction, content: ContentItem) -> float:
        """Calculate accuracy-based reward"""
        
        if not content.true_label:
            # Use heuristic when no labels
            return self._heuristic_accuracy(action, content)
        
        # Define correct action per label
        correct_map = {
            'safe': ModerationAction.ALLOW,
            'toxic': ModerationAction.FLAG,
            'hate_speech': ModerationAction.REMOVE,
            'nudity': ModerationAction.REMOVE,
            'violence': ModerationAction.REMOVE,
            'harassment': ModerationAction.FLAG,
            'misinformation': ModerationAction.FLAG,
            'spam': ModerationAction.FLAG
        }
        
        correct = correct_map.get(content.true_label, ModerationAction.ALLOW)
        
        if action == correct:
            return 1.0
        
        # Partial credit for close decisions
        if (action == ModerationAction.FLAG and 
            correct == ModerationAction.REMOVE):
            return 0.5  # Flagging harmful content is partially correct
        
        if (action == ModerationAction.FLAG and 
            correct == ModerationAction.ALLOW):
            return 0.3  # Better to flag than remove safe content
        
        return -0.5  # Wrong decision
    
    def _heuristic_accuracy(self, action: ModerationAction, content: ContentItem) -> float:
        """Heuristic when labels unavailable"""
        
        # Simple toxicity detection
        toxic_words = ['hate', 'kill', 'stupid', 'idiot', 'trash', 'nude', 'violence']
        text_lower = content.text.lower()
        
        toxicity_score = sum(1 for word in toxic_words if word in text_lower) / len(toxic_words)
        
        # Adjust by reputation and reports
        adjusted_score = (toxicity_score * 0.6 + 
                         (1 - content.user_reputation) * 0.2 + 
                         min(content.report_count / 10, 1) * 0.2)
        
        # Reward based on action appropriateness
        if adjusted_score > 0.7 and action == ModerationAction.REMOVE:
            return 0.8
        elif adjusted_score > 0.4 and action == ModerationAction.FLAG:
            return 0.6
        elif adjusted_score < 0.3 and action == ModerationAction.ALLOW:
            return 0.7
        else:
            return -0.3
    
    def _efficiency_component(self, context: Dict) -> float:
        """Reward for quick decisions (partial progress)"""
        
        step = context.get('step', 0)
        total_steps = context.get('total_steps', 20)
        
        # Early decisions get bonus
        if step < total_steps * 0.3:
            return 0.2
        elif step < total_steps * 0.6:
            return 0.1
        elif step < total_steps * 0.9:
            return 0.0
        else:
            return -0.1  # Penalty for late decisions
    
    def _consistency_component(self, current_action: ModerationAction) -> float:
        """Reward for consistent decision patterns"""
        
        if len(self.history) < 2:
            return 0.0
        
        # Check last 3 actions
        last_actions = [h['action'] for h in self.history[-3:]]
        
        if len(last_actions) >= 3 and all(a == last_actions[0] for a in last_actions):
            # Very consistent
            return 0.15
        elif len(last_actions) >= 2 and current_action == last_actions[-1]:
            # Consistent with last
            return 0.05
        else:
            return -0.05  # Inconsistent
    
    def _exploration_component(self, action: ModerationAction, content: ContentItem) -> float:
        """Encourage exploration of uncertain content"""
        
        if not content.true_label:
            # Encourage trying different actions on uncertain content
            if len(self.history) < 5:
                return 0.1
        return 0.0
    
    def reset(self):
        """Reset reward function state"""
        self.history = []