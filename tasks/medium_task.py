 
from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction
from graders.grader import ModerationGrader
import numpy as np

class MediumTask:
    """
    Medium Task: Ambiguous Content Detection
    Handle sarcasm, mixed language, and context-dependent content
    """
    
    def __init__(self):
        self.env = ModerationEnv(
            dataset_path='data/dataset_medium.json',
            max_steps=20,
            task_difficulty='medium'
        )
        self.grader = ModerationGrader('medium')
        self.ambiguity_scores = []
    
    def run(self, agent) -> float:
        """Run task with agent and return score 0.0-1.0"""
        
        observation, info = self.env.reset()
        done = False
        self.grader.reset()
        self.ambiguity_scores = []
        
        print("\n" + "="*50)
        print("MEDIUM TASK: Ambiguous Content Detection")
        print("="*50)
        
        while not done:
            # Calculate content ambiguity
            ambiguity = self._calculate_ambiguity(observation)
            self.ambiguity_scores.append(ambiguity)
            
            # Get action from agent with ambiguity context
            action = agent.decide(observation, context={'ambiguity': ambiguity})
            action_enum = ModerationAction(action) if isinstance(action, int) else action
            
            # Execute step
            observation, reward, done, _, info = self.env.step(action)
            
            # Grade decision
            if self.env.current_content and hasattr(self.env.current_content, 'true_label'):
                self.grader.grade_decision(
                    action_enum.value,
                    self.env.current_content.true_label,
                    {'ambiguity': ambiguity}
                )
            
            print(f"Step {info['step']}: {action_enum.value} | Ambiguity: {ambiguity:.2f} | Reward: {reward:.2f}")
        
        # Calculate score with ambiguity penalty
        base_score = self.grader.calculate_final_score()
        avg_ambiguity = np.mean(self.ambiguity_scores) if self.ambiguity_scores else 0
        
        # Penalize for mishandling ambiguous content
        ambiguity_penalty = avg_ambiguity * 0.2
        final_score = max(0.0, min(1.0, base_score * (1 - ambiguity_penalty)))
        
        print(f"\n✅ Medium Task Score: {final_score:.3f}/1.0")
        print(f"   Base: {base_score:.3f} | Ambiguity Penalty: {ambiguity_penalty:.3f}")
        
        return final_score
    
    def _calculate_ambiguity(self, observation: dict) -> float:
        """Calculate ambiguity score for current content"""
        
        text = observation.get('text', '').lower()
        
        # Ambiguous patterns
        ambiguous_patterns = [
            'sarcasm', 'actually', 'technically', 'obviously', '🙄',
            'just', 'opinion', 'maybe', 'possibly', 'ironically',
            'not saying', 'but actually', 'supposedly'
        ]
        
        # Count ambiguous indicators
        pattern_count = sum(1 for pattern in ambiguous_patterns if pattern in text)
        
        # Mixed language detection (non-ASCII)
        has_mixed_lang = any(ord(c) > 127 for c in text)
        
        # Calculate ambiguity (0-1)
        ambiguity = (pattern_count / 10) + (0.2 if has_mixed_lang else 0)
        
        return min(1.0, ambiguity)