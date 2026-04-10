from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction
from graders.grader import ModerationGrader
import numpy as np

class EasyTask:
    """
    Easy Task: Basic Toxicity Detection
    Detect clearly toxic content with high accuracy
    """
    
    def __init__(self):
        self.env = ModerationEnv(
            dataset_path='data/dataset_easy.json',
            max_steps=15,
            task_difficulty='easy'
        )
        self.grader = ModerationGrader('easy')
    
    def run(self, agent) -> float:
        """Run task with agent and return score strictly between 0 and 1"""
        
        observation, info = self.env.reset()
        done = False
        self.grader.reset()
        
        print("\n" + "="*50)
        print("EASY TASK: Basic Toxicity Detection")
        print("="*50)
        
        while not done:
            # Get action from agent
            action = agent.decide(observation)
            action_enum = ModerationAction(action) if isinstance(action, int) else action
            
            # Execute step - FIXED: use action_enum, not action
            observation, reward, done, _, info = self.env.step(action_enum)
            
            # Grade decision
            if self.env.current_content and hasattr(self.env.current_content, 'true_label'):
                self.grader.grade_decision(
                    action_enum.value,
                    self.env.current_content.true_label
                )
            
            print(f"Step {info['step']}: {action_enum.value} | Reward: {reward:.2f}")
        
        # Calculate final score
        score = self.grader.calculate_final_score()
        
        # CRITICAL FIX: Score must be strictly between 0 and 1
        if score <= 0.0:
            score = 0.001
        if score >= 1.0:
            score = 0.999
        
        print(f"\n✅ Easy Task Score: {score:.3f}/1.0")
        
        return score
