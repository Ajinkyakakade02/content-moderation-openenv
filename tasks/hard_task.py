from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction
from graders.grader import ModerationGrader
import time
import numpy as np

class HardTask:
    """
    Hard Task: Large-Scale Moderation with Trade-offs
    Process many posts quickly while balancing precision and recall
    """
    
    def __init__(self):
        self.env = ModerationEnv(
            dataset_path='data/dataset_hard.json',
            max_steps=50,
            task_difficulty='hard'
        )
        self.grader = ModerationGrader('hard')
        self.time_limit = 2.0  # seconds per decision
        self.decision_times = []
    
    def run(self, agent) -> float:
        """Run task with time constraints and return score strictly between 0 and 1"""
        
        observation, info = self.env.reset()
        done = False
        self.grader.reset()
        self.decision_times = []
        
        print("\n" + "="*50)
        print("HARD TASK: Large-Scale Moderation")
        print(f"Time Limit: {self.time_limit}s per decision")
        print("="*50)
        
        while not done:
            start_time = time.time()
            
            # Get decision from agent
            action = agent.decide(observation)
            action_enum = ModerationAction(action) if isinstance(action, int) else action
            
            # Measure decision time
            decision_time = time.time() - start_time
            self.decision_times.append(decision_time)
            
            # Time penalty
            if decision_time > self.time_limit:
                print(f"❌ Time limit exceeded! {decision_time:.2f}s > {self.time_limit}s")
                break
            
            # Execute step
            observation, reward, done, _, info = self.env.step(action_enum)
            
            # Grade decision
            if self.env.current_content and hasattr(self.env.current_content, 'true_label'):
                self.grader.grade_decision(
                    action_enum.value,
                    self.env.current_content.true_label,
                    {'decision_time': decision_time}
                )
            
            print(f"Step {info['step']}: {action_enum.value} | Time: {decision_time:.2f}s | Reward: {reward:.2f}")
        
        # Calculate trade-off score
        base_score = self.grader.calculate_final_score()
        
        # Efficiency component
        avg_time = np.mean(self.decision_times) if self.decision_times else self.time_limit
        efficiency = max(0, 1 - (avg_time / self.time_limit))
        
        # Consistency component (less variety is better for scale)
        decisions = self.grader.get_decision_log()
        if decisions:
            actions = [d['action'] for d in decisions]
            unique_actions = len(set(actions))
            consistency = 1 - (unique_actions - 1) / 2
        else:
            consistency = 0
        
        # Final score balancing accuracy, efficiency, and consistency
        final_score = 0.5 * base_score + 0.3 * efficiency + 0.2 * consistency
        
        # Final safety clamp - NEVER return 0.0 or 1.0
        if final_score <= 0.0:
            final_score = 0.1
        if final_score >= 1.0:
            final_score = 0.9
        
        print(f"\n✅ Hard Task Score: {final_score:.3f}/1.0")
        print(f"   Base Accuracy: {base_score:.3f}")
        print(f"   Efficiency: {efficiency:.3f} (avg time: {avg_time:.2f}s)")
        print(f"   Consistency: {consistency:.3f}")
        
        return final_score
