 
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field

@dataclass
class GradingResult:
    """Structure for grading results"""
    score: float
    correct_decisions: int
    false_positives: int
    false_negatives: int
    total_decisions: int
    precision: float
    recall: float
    f1_score: float
    details: Dict

class ModerationGrader:
    """
    Deterministic grader for content moderation tasks
    Returns scores between 0.0 and 1.0
    """
    
    def __init__(self, task_name: str = "easy"):
        self.task_name = task_name
        self.reset()
    
    def reset(self):
        """Reset grader state"""
        self.correct_decisions = 0
        self.false_positives = 0
        self.false_negatives = 0
        self.total_decisions = 0
        self.decision_log = []
        
        # Confusion matrix
        self.confusion_matrix = {
            'ALLOW': {'ALLOW': 0, 'FLAG': 0, 'REMOVE': 0},
            'FLAG': {'ALLOW': 0, 'FLAG': 0, 'REMOVE': 0},
            'REMOVE': {'ALLOW': 0, 'FLAG': 0, 'REMOVE': 0}
        }
    
    def grade_decision(self, action: str, true_label: str, context: Dict = None) -> Tuple[float, bool]:
        """
        Grade a single decision
        
        Returns:
            point: Score contribution (-1 to 1)
            is_correct: Whether decision was correct
        """
        
        self.total_decisions += 1
        
        # Define correct action mapping
        correct_mapping = {
            'safe': 'ALLOW',
            'toxic': 'FLAG',
            'hate_speech': 'REMOVE',
            'nudity': 'REMOVE',
            'violence': 'REMOVE',
            'harassment': 'FLAG',
            'misinformation': 'FLAG',
            'spam': 'FLAG'
        }
        
        correct_action = correct_mapping.get(true_label, 'ALLOW')
        is_correct = (action == correct_action)
        
        # Update confusion matrix
        self.confusion_matrix[correct_action][action] += 1
        
        if is_correct:
            self.correct_decisions += 1
            point = 1.0
        else:
            # Categorize error type
            if action == 'ALLOW' and true_label in ['toxic', 'hate_speech', 'nudity', 'violence']:
                self.false_negatives += 1  # Missed harmful content
                point = -0.5
            elif action in ['FLAG', 'REMOVE'] and true_label == 'safe':
                self.false_positives += 1  # Wrongly flagged/removed safe content
                point = -0.2
            else:
                point = -0.3
        
        # Record decision
        self.decision_log.append({
            'action': action,
            'true_label': true_label,
            'correct': is_correct,
            'point': point,
            'context': context or {}
        })
        
        return point, is_correct
    
    def calculate_final_score(self) -> float:
        """
        Calculate final normalized score between 0 and 1
        Uses weighted F1 score with penalties
        """
        
        if self.total_decisions == 0:
            return 0.0
        
        # Calculate metrics
        tp = self.correct_decisions
        fp = self.false_positives
        fn = self.false_negatives
        
        # Precision and Recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # F1 Score
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        # Accuracy
        accuracy = self.correct_decisions / self.total_decisions
        
        # Penalties (heavier for false negatives)
        false_positive_rate = self.false_positives / self.total_decisions
        false_negative_rate = self.false_negatives / self.total_decisions
        
        # Weighted score
        weighted_score = accuracy - (0.2 * false_positive_rate) - (0.5 * false_negative_rate)
        
        # Final score (0-1)
        final_score = max(0.0, min(1.0, weighted_score))
        
        return final_score
    
    def get_detailed_report(self) -> GradingResult:
        """Get detailed grading report"""
        
        if self.total_decisions == 0:
            return GradingResult(
                score=0.0,
                correct_decisions=0,
                false_positives=0,
                false_negatives=0,
                total_decisions=0,
                precision=0.0,
                recall=0.0,
                f1_score=0.0,
                details={}
            )
        
        tp = self.correct_decisions
        fp = self.false_positives
        fn = self.false_negatives
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return GradingResult(
            score=self.calculate_final_score(),
            correct_decisions=self.correct_decisions,
            false_positives=self.false_positives,
            false_negatives=self.false_negatives,
            total_decisions=self.total_decisions,
            precision=precision,
            recall=recall,
            f1_score=f1,
            details={
                'confusion_matrix': self.confusion_matrix,
                'accuracy': self.correct_decisions / self.total_decisions,
                'false_positive_rate': self.false_positives / self.total_decisions,
                'false_negative_rate': self.false_negatives / self.total_decisions
            }
        )
    
    def get_decision_log(self) -> List[Dict]:
        """Return decision log"""
        return self.decision_log