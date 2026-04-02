import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI
from environment.models import ModerationAction

class BaselineAgent:
    """
    Baseline agent using OpenAI API
    Reads credentials from environment variables
    """
    
    def __init__(self, use_openai: bool = True):
        self.use_openai = use_openai
        
        if use_openai:
            # Read from environment variables
            self.api_base_url = os.getenv('API_BASE_URL', 'https://api.openai.com/v1')
            self.api_key = os.getenv('HF_TOKEN') or os.getenv('OPENAI_API_KEY')
            self.model_name = os.getenv('MODEL_NAME', 'gpt-3.5-turbo')
            
            if self.api_key:
                self.client = OpenAI(
                    base_url=self.api_base_url,
                    api_key=self.api_key
                )
            else:
                print("WARNING: No API key found. Using heuristic agent.")
                self.use_openai = False
        
        # Fallback heuristic keywords
        self.toxic_keywords = {
            'high': ['hate', 'kill', 'die', 'stupid', 'idiot', 'trash', 'worthless', 'nude', 'violence'],
            'medium': ['bad', 'sucks', 'terrible', 'awful', 'horrible'],
            'low': ['dislike', 'annoying', 'boring']
        }
    
    def decide(self, observation: Dict, context: Dict = None) -> ModerationAction:
        """
        Make moderation decision - RETURNS ModerationAction enum
        
        Returns:
            ModerationAction.ALLOW (0)
            ModerationAction.FLAG (1)
            ModerationAction.REMOVE (2)
        """
        
        if self.use_openai and hasattr(self, 'client') and self.api_key:
            return self._openai_decision(observation, context)
        else:
            return self._heuristic_decision(observation, context)
    
    def _openai_decision(self, observation: Dict, context: Dict = None) -> ModerationAction:
        """Use OpenAI for decision making"""
        
        try:
            text = observation.get('text', '')
            user_rep = observation.get('user_reputation', [0.5])[0] if isinstance(observation.get('user_reputation'), list) else observation.get('user_reputation', 0.5)
            reports = observation.get('report_count', 0)
            
            prompt = f"""You are a content moderator. Classify the following content as:
- ALLOW (safe content)
- FLAG (potentially problematic, needs review)
- REMOVE (clearly violates policies)

Content: "{text}"
User reputation: {user_rep:.2f}/1.0
Report count: {reports}

Respond with ONLY the action name: ALLOW, FLAG, or REMOVE"""

            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a content moderation AI. Respond with exactly one word: ALLOW, FLAG, or REMOVE."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.1,
                max_tokens=10
            )
            
            action_text = response.choices[0].message.content.strip().upper()
            
            # Map to ModerationAction enum
            if action_text == 'ALLOW':
                return ModerationAction.ALLOW
            elif action_text == 'FLAG':
                return ModerationAction.FLAG
            elif action_text == 'REMOVE':
                return ModerationAction.REMOVE
            else:
                return ModerationAction.ALLOW
            
        except Exception as e:
            print(f"OpenAI error: {e}")
            return self._heuristic_decision(observation, context)
    
    def _heuristic_decision(self, observation: Dict, context: Dict = None) -> ModerationAction:
        """Fallback heuristic decision"""
        
        text = observation.get('text', '').lower()
        user_rep = observation.get('user_reputation', [0.5])[0] if isinstance(observation.get('user_reputation'), list) else observation.get('user_reputation', 0.5)
        reports = observation.get('report_count', 0)
        
        # Calculate toxicity score
        toxicity_score = 0.0
        
        # High toxicity keywords
        for word in self.toxic_keywords['high']:
            if word in text:
                toxicity_score += 0.35
        
        # Medium toxicity keywords
        for word in self.toxic_keywords['medium']:
            if word in text:
                toxicity_score += 0.2
        
        # Low toxicity keywords
        for word in self.toxic_keywords['low']:
            if word in text:
                toxicity_score += 0.1
        
        # Adjust based on user reputation
        if user_rep < 0.3:
            toxicity_score += 0.2
        elif user_rep > 0.7:
            toxicity_score -= 0.1
        
        # Adjust based on reports
        toxicity_score += min(reports / 10, 0.3)
        
        # Decision logic - return ModerationAction enum
        if toxicity_score > 0.7:
            return ModerationAction.REMOVE
        elif toxicity_score > 0.4:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ALLOW