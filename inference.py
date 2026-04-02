 
#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script
Required for hackathon submission

Reads credentials from environment:
- API_BASE_URL: The API endpoint for the LLM
- MODEL_NAME: The model identifier to use for inference
- HF_TOKEN: Your Hugging Face / API key
"""

import os
import sys
import json
from typing import List, Dict, Any
from openai import OpenAI
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask
from agents.baseline_agent import BaselineAgent

# Read environment variables (required for submission)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

# Configuration
MAX_STEPS = 8  # As per example
TEMPERATURE = 0.2
MAX_TOKENS = 200

def run_inference():
    """Run baseline inference on all three tasks"""
    
    print("="*60)
    print("OpenEnv Content Moderation - Baseline Inference")
    print(f"API Base URL: {API_BASE_URL}")
    print(f"Model: {MODEL_NAME}")
    print("="*60)
    
    # Check API key
    if not API_KEY:
        print("WARNING: No API key found. Using heuristic agent.")
        print("Set HF_TOKEN or OPENAI_API_KEY environment variable for OpenAI inference.")
    
    # Initialize agent
    agent = BaselineAgent(use_openai=bool(API_KEY))
    
    # Run tasks
    results = {}
    
    try:
        # Easy Task
        print("\n📋 Running Easy Task...")
        easy_task = EasyTask()
        results['easy'] = easy_task.run(agent)
        
        # Medium Task
        print("\n📋 Running Medium Task...")
        medium_task = MediumTask()
        results['medium'] = medium_task.run(agent)
        
        # Hard Task
        print("\n📋 Running Hard Task...")
        hard_task = HardTask()
        results['hard'] = hard_task.run(agent)
        
        # Calculate final score
        final_score = (results['easy'] * 0.3 + 
                       results['medium'] * 0.3 + 
                       results['hard'] * 0.4)
        
        # Print results
        print("\n" + "="*60)
        print("📊 BASELINE RESULTS")
        print("="*60)
        print(f"Easy Task:   {results['easy']:.3f}/1.0")
        print(f"Medium Task: {results['medium']:.3f}/1.0")
        print(f"Hard Task:   {results['hard']:.3f}/1.0")
        print("-"*60)
        print(f"FINAL SCORE: {final_score:.3f}/1.0")
        print("="*60)
        
        # Save results
        with open('baseline_results.json', 'w') as f:
            json.dump({
                'easy': results['easy'],
                'medium': results['medium'],
                'hard': results['hard'],
                'final': final_score,
                'config': {
                    'model': MODEL_NAME,
                    'api_base': API_BASE_URL
                }
            }, f, indent=2)
        
        print("\n✅ Results saved to baseline_results.json")
        
        return final_score
        
    except Exception as e:
        print(f"\n❌ Inference failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    run_inference()