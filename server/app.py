#!/usr/bin/env python3
"""
OpenEnv Server Entry Point for Content Moderation Environment
This is the main entry point required for OpenEnv multi-mode deployment
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.moderation_env import ModerationEnv
from environment.models import Observation, Action, Reward
from tasks.easy_task import EasyTask
from tasks.medium_task import MediumTask
from tasks.hard_task import HardTask
from graders.grader import ModerationGrader


def create_env(dataset_path: str = "data/dataset_easy.json", 
               max_steps: int = 20,
               task_difficulty: str = "easy") -> ModerationEnv:
    """
    Factory function to create the environment.
    This is the entry point for OpenEnv.
    """
    return ModerationEnv(
        dataset_path=dataset_path,
        max_steps=max_steps,
        task_difficulty=task_difficulty
    )


def get_tasks():
    """
    Return available tasks for the environment.
    Required by OpenEnv.
    """
    return {
        "easy": {
            "name": "Basic Toxicity Detection",
            "description": "Detect clearly toxic content with high accuracy",
            "grader": ModerationGrader,
            "max_score": 1.0
        },
        "medium": {
            "name": "Ambiguous Content Detection",
            "description": "Handle sarcasm, mixed language, and context-dependent content",
            "grader": ModerationGrader,
            "max_score": 1.0
        },
        "hard": {
            "name": "Large-Scale Moderation",
            "description": "Process 50+ posts with time constraints",
            "grader": ModerationGrader,
            "max_score": 1.0
        }
    }


def get_action_space():
    """
    Return action space definition.
    Required by OpenEnv.
    """
    return {
        "type": "Discrete",
        "n": 3,
        "mapping": {
            0: "ALLOW",
            1: "FLAG",
            2: "REMOVE"
        }
    }


def get_observation_space():
    """
    Return observation space definition.
    Required by OpenEnv.
    """
    return {
        "text": {"type": "string", "max_length": 1000},
        "user_reputation": {"type": "float", "range": [0, 1]},
        "report_count": {"type": "integer", "range": [0, 100]},
        "policy_summary": {"type": "string", "max_length": 500},
        "content_type": {"type": "integer", "range": [0, 3]}
    }


def main():
    """
    Main entry point for the OpenEnv server.
    This function is required by OpenEnv multi-mode deployment.
    """
    print("=" * 60)
    print("🚀 Content Moderation OpenEnv Server")
    print("=" * 60)
    print(f"Environment: content-moderation-openenv")
    print(f"Version: 1.0.0")
    print(f"Tasks: easy, medium, hard")
    print("=" * 60)
    
    # Quick test to verify environment works
    try:
        env = create_env()
        obs, info = env.reset()
        print(f"✅ Environment initialized successfully")
        print(f"   Observation keys: {list(obs.keys())}")
        print(f"   Action space: {get_action_space()}")
        return 0
    except Exception as e:
        print(f"❌ Environment initialization failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
