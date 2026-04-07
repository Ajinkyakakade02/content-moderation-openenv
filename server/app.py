#!/usr/bin/env python3
"""
OpenEnv Server Entry Point for Content Moderation Environment
"""

import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from environment.moderation_env import ModerationEnv
from environment.models import Observation, Action, Reward


def create_env(dataset_path: str = "data/dataset_easy.json", 
               max_steps: int = 20,
               task_difficulty: str = "easy"):
    """Create and return the environment instance."""
    return ModerationEnv(
        dataset_path=dataset_path,
        max_steps=max_steps,
        task_difficulty=task_difficulty
    )


def main():
    """Main entry point - required by OpenEnv."""
    print("=" * 60)
    print("Content Moderation OpenEnv Server")
    print("=" * 60)
    
    env = create_env()
    obs, info = env.reset()
    
    print(f"✅ Environment ready")
    print(f"   Observation space: {list(obs.keys())}")
    print(f"   Action space: 0=ALLOW, 1=FLAG, 2=REMOVE")
    print(f"   Max steps: {env.max_steps}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
