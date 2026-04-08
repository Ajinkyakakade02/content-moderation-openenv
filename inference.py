#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
Uses heuristic-only mode (no external API calls)
"""

import os
import sys
from typing import List, Optional

BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20

# Import environment
from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def heuristic_decision(observation: dict) -> ModerationAction:
    """Heuristic-based decision making"""
    text = observation.get('text', '').lower()
    
    # Violence keywords -> REMOVE
    violence_words = ['hate', 'kill', 'die', 'murder', 'violence', 'blood', 'attack', 'hurt']
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE
    
    # Profanity/harassment -> FLAG
    profanity_words = ['stupid', 'idiot', 'fuck', 'shit', 'bitch', 'asshole', 'damn', 'crap']
    for word in profanity_words:
        if word in text:
            return ModerationAction.FLAG
    
    # Default -> ALLOW
    return ModerationAction.ALLOW


def run_task(env: ModerationEnv, task_name: str) -> tuple:
    """Run a single task and return results"""
    rewards = []
    steps_taken = 0
    
    observation, info = env.reset()
    
    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break
        
        action_enum = heuristic_decision(observation)
        action_str = action_enum.value
        
        observation, reward, done, _, info = env.step(action_enum)
        rewards.append(reward)
        steps_taken = step
        
        log_step(step=step, action=action_str, reward=reward, done=done, error=None)
        
        if done:
            break
    
    total_reward = sum(rewards)
    max_possible = env.max_steps * 1.0
    score = min(max(total_reward / max_possible, 0.0), 1.0) if max_possible > 0 else 0.0
    success = score >= 0.5
    
    return success, steps_taken, score, rewards


def main():
    """Run all 3 tasks with proper logging format"""
    
    tasks = [
        ('easy', 'data/dataset_easy.json', 15),
        ('medium', 'data/dataset_medium.json', 20),
        ('hard', 'data/dataset_hard.json', 50),
    ]
    
    for task_name, dataset_path, max_steps in tasks:
        log_start(task=task_name, env=BENCHMARK, model="heuristic")
        
        env = ModerationEnv(
            dataset_path=dataset_path,
            max_steps=max_steps,
            task_difficulty=task_name,
            render_mode=None
        )
        
        success, steps, score, rewards = run_task(env, task_name)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
