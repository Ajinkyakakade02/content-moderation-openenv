#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20

if not API_BASE_URL or not API_KEY:
    print("ERROR: API_BASE_URL and API_KEY must be set", file=sys.stderr)
    sys.exit(1)

# Initialize OpenAI client (MANDATORY for hackathon)
client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)

from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error_val}", flush=True)


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}", flush=True)


def get_action_from_llm(text: str) -> str:
    """Get action from LLM using OpenAI client"""
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": "You are a content moderator. Respond with ONLY: ALLOW, FLAG, or REMOVE."},
                {"role": "user", "content": f"Content: {text}\nChoose: ALLOW, FLAG, or REMOVE"}
            ],
            max_tokens=10,
            temperature=0,
        )
        return response.choices[0].message.content.strip().upper()
    except Exception as e:
        print(f"[DEBUG] API error: {e}", file=sys.stderr)
        # Fallback to heuristic
        text_lower = text.lower()
        if any(w in text_lower for w in ['hate', 'kill', 'die', 'violence', 'blood']):
            return 'REMOVE'
        if any(w in text_lower for w in ['stupid', 'idiot', 'fuck', 'shit']):
            return 'FLAG'
        return 'ALLOW'


def run_task(env: ModerationEnv) -> tuple:
    rewards = []
    steps_taken = 0
    observation, _ = env.reset()

    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break

        text = observation.get('text', '')
        action_str = get_action_from_llm(text)
        
        # Map string to ModerationAction
        if action_str == 'ALLOW':
            action_enum = ModerationAction.ALLOW
        elif action_str == 'FLAG':
            action_enum = ModerationAction.FLAG
        else:
            action_enum = ModerationAction.REMOVE

        observation, reward, done, _, _ = env.step(action_enum)
        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_str, reward=reward, done=done, error=None)

        if done:
            break

    max_possible = env.max_steps * 1.0
    score = min(max(sum(rewards) / max_possible, 0.0), 1.0) if max_possible > 0 else 0.0
    return score >= 0.5, steps_taken, score, rewards


def main():
    tasks = [
        ('easy', 'data/dataset_easy.json', 15),
        ('medium', 'data/dataset_medium.json', 20),
        ('hard', 'data/dataset_hard.json', 50),
    ]

    for task_name, dataset_path, max_steps in tasks:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
        # FIXED: Use keyword arguments properly
        env = ModerationEnv(
            dataset_path=dataset_path,
            max_steps=max_steps,
            task_difficulty=task_name,
            render_mode=None
        )
        
        success, steps, score, rewards = run_task(env)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
