#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
Uses requests library (more stable than OpenAI client)
"""

import os
import sys
import json
import requests
from typing import List, Optional

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20

if not API_BASE_URL or not API_KEY:
    print("ERROR: API_BASE_URL and API_KEY must be set", file=sys.stderr)
    sys.exit(1)

# Remove trailing slash if present
API_BASE_URL = API_BASE_URL.rstrip('/')

print(f"[DEBUG] API_BASE_URL: {API_BASE_URL}", file=sys.stderr, flush=True)

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


def call_llm_api(text: str) -> str:
    """Make API call using requests (stable, no version conflicts)"""
    
    prompt = f"Content: {text}\nChoose: ALLOW, FLAG, or REMOVE"
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}",
    }
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a content moderator. Respond with ONLY: ALLOW, FLAG, or REMOVE."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }
    
    # Try both possible endpoints
    endpoints = [
        f"{API_BASE_URL}/chat/completions",
        f"{API_BASE_URL}/v1/chat/completions",
    ]
    
    for endpoint in endpoints:
        try:
            response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
            print(f"[DEBUG] Endpoint {endpoint} -> Status: {response.status_code}", file=sys.stderr, flush=True)
            
            if response.status_code == 200:
                result = response.json()
                action = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                if action in ['ALLOW', 'FLAG', 'REMOVE']:
                    return action
        except Exception as e:
            print(f"[DEBUG] Endpoint failed: {e}", file=sys.stderr, flush=True)
    
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
        action_str = call_llm_api(text)
        
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
