#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script - No external dependencies
Uses urllib (built-in) instead of requests
"""

import os
import sys
import json
import urllib.request
import urllib.error
from typing import List, Optional

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20

# Validate
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
    """Make API call using urllib (built-in, no extra dependencies)"""
    
    prompt = f"Content: {text}\nChoose: ALLOW, FLAG, or REMOVE"
    
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "You are a content moderator. Respond with ONLY: ALLOW, FLAG, or REMOVE."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 10,
        "temperature": 0,
    }
    
    endpoints = [
        f"{API_BASE_URL}/chat/completions",
        f"{API_BASE_URL}/v1/chat/completions",
    ]
    
    data = json.dumps(payload).encode('utf-8')
    
    for endpoint in endpoints:
        try:
            req = urllib.request.Request(
                endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {API_KEY}",
                },
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                result = json.loads(response.read().decode('utf-8'))
                action = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                if action in ['ALLOW', 'FLAG', 'REMOVE']:
                    return action
        except Exception as e:
            print(f"[DEBUG] Endpoint {endpoint} failed: {e}", file=sys.stderr, flush=True)
    
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
        
        action_map = {'ALLOW': 0, 'FLAG': 1, 'REMOVE': 2}
        action_enum = ModerationAction(action_map.get(action_str, 0))
        
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
        env = ModerationEnv(dataset_path, max_steps, task_name, render_mode=None)
        success, steps, score, rewards = run_task(env)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
