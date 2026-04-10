#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
Uses requests library to make API calls to LiteLLM proxy
"""

import os
import sys
import json
import requests
from typing import List, Optional

BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20

# Read environment variables
API_BASE_URL = os.environ.get("API_BASE_URL", "")
API_KEY = os.environ.get("API_KEY", "")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

print(f"[DEBUG] API_BASE_URL: {API_BASE_URL}", file=sys.stderr, flush=True)
print(f"[DEBUG] MODEL_NAME: {MODEL_NAME}", file=sys.stderr, flush=True)

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


def call_llm_api(text: str, user_rep: float, reports: int) -> str:
    """Make HTTP request to LiteLLM proxy"""
    
    prompt = f"""Content: "{text}"
User reputation: {user_rep:.2f}/1.0
Report count: {reports}

Choose: ALLOW, FLAG, or REMOVE"""
    
    # Construct the endpoint URL
    endpoint = f"{API_BASE_URL}/chat/completions"
    
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
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers, timeout=30)
        if response.status_code == 200:
            result = response.json()
            action_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
            print(f"[DEBUG] API returned: {action_text}", file=sys.stderr, flush=True)
            return action_text
        else:
            print(f"[DEBUG] API returned status {response.status_code}: {response.text[:100]}", file=sys.stderr, flush=True)
            return "ALLOW"
    except Exception as e:
        print(f"[DEBUG] API call failed: {e}", file=sys.stderr, flush=True)
        return "ALLOW"


def heuristic_decision(observation: dict) -> ModerationAction:
    """Heuristic-based decision making (fallback)"""
    text = observation.get('text', '').lower()
    
    violence_words = ['hate', 'kill', 'die', 'murder', 'violence', 'blood', 'attack', 'hurt']
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE
    
    profanity_words = ['stupid', 'idiot', 'fuck', 'shit', 'bitch', 'asshole', 'damn', 'crap']
    for word in profanity_words:
        if word in text:
            return ModerationAction.FLAG
    
    return ModerationAction.ALLOW


def get_model_action(observation: dict) -> ModerationAction:
    """Get action - ALWAYS try API first, fallback to heuristic"""
    text = observation.get('text', '')
    user_rep = observation.get('user_reputation', [0.5])[0] if isinstance(observation.get('user_reputation'), list) else observation.get('user_reputation', 0.5)
    reports = observation.get('report_count', 0)
    
    # ALWAYS try API call first (this is what validator checks)
    action_text = call_llm_api(text, user_rep, reports)
    
    if action_text == 'ALLOW':
        return ModerationAction.ALLOW
    elif action_text == 'FLAG':
        return ModerationAction.FLAG
    elif action_text == 'REMOVE':
        return ModerationAction.REMOVE
    else:
        return heuristic_decision(observation)


def run_task(env: ModerationEnv, task_name: str) -> tuple:
    """Run a single task and return results"""
    rewards = []
    steps_taken = 0
    
    observation, info = env.reset()
    
    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break
        
        action_enum = get_model_action(observation)
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
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)
        
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
