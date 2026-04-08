#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
Follows strict [START]/[STEP]/[END] format required for hackathon
"""

import os
import sys
import json
import urllib.request
import urllib.error
from typing import List, Optional

# ============================================
# Read environment variables (injected by hackathon)
# ============================================
API_BASE_URL = os.environ.get("API_BASE_URL")
API_KEY = os.environ.get("API_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 50

# ============================================
# Validate credentials
# ============================================
if not API_BASE_URL:
    print("[DEBUG] ERROR: API_BASE_URL not set", file=sys.stderr, flush=True)
    sys.exit(1)

if not API_KEY:
    print("[DEBUG] ERROR: API_KEY not set", file=sys.stderr, flush=True)
    sys.exit(1)

print(f"[DEBUG] Using API_BASE_URL: {API_BASE_URL}", file=sys.stderr, flush=True)
print(f"[DEBUG] Using MODEL_NAME: {MODEL_NAME}", file=sys.stderr, flush=True)

# Import environment
from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction

SYSTEM_PROMPT = """You are an AI content moderator. Your task is to analyze content and decide:
- ALLOW: Content is safe and appropriate
- FLAG: Content is suspicious and needs human review  
- REMOVE: Content violates policies (hate speech, nudity, violence)

Respond with ONLY one word: ALLOW, FLAG, or REMOVE"""


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


def call_llm(prompt: str) -> str:
    """Make HTTP request using urllib (no external dependencies)"""
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
    }
    
    endpoints_to_try = [
        f"{API_BASE_URL}/v1/chat/completions",
        f"{API_BASE_URL}/chat/completions",
    ]
    
    data = json.dumps(payload).encode('utf-8')
    
    for endpoint in endpoints_to_try:
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
                action_text = result.get("choices", [{}])[0].get("message", {}).get("content", "").strip().upper()
                if action_text in ['ALLOW', 'FLAG', 'REMOVE']:
                    return action_text
        except Exception as e:
            print(f"[DEBUG] Endpoint {endpoint} failed: {e}", file=sys.stderr, flush=True)
            continue
    
    return "ALLOW"  # Default fallback


def get_model_action(observation: dict) -> ModerationAction:
    """Get action from LLM using manual HTTP request"""
    text = observation.get('text', '')
    user_rep = observation.get('user_reputation', [0.5])[0] if isinstance(observation.get('user_reputation'), list) else observation.get('user_reputation', 0.5)
    reports = observation.get('report_count', 0)
    
    prompt = f"""Content: "{text}"
User reputation: {user_rep:.2f}/1.0
Report count: {reports}

Choose: ALLOW, FLAG, or REMOVE"""
    
    try:
        action_text = call_llm(prompt)
        if action_text == 'ALLOW':
            return ModerationAction.ALLOW
        elif action_text == 'FLAG':
            return ModerationAction.FLAG
        elif action_text == 'REMOVE':
            return ModerationAction.REMOVE
        else:
            return ModerationAction.ALLOW
    except Exception as e:
        print(f"[DEBUG] API call failed: {e}", file=sys.stderr, flush=True)
        return heuristic_decision(observation)


def heuristic_decision(observation: dict) -> ModerationAction:
    """Fallback heuristic when API is unavailable"""
    text = observation.get('text', '').lower()
    
    violence_words = ['hate', 'kill', 'die', 'murder', 'violence', 'blood']
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE
    
    profanity_words = ['stupid', 'idiot', 'fuck', 'shit', 'bitch', 'asshole']
    for word in profanity_words:
        if word in text:
            return ModerationAction.FLAG
    
    return ModerationAction.ALLOW


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
