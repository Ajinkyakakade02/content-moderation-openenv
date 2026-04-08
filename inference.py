#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 50

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


def get_model_action(client: OpenAI, observation: dict) -> ModerationAction:
    text = observation.get('text', '')
    user_rep = observation.get('user_reputation', [0.5])[0] if isinstance(observation.get('user_reputation'), list) else observation.get('user_reputation', 0.5)
    reports = observation.get('report_count', 0)
    
    prompt = f"""Content: "{text}"
User reputation: {user_rep:.2f}/1.0
Report count: {reports}

Choose: ALLOW, FLAG, or REMOVE"""
    
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        action_text = completion.choices[0].message.content.strip().upper()
        if action_text == 'ALLOW':
            return ModerationAction.ALLOW
        elif action_text == 'FLAG':
            return ModerationAction.FLAG
        elif action_text == 'REMOVE':
            return ModerationAction.REMOVE
        else:
            return ModerationAction.ALLOW
    except Exception as e:
        return heuristic_decision(observation)


def heuristic_decision(observation: dict) -> ModerationAction:
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


def run_task(env: ModerationEnv, task_name: str, client: Optional[OpenAI]) -> tuple:
    rewards = []
    steps_taken = 0
    success = False
    
    observation, info = env.reset()
    
    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break
        
        if client and API_KEY:
            action_enum = get_model_action(client, observation)
        else:
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
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as e:
            pass
    
    tasks = [
        ('easy', 'data/dataset_easy.json', 15),
        ('medium', 'data/dataset_medium.json', 20),
        ('hard', 'data/dataset_hard.json', 50),
    ]
    
    for task_name, dataset_path, max_steps in tasks:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME if API_KEY else "heuristic")
        
        try:
            # CRITICAL: Set render_mode=None to suppress extra output
            env = ModerationEnv(
                dataset_path=dataset_path,
                max_steps=max_steps,
                task_difficulty=task_name,
                render_mode=None  # ← SILENT MODE - NO EXTRA PRINTS
            )
            
            success, steps, score, rewards = run_task(env, task_name, client)
            log_end(success=success, steps=steps, score=score, rewards=rewards)
        except Exception as e:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            import sys
            print(f"DEBUG: Task {task_name} failed: {e}", file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
