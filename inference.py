#!/usr/bin/env python3
"""
OpenEnv Baseline Inference Script for Content Moderation Environment
Follows strict [START]/[STEP]/[END] format required for hackathon

MANDATORY Environment Variables:
- API_BASE_URL: The API endpoint for the LLM
- MODEL_NAME: The model identifier to use for inference  
- HF_TOKEN: Your Hugging Face / API key
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI

# Read environment variables (required for submission)
API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
API_KEY = os.getenv("HF_TOKEN") or os.getenv("OPENAI_API_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")
BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 50

# Import environment
from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction

SYSTEM_PROMPT = """You are an AI content moderator. Your task is to analyze content and decide:
- ALLOW: Content is safe and appropriate
- FLAG: Content is suspicious and needs human review  
- REMOVE: Content violates policies (hate speech, nudity, violence)

Respond with ONLY one word: ALLOW, FLAG, or REMOVE"""


def log_start(task: str, env: str, model: str) -> None:
    """Emit [START] line - REQUIRED FORMAT"""
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    """Emit [STEP] line - REQUIRED FORMAT"""
    error_val = error if error else "null"
    done_val = str(done).lower()
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={done_val} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    """Emit [END] line - REQUIRED FORMAT"""
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def get_model_action(client: OpenAI, observation: dict) -> str:
    """Get action from OpenAI model"""
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
        if action_text not in ['ALLOW', 'FLAG', 'REMOVE']:
            return 'ALLOW'
        return action_text
    except Exception as e:
        # Fallback to heuristic if API fails
        return heuristic_decision(observation)


def heuristic_decision(observation: dict) -> str:
    """Fallback heuristic when API is unavailable"""
    text = observation.get('text', '').lower()
    toxic_words = ['hate', 'kill', 'die', 'stupid', 'idiot', 'nude', 'sex', 'violence']
    
    for word in toxic_words:
        if word in text:
            if word in ['hate', 'kill', 'die']:
                return 'REMOVE'
            elif word in ['stupid', 'idiot']:
                return 'FLAG'
    return 'ALLOW'


def run_task(env: ModerationEnv, task_name: str, client: Optional[OpenAI]) -> tuple:
    """Run a single task and return results"""
    rewards = []
    actions = []
    steps_taken = 0
    success = False
    
    observation, info = env.reset()
    
    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break
        
        # Get action from model (or heuristic if API not available)
        if client and API_KEY:
            action_str = get_model_action(client, observation)
        else:
            action_str = heuristic_decision(observation)
        
        # Map action to integer
        action_map = {'ALLOW': 0, 'FLAG': 1, 'REMOVE': 2}
        action = action_map.get(action_str, 0)
        
        # Take step in environment
        observation, reward, done, _, info = env.step(action)
        rewards.append(reward)
        actions.append(action_str)
        steps_taken = step
        
        error = None
        log_step(step=step, action=action_str, reward=reward, done=done, error=error)
        
        if done:
            break
    
    # Calculate normalized score (0.0 to 1.0)
    total_reward = sum(rewards)
    max_possible = env.max_steps * 1.0  # Max 1.0 per step
    score = min(max(total_reward / max_possible, 0.0), 1.0)
    success = score >= 0.5
    
    return success, steps_taken, score, rewards


def main():
    """Run all 3 tasks with proper logging format"""
    
    # Initialize OpenAI client only if API key is available
    client = None
    if API_KEY:
        try:
            client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
        except Exception as e:
            print(f"[DEBUG] Failed to initialize OpenAI client: {e}", flush=True)
    
    # Define tasks with their configurations
    tasks = [
        ('easy', 'data/dataset_easy.json', 15),
        ('medium', 'data/dataset_medium.json', 20),
        ('hard', 'data/dataset_hard.json', 50),
    ]
    
    all_success = True
    total_score = 0.0
    
    for task_name, dataset_path, max_steps in tasks:
        # Log start of task
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME if API_KEY else "heuristic")
        
        # Create environment for this task
        env = ModerationEnv(
            dataset_path=dataset_path,
            max_steps=max_steps,
            task_difficulty=task_name
        )
        
        try:
            success, steps, score, rewards = run_task(env, task_name, client)
            all_success = all_success and success
            total_score += score
            log_end(success=success, steps=steps, score=score, rewards=rewards)
        except Exception as e:
            log_end(success=False, steps=0, score=0.0, rewards=[])
            print(f"[DEBUG] Task {task_name} failed: {e}", flush=True)
        finally:
            # Clean up
            pass


if __name__ == "__main__":
    main()
