#!/usr/bin/env python3

import os
import sys
from typing import Optional, List
from openai import OpenAI

from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4.1-mini")
HF_TOKEN = os.getenv("HF_TOKEN")
if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

client = OpenAI(
    base_url=API_BASE_URL,
    api_key=HF_TOKEN,
)

BENCHMARK = "content-moderation-openenv"
MAX_STEPS = 20
TEMPERATURE = 0.2
MAX_TOKENS = 50

SYSTEM_PROMPT = """You are an AI content moderator. Your task is to analyze content and decide:
- ALLOW: Content is safe and appropriate
- FLAG: Content is suspicious and needs human review
- REMOVE: Content violates policies

Respond with ONLY one word: ALLOW, FLAG, or REMOVE"""


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error={error if error else 'null'}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


def heuristic_decision(observation: dict) -> ModerationAction:
    text = observation.get("text", "").lower()

    violence_words = ["hate", "kill", "die", "murder", "violence", "blood"]
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE

    profanity_words = ["stupid", "idiot", "fuck", "shit", "bitch", "asshole"]
    for word in profanity_words:
        if word in text:
            return ModerationAction.FLAG

    return ModerationAction.ALLOW


def get_model_action(observation: dict) -> ModerationAction:
    text = observation.get("text", "")
    user_rep = observation.get("user_reputation", 0.5)
    if isinstance(user_rep, list):
        user_rep = user_rep[0] if user_rep else 0.5
    report_count = observation.get("report_count", 0)

    prompt = f"""Content: "{text}"
User reputation: {user_rep:.2f}/1.0
Report count: {report_count}

Choose: ALLOW, FLAG, or REMOVE"""

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        action_text = completion.choices[0].message.content.strip().upper()

        if action_text == "ALLOW":
            return ModerationAction.ALLOW
        elif action_text == "FLAG":
            return ModerationAction.FLAG
        elif action_text == "REMOVE":
            return ModerationAction.REMOVE
        return heuristic_decision(observation)

    except Exception as e:
        print(f"[DEBUG] API call failed: {e}", file=sys.stderr, flush=True)
        return heuristic_decision(observation)


def run_task(env: ModerationEnv) -> tuple[bool, int, List[float]]:
    rewards = []
    steps_taken = 0
    observation, info = env.reset()

    try:
        for step in range(1, MAX_STEPS + 1):
            action_enum = get_model_action(observation)
            action_str = action_enum.value

            observation, reward, done, _, info = env.step(action_enum)
            rewards.append(reward)
            steps_taken = step

            log_step(step=step, action=action_str, reward=reward, done=done, error=None)

            if done:
                break
    finally:
        env.close()

    success = any(r > 0 for r in rewards)
    return success, steps_taken, rewards


def main():
    tasks = [
        ("easy", "data/dataset_easy.json", 15),
        ("medium", "data/dataset_medium.json", 20),
        ("hard", "data/dataset_hard.json", 50),
    ]

    for task_name, dataset_path, max_steps in tasks:
        global MAX_STEPS
        MAX_STEPS = max_steps

        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        env = ModerationEnv(
            dataset_path=dataset_path,
            max_steps=max_steps,
            task_difficulty=task_name,
            render_mode=None,
        )

        success, steps, rewards = run_task(env)
        log_end(success=success, steps=steps, rewards=rewards)


if __name__ == "__main__":
    main()
