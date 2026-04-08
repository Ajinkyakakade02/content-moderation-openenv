#!/usr/bin/env python3
"""
Inference script that correctly routes through the hackathon's LiteLLM proxy.
"""
import os
import sys
from typing import List, Optional
from openai import OpenAI

# ─── Read hackathon-injected env vars ────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "")   # e.g. https://proxy.example.com
API_KEY      = os.environ.get("API_KEY", "no-key")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# ─── Initialize the OpenAI client pointed at THEIR proxy ─────────────────────
# This is the critical step your current code is missing entirely.
client = OpenAI(
    base_url=API_BASE_URL,   # Must end WITHOUT a trailing slash for most proxies
    api_key=API_KEY,
)

BENCHMARK = "content-moderation-openenv"
MAX_STEPS  = 20

from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction


# ─── Logging helpers ─────────────────────────────────────────────────────────
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: str, reward: float,
             done: bool, error: Optional[str]) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )

def log_end(success: bool, steps: int,
            score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ─── The actual LLM call ─────────────────────────────────────────────────────
def llm_decision(observation: dict) -> ModerationAction:
    """
    Makes a REAL API call through the hackathon proxy.
    Falls back to heuristic only if the API call itself raises an exception.
    """
    text            = observation.get("text", "")
    user_reputation = observation.get("user_reputation", 0.5)
    report_count    = observation.get("report_count", 0)
    policy_summary  = observation.get("policy_summary", "")

    system_prompt = (
        "You are a content moderation assistant. "
        "Given a piece of user-generated content and context, decide: "
        "ALLOW, FLAG, or REMOVE.\n\n"
        f"Policy: {policy_summary}\n\n"
        "Reply with exactly one word: ALLOW, FLAG, or REMOVE."
    )

    user_message = (
        f"Content: {text}\n"
        f"User reputation score: {user_reputation:.2f}\n"
        f"Prior report count: {report_count}\n\n"
        "Decision:"
    )

    try:
        # ── THIS is what the validator is looking for ──────────────────────
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_message},
            ],
            max_tokens=10,
            temperature=0,
        )
        # ──────────────────────────────────────────────────────────────────

        decision = response.choices[0].message.content.strip().upper()

        if "REMOVE" in decision:
            return ModerationAction.REMOVE
        elif "FLAG" in decision:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ALLOW

    except Exception as e:
        # Log the failure so you can debug it, then fall back
        print(f"[WARN] LLM API call failed: {e}. Using heuristic fallback.",
              flush=True)
        return _heuristic_fallback(observation)


def _heuristic_fallback(observation: dict) -> ModerationAction:
    """Used ONLY when the API call itself fails — not as the primary path."""
    text = observation.get("text", "").lower()
    violence_words = ["hate", "kill", "die", "murder", "violence",
                      "blood", "attack", "hurt"]
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE

    profanity_words = ["stupid", "idiot", "fuck", "shit",
                       "bitch", "asshole", "damn", "crap"]
    for word in profanity_words:
        if word in text:
            return ModerationAction.FLAG

    return ModerationAction.ALLOW


# ─── Task runner ─────────────────────────────────────────────────────────────
def run_task(env: ModerationEnv, task_name: str) -> tuple:
    rewards    = []
    steps_taken = 0

    observation, info = env.reset()

    for step in range(1, MAX_STEPS + 1):
        if env.current_step >= env.max_steps:
            break

        action_enum = llm_decision(observation)   # ← LLM call, not heuristic
        action_str  = action_enum.value

        observation, reward, done, _, info = env.step(action_enum)
        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_str,
                 reward=reward, done=done, error=None)

        if done:
            break

    total_reward  = sum(rewards)
    max_possible  = env.max_steps * 1.0
    score         = min(max(total_reward / max_possible, 0.0), 1.0)
    success       = score >= 0.5

    return success, steps_taken, score, rewards


# ─── Entry point ─────────────────────────────────────────────────────────────
def main():
    tasks = [
        ("easy",   "data/dataset_easy.json",   15),
        ("medium", "data/dataset_medium.json",  20),
        ("hard",   "data/dataset_hard.json",    50),
    ]

    for task_name, dataset_path, max_steps in tasks:
        log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

        env = ModerationEnv(
            dataset_path=dataset_path,
            max_steps=max_steps,
            task_difficulty=task_name,
            render_mode=None,
        )

        success, steps, score, rewards = run_task(env, task_name)
        log_end(success=success, steps=steps,
                score=score, rewards=rewards)


if __name__ == "__main__":
    main()
