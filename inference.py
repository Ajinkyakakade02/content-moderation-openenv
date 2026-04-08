#!/usr/bin/env python3
"""
OpenEnv Inference Script for Content Moderation Environment
Makes REAL API calls through the hackathon's LiteLLM proxy.
"""

import os
import sys
from typing import List, Optional
from openai import OpenAI

# ─── Step 1: Read hackathon-injected environment variables ───────────────────
# These are overwritten by the validator at runtime — your HF Space values
# are just fallbacks for local testing.
API_BASE_URL = os.environ.get("API_BASE_URL", "").rstrip("/")
API_KEY      = os.environ.get("API_KEY", "no-key")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-3.5-turbo")

# ─── Step 2: Initialize OpenAI client pointed at THEIR proxy ─────────────────
# The SDK automatically appends /chat/completions to API_BASE_URL.
# This is the call the validator's proxy logs will record.
client = OpenAI(
    base_url=API_BASE_URL if API_BASE_URL else None,
    api_key=API_KEY,
)

BENCHMARK = "content-moderation-openenv"
MAX_STEPS  = 20

# ─── Step 3: Import your environment ─────────────────────────────────────────
from environment.moderation_env import ModerationEnv
from environment.models import ModerationAction


# ─── Logging helpers (format required by hackathon) ──────────────────────────
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


# ─── LLM decision: makes a REAL API call through their proxy ─────────────────
def llm_decision(observation: dict) -> ModerationAction:
    """
    Primary decision function — calls the hackathon LiteLLM proxy.
    This is what causes a log entry on their server.
    Falls back to heuristic ONLY if the HTTP call raises an exception.
    """
    text            = observation.get("text", "")
    user_reputation = observation.get("user_reputation", 0.5)
    report_count    = observation.get("report_count", 0)
    policy_summary  = observation.get("policy_summary", "")

    system_prompt = (
        "You are a strict content moderation assistant for a social platform. "
        "Review the content and context below, then decide: ALLOW, FLAG, or REMOVE.\n\n"
        "Rules:\n"
        "- ALLOW: Safe content, no policy violation.\n"
        "- FLAG: Borderline or ambiguous — send for human review.\n"
        "- REMOVE: Clear violation — hate speech, nudity, graphic violence, threats.\n\n"
        f"Platform policy: {policy_summary}\n\n"
        "Reply with exactly one word: ALLOW, FLAG, or REMOVE. No explanation."
    )

    user_message = (
        f"Content to moderate:\n\"{text}\"\n\n"
        f"User reputation score: {user_reputation:.2f} (0=spammer, 1=trusted)\n"
        f"Number of prior reports on this post: {report_count}\n\n"
        "Your decision (ALLOW / FLAG / REMOVE):"
    )

    try:
        # ── THIS LINE is what the validator proxy records ──────────────────
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

        raw = response.choices[0].message.content.strip().upper()

        if "REMOVE" in raw:
            return ModerationAction.REMOVE
        elif "FLAG" in raw:
            return ModerationAction.FLAG
        else:
            return ModerationAction.ALLOW

    except Exception as api_error:
        # Log the error so you can debug post-submission
        print(f"[WARN] LLM API call failed ({api_error}). "
              "Using heuristic fallback.", flush=True)
        return _heuristic_fallback(observation)


# ─── Heuristic fallback — used ONLY when the API call itself fails ────────────
def _heuristic_fallback(observation: dict) -> ModerationAction:
    text = observation.get("text", "").lower()

    violence_words = [
        "kill", "murder", "die", "death", "blood", "gore", "attack",
        "stab", "shoot", "bomb", "explode", "torture", "abuse", "assault",
    ]
    for word in violence_words:
        if word in text:
            return ModerationAction.REMOVE

    nsfw_words = [
        "nude", "naked", "nudity", "porn", "sex", "xxx", "nsfw",
        "explicit", "erotic",
    ]
    for word in nsfw_words:
        if word in text:
            return ModerationAction.REMOVE

    hate_words = [
        "hate", "fuck", "shit", "bitch", "asshole", "bastard",
        "stupid", "idiot", "moron", "worthless", "retard",
    ]
    for word in hate_words:
        if word in text:
            return ModerationAction.FLAG

    return ModerationAction.ALLOW


# ─── Task runner ──────────────────────────────────────────────────────────────
def run_task(env: ModerationEnv, task_name: str, max_steps: int) -> tuple:
    rewards     = []
    steps_taken = 0

    observation, info = env.reset()

    for step in range(1, max_steps + 1):
        if env.current_step >= env.max_steps:
            break

        # ← Always goes through LLM first
        action_enum = llm_decision(observation)
        action_str  = action_enum.value

        observation, reward, done, _, info = env.step(action_enum)
        rewards.append(reward)
        steps_taken = step

        log_step(step=step, action=action_str,
                 reward=reward, done=done, error=None)

        if done:
            break

    total_reward = sum(rewards)
    max_possible = env.max_steps * 1.0
    score        = min(max(total_reward / max_possible, 0.0), 1.0) if max_possible > 0 else 0.0
    success      = score >= 0.5

    return success, steps_taken, score, rewards


# ─── Entry point ──────────────────────────────────────────────────────────────
def main():
    # Confirm env vars are loaded (safe to leave in — helps debug)
    print(f"[INFO] API_BASE_URL = {API_BASE_URL or 'NOT SET'}", flush=True)
    print(f"[INFO] MODEL_NAME   = {MODEL_NAME}", flush=True)
    print(f"[INFO] API_KEY      = {'SET' if API_KEY != 'no-key' else 'NOT SET'}", flush=True)

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

        success, steps, score, rewards = run_task(env, task_name, max_steps)
        log_end(success=success, steps=steps, score=score, rewards=rewards)


if __name__ == "__main__":
    main()
