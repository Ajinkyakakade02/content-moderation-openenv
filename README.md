# 🛡️ AI Content Moderation OpenEnv

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)](https://github.com/openenv)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://python.org)
[![Docker](https://img.shields.io/badge/Docker-Ready-blue)](https://docker.com)

A production-ready content moderation environment built for the Meta XScalar Hackathon. Simulates real-world AI moderation where agents must decide to ALLOW, FLAG, or REMOVE user content while balancing safety and false positives.

## 🎯 What This Environment Does

- **Real-time Content Moderation**: AI agent reviews text, images, videos, and reels
- **Harmful Content Detection**: Nudity, hate speech, violence, harassment, dangerous acts
- **3-Tier Decision System**: ALLOW (safe), FLAG (needs review), REMOVE (violates policy)
- **Multi-Language Support**: Handles English, Hindi, Hinglish mixed content
- **Time Constraints**: 2-second decisions at scale (Hard Task)

## 🏗️ Environment Architecture

Content Moderation Environment
├── Observation Space
│ ├── text: User content
│ ├── user_reputation: Trust score (0-1)
│ ├── report_count: Previous reports
│ └── policy_summary: Current guidelines
├── Action Space
│ ├── 0: ALLOW (approve content)
│ ├── 1: FLAG (human review)
│ └── 2: REMOVE (delete + warning)
└── Reward Function
├── Correct: +1.0
├── False Negative: -0.5 to -2.0
├── False Positive: -0.2 to -0.5
├── Efficiency Bonus: +0.1
└── Consistency Bonus: +0.05


## 📋 Tasks

| Task | Difficulty | Description | Max Score |
|------|------------|-------------|-----------|
| Basic Toxicity Detection | Easy | Detect hate speech, nudity, violence | 1.0 |
| Ambiguous Content | Medium | Sarcasm, Hinglish, context-dependent | 1.0 |
| Large-Scale Moderation | Hard | 50+ posts, 2s limit, precision vs recall | 1.0 |

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- Docker (optional)
- OpenAI API key (for baseline agent)

### Installation
```bash
# Clone repository
git clone https://github.com/yourteam/content-moderation-openenv
cd content-moderation-openenv

# Install dependencies
pip install -r requirements.txt

# Set environment variables
export HF_TOKEN="your_api_key"
export MODEL_NAME="gpt-3.5-turbo"
export API_BASE_URL="https://api.openai.com/v1"

# Run baseline inference
python inference.py