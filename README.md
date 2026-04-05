---
title: AI Content Moderation System
emoji: 🛡️
colorFrom: blue
colorTo: purple
sdk: docker
app_file: backend/api_server.py
pinned: false
---

# 🛡️ AI Content Moderation System

Automatically detects violence, nudity, hate speech, and inappropriate content in text, images, and videos.

## Features

- 🔪 **Violence Detection** - Fighting, weapons, blood
- 🔞 **Nudity Detection** - NSFW, sexual content, bikini
- 😡 **Hate Speech Detection** - Profanity, threats, anger
- 🎬 **Video Analysis** - Frame-by-frame detection
- 📝 **Text Moderation** - Real-time content filtering

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Check server status |
| `/moderate` | POST | Analyze content |

## Frontend

Access the web interface at: `/frontend/index.html`

## Test the API

```bash
# Health check
curl https://Ajinkyakakade02-content-moderation-openenv.hf.space/health

# Moderate text
curl -X POST https://Ajinkyakakade02-content-moderation-openenv.hf.space/moderate \
  -H "Content-Type: application/json" \
  -d '{"content":{"type":"text","data":"I hate you"}}'