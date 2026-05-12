
# 🤖 AI Content Generator

> A multimodal AI-powered content generation platform built with LangGraph, Streamlit, and multi-agent architecture.

---

## 📌 Table of Contents

- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Installation & Setup](#installation--setup)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Project Structure](#project-structure)
- [Phase 1 vs Phase 2](#phase-1-vs-phase-2)
- [🚀 Future Work — Phase 2 (Upcoming)](#-future-work--phase-2-upcoming)
- [Known Limitations](#known-limitations)
- [Contributors](#contributors)

---

## Project Overview

The AI Content Generator is a Proof-of-Concept (PoC) multimodal AI system designed to transform a single input idea into multiple content formats automatically.

**Example:**

| Input | Outputs |
|-------|---------|
| `"AI in Healthcare"` | ✅ Blog post &nbsp; ✅ LinkedIn summary &nbsp; ✅ Generated image |

Built for marketers, content creators, startups, and media organizations who need to scale content production without losing quality or consistency.

---

## System Architecture

```
User Input
    │
    ▼
┌─────────────┐
│  Supervisor  │  ← Main orchestrator
└──────┬──────┘
       │
       ▼
┌─────────────┐
│    Router    │  ← Routes to appropriate sub-agent
└──────┬──────┘
       │
  ┌────┴────┐
  │         │
  ▼         ▼
┌──────┐  ┌──────┐
│ Text │  │Image │  ← Sub-agents
│Agent │  │Agent │
└──────┘  └──────┘
  │
  ├── Generator     → Calls LLM (Groq / OpenAI)
  ├── Optimizer     → Spell check, readability scoring
  └── Evaluator     → Quality scoring, review pipeline
```

---

## Features

- ✅ **Text Generation** — AI-powered blog posts, summaries, and social content
- ✅ **Image Generation** — AI-generated visuals from text prompts
- ✅ **Multi-Agent Architecture** — Supervisor → Router → Sub-agents using LangGraph
- ✅ **Content Optimization** — Spell checking, readability analysis, tone control
- ✅ **Quality Evaluation** — Automated scoring and review pipeline
- ✅ **Export Support** — Publish directly to Dev.to
- ✅ **Multi-Provider Support** — Switch between OpenAI and Groq for text generation
- ✅ **Streamlit UI** — Clean, interactive web interface

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | Streamlit 1.30 |
| Agent Framework | LangGraph, LangChain |
| Text LLM | Groq (`llama-3.3-70b-versatile`) / OpenAI GPT-4o |
| Image Generation | Hugging Face (`FLUX.1-schnell`) |
| Spell Checking | pyspellchecker |
| Readability | textstat |
| Sentiment | vaderSentiment |
| Export | Dev.to API |

---

## Installation & Setup

### Prerequisites

- Python 3.11+
- pip

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git
cd YOUR_REPO_NAME
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure environment variables

Copy the example env file and fill in your keys:

```bash
cp .env.example .env
```

Then edit `.env` with your actual API keys (see [Configuration](#configuration) below).

### 4. Run the app

```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501`

---

## Configuration

Create a `.env` file in the project root with the following variables:

```dotenv
# ── TEXT PROVIDER ──────────────────────────────────────────
# Options: "openai" or "groq" (groq is free)
TEXT_PROVIDER=groq

# ── OPENAI (if TEXT_PROVIDER=openai) ───────────────────────
OPENAI_API_KEY=your_openai_key_here
OPENAI_MODEL=gpt-4o

# ── GROQ (if TEXT_PROVIDER=groq — FREE) ────────────────────
# Get free key at: https://console.groq.com
GROQ_API_KEY=your_groq_key_here
GROQ_MODEL=llama-3.3-70b-versatile

# ── IMAGE GENERATION ───────────────────────────────────────
# Get free key at: https://huggingface.co/settings/tokens
# Required permission: "Make calls to Inference Providers"
HUGGINGFACE_API_KEY=your_hf_token_here

# ── EXPORT (optional) ──────────────────────────────────────
DEVTO_API_KEY=your_devto_key_here
DEVTO_DEFAULT_PUBLISHED=false

# ── EXPORT CONTROLS ────────────────────────────────────────
EXPORT_ALLOW_UNAPPROVED=0
```

### Getting Free API Keys

| Service | URL | Notes |
|---------|-----|-------|
| Groq (text) | https://console.groq.com | Free, no credit card |
| Hugging Face (images) | https://huggingface.co/settings/tokens | Free, enable "Inference Providers" permission |
| Dev.to (export) | https://dev.to/settings/extensions | Optional |

---

## Running the App

> ⚠️ Always `cd` into the project folder first, and use `streamlit run` — not `python app.py`.

```bash
cd path/to/AI_Content_Generator
streamlit run app.py
```

The app will be available at:
- **Local:** http://localhost:8501
- **Network:** http://YOUR_IP:8501

---

## Project Structure

```
AI_Content_Generator/
│
├── app.py                          # Main Streamlit application
├── .env                            # Environment variables (not committed)
├── .env.example                    # Template for environment variables
├── requirements.txt                # Python dependencies
│
├── src/
│   ├── main_agent/
│   │   ├── supervisor.py           # Top-level orchestrator
│   │   └── router.py               # Routes tasks to sub-agents
│   │
│   └── sub_agents/
│       ├── text_generator/
│       │   ├── text_agent.py
│       │   └── modules/
│       │       ├── generator/
│       │       │   └── content_generator.py   # LLM calls (Groq/OpenAI)
│       │       ├── optimizer/
│       │       │   └── optimizer.py           # Spell check, readability
│       │       └── evaluator/
│       │
│       └── image_generator/
│           ├── image_agent.py
│           └── modules/
│               └── generator/
│                   └── image_generator.py     # HuggingFace FLUX
│
├── exports/                        # Generated content exports
├── runs/                           # Run logs (auto-created)
└── models/                         # Local model files
```

---

## Phase 1 vs Phase 2

| Capability | Phase 1 (Current) | Phase 2 (Planned) |
|-----------|:-----------------:|:-----------------:|
| Text Generation | ✅ | ✅ |
| Image Generation | ✅ | ✅ |
| Audio Generation | ❌ | 🔜 |
| Video Generation | ❌ | 🔜 |
| Multimodal Pipeline | ❌ | 🔜 |
| Cross-Modal Consistency | ❌ | 🔜 |

---

## 🚀 Future Work — Phase 2 (Upcoming)

> ### ⚠️ This section outlines the planned extensions to the system. Phase 2 is currently under active development.

Phase 2 extends the system from a text+image generator into a **fully multimodal AI content platform**. The goal is a single pipeline where one input idea produces content across all major formats automatically.

---

### 🎯 Target Pipeline

```
User Input ──► Text ──► Image ──► Audio ──► Video
```

---

### 🎙️ Audio Generation Module

Convert generated text into natural-sounding spoken narration.

**Planned Capabilities:**
- 🔊 Natural-sounding voice narration from generated text
- 🎭 Tone-aware speech (formal, conversational, storytelling)
- 🌍 Multi-language support
- 🎙️ Use cases: podcasts, voice assistants, social media audio

**Planned Tools:**
- `gTTS` (Google Text-to-Speech) — free, no API key needed
- ElevenLabs API — for higher quality voices (free tier available)

---

### 🎬 Video Generation Module *(Mandatory)*

Transform the full content pipeline into a short explainer video.

**Minimum Requirement (PoC):**
- Combine generated **text** + **image** + **audio** into a video
- Output: 30–60 second short explainer video

**Advanced (Optional):**
- 🎞️ AI-generated video scenes
- ✨ Dynamic transitions between slides
- 📱 Social media-ready clip formats (9:16, 1:1, 16:9)

**Planned Tools:**
- `moviepy` — free video assembly library
- `ffmpeg` — audio/video processing

---

### 🔗 Multimodal Pipeline Integration *(Core Challenge)*

Design a fully integrated pipeline with intelligent agent coordination:

| Agent | Role |
|-------|------|
| **Content Planner Agent** | Decides what content to generate based on input |
| **Modality Router Agent** | Selects appropriate output formats |
| **Audio Agent** | Handles speech generation |
| **Video Agent** | Handles video creation and assembly |
| **Consistency Agent** *(optional)* | Ensures tone/style coherence across all outputs |

---

### 🎨 Cross-Modal Consistency

All outputs must align in tone, style, and message:

| Tone | Text Style | Voice | Visuals |
|------|-----------|-------|---------|
| Professional | Formal, structured | Neutral, clear | Clean, minimal |
| Storytelling | Engaging, narrative | Expressive, warm | Dynamic, rich |
| Conversational | Casual, direct | Friendly, upbeat | Bright, relatable |

---

### 📊 Phase 2 Evaluation Criteria

| Modality | Metrics |
|---------|---------|
| Text | Readability, coherence, tone accuracy |
| Image | Visual relevance, style consistency |
| Audio | Clarity, naturalness, tone match |
| Video | Synchronization, narrative quality, visual flow |

---

### 📦 Phase 2 Deliverables

1. **Concept & Architecture Report** — Full system design documentation
2. **Functional Prototype** — Working PoC with all 4 modalities
3. **End-to-End Demonstration** — Live pipeline demo
4. **Critical Reflection** — Analysis of performance, limitations, and learnings

---

## Known Limitations

- Image generation via Hugging Face may take 15–60 seconds on first call (model cold start)
- Groq free tier has rate limits — heavy usage may hit limits
- Video generation is not yet implemented (Phase 2)
- Audio generation is not yet implemented (Phase 2)
- Dev.to export requires a valid API key

---

## Contributors

- **Phase 1 (Original Team):** Architecture design, text generation, image generation, evaluation pipeline
- **Phase 2 (Current Team):** Bug fixes, provider migration, audio & video generation *(in progress)*

---

*Built as part of the Collaborative Industry Project — Masters Programme*
=======
# AI-Content-Generator

