# EmoMem — 记忆增强 AI 情感陪伴 Agent

A memory-augmented emotional companion agent with real-time emotion tracking, adaptive retrieval, and multi-LLM backend support.

![EmoMem Interface](docs/screenshot.png)

## Features

- **Emotional Radar** — 8-axis Plutchik emotion model with real-time valence tracking
- **Episodic Memory** — stores and retrieves past conversations for context-aware responses
- **Affective Memory Layer** — adaptive emotional baseline that evolves over sessions
- **Mood-Adaptive Retrieval (MAR)** — adjusts memory retrieval weights based on current emotional state
- **Thompson Sampling Strategy** — context-stratified strategy selection (§4.5)
- **Crisis Detection** — urgency scoring with fast-path intervention at threshold 0.9
- **Multi-LLM Support** — Volcengine ARK (Doubao Seed), Anthropic Claude, OpenAI, or Mock fallback
- **Ablation Study Panel** — toggle core modules on/off for research comparison

## Project Structure

```
emomem_v1/
├── src/
│   ├── main.py          # EmoMemAgent orchestrator
│   ├── perception.py    # Emotion & intent perception
│   ├── memory.py        # Working / episodic / affective memory
│   ├── planning.py      # Strategy planning & recovery trajectory
│   ├── action.py        # Response generation
│   ├── adaptation.py    # Feedback & adaptive learning
│   ├── models.py        # Core data models (StateVector, etc.)
│   ├── llm_provider.py  # Multi-backend LLM abstraction
│   └── config.py        # System parameters
├── web/
│   ├── index.html       # Chat UI
│   └── server.py        # Flask REST API server
└── tests/               # Unit, integration, crisis, emotion tests
```

## Quick Start

### 1. Install dependencies

```bash
pip install flask flask-cors anthropic openai
```

### 2. Configure your API token

In `src/llm_provider.py`, replace `"please enter your token"` with your Volcengine ARK API key, or set an environment variable:

```bash
export OPENAI_API_KEY=your_key_here
export OPENAI_BASE_URL=https://ark.cn-beijing.volces.com/api/v3/
```

### 3. Run the web interface

```bash
cd emomem_v1
python -m web.server
# Open http://127.0.0.1:8080
```

### 4. Run tests

```bash
pytest tests/
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `OPENAI_API_KEY` | Volcengine ARK or OpenAI API key | — |
| `ANTHROPIC_API_KEY` | Anthropic Claude API key | — |
| `OPENAI_BASE_URL` | API base URL for OpenAI-compatible endpoints | `https://ark.cn-beijing.volces.com/api/v3/` |
| `EMOMEM_LLM_PROVIDER` | `anthropic` / `openai` / `mock` | auto-detect |
| `EMOMEM_LLM_MODEL` | Model name override | `doubao-seed-2-0-lite-260215` |
| `EMOMEM_REASONING_EFFORT` | Reasoning depth: `off` / `low` / `medium` / `high` | `off` |

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/api/chat` | Send message, get response + state |
| POST | `/api/reset` | Reset session |
| GET | `/api/state` | Get current agent state |
| GET/POST | `/api/config` | Get/update LLM configuration |
| GET/POST | `/api/reasoning` | Get/set reasoning effort level |
| POST | `/api/demo` | Run preset demo scenario |
| GET/POST | `/api/ablation` | Ablation study controls |

## License

MIT
