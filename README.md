# Multi-Agent Research Assistant

A collaborative multi-agent research system built with **LangGraph** and **Streamlit**. Nine specialized AI agents work together to research a topic, analyze findings, and produce a structured report.

## Architecture

The system uses a **supervisor‑worker** pattern — a Supervisor agent orchestrates the workflow, dispatching tasks to specialized agents and routing their outputs through a defined pipeline.

```
Supervisor
   |
   +---> Researcher (general search)
   +---> Researcher (hardware/compute persona)
   +---> Researcher (ethics/sociology persona)
   +---> Browser (Google Scholar)
   +---> Classifier (source scoring)
   +---> NER (entity extraction)
   +---> Analyzer (themes & outline)
   +---> Illustrator (diagrams via Stable Diffusion)
   +---> Writer (draft & revision)
   +---> Critiquer (scoring + repetition detection)
   +---> Finalize (PDF export)
```

## Agents

| Agent | Role |
|-------|------|
| **Supervisor** | Routes work using rule-based logic; generates sub-questions |
| **Researcher** | Web search via DuckDuckGo with specialized hardware/ethics personas |
| **Browser** | Google Scholar automation via Playwright |
| **Classifier** | Scores sources (1–10), filters low-relevance results |
| **NER** | Extracts people, orgs, technologies, dates, locations + entity relationships |
| **Analyzer** | Identifies key themes and builds a structured report outline |
| **Illustrator** | Generates image prompts (and optionally real images via Stability AI) |
| **Writer** | Produces the draft; revises based on critic feedback |
| **Critiquer** | Scores 5 criteria (originality, accuracy, completeness, clarity, depth); detects repetitive phrasing and redundant examples; forces REVISE if 3+ repetition issues found |

## Features

- **ChromaDB** vector store for persisting and searching research findings
- **MCP tool abstraction** layer for search, scrape, classify, and entity extraction
- **PDF export** via WeasyPrint
- **Duplicate detection**: The Critic scans for repetitive phrasing and forces rewrites
- **Specialized research personas**: Hardware/compute and ethics/sociology agents produce distinct, non-overlapping contributions

## Quick Start

1. **Clone the repo**
   ```bash
   git clone https://github.com/MohammedBelalTaharwah/Multi-Agent.git
   cd Multi-Agent
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Set up environment variables**
   Create a `.env` file:
   ```env
   LLM_PROVIDER=groq
   GROQ_API_KEY=gsk_your_key_here
   GROQ_MODEL=llama-3.3-70b-versatile
   SEARCH_PROVIDER=duckduckgo
   ```

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Configuration

All configuration is via `.env`:

| Variable | Default | Options |
|----------|---------|---------|
| `LLM_PROVIDER` | `groq` | `groq`, `together`, `ollama` |
| `GROQ_API_KEY` | — | Get free at [console.groq.com](https://console.groq.com) |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Any Groq model |
| `SEARCH_PROVIDER` | `duckduckgo` | `duckduckgo`, `tavily` |
| `TAVILY_API_KEY` | — | Required if using Tavily |
| `STABILITY_API_KEY` | — | Optional — enables image generation |

## Deployment

Deploy on **Streamlit Community Cloud**:

1. Push this repo to GitHub
2. Log in to [share.streamlit.io](https://share.streamlit.io)
3. Select the repo and set `app.py` as the entry point
4. Add your secrets (API keys) in the Streamlit dashboard

## Project Structure

```
├── app.py                 # Streamlit UI
├── agents.py              # Agent function factories + LLM/search init
├── graph.py               # LangGraph state schema + workflow definition
├── prompts.py             # Prompt templates for all 9 agents
├── vector_store.py        # ChromaDB persistence
├── scraper.py             # Web scraping (BeautifulSoup + Trafilatura)
├── mcp_tools.py           # MCP tool handler definitions
├── md_to_pdf.py           # Markdown-to-PDF conversion (WeasyPrint)
├── requirements.txt       # Python dependencies
└── .env                   # API keys (not committed)
```

## License

MIT
