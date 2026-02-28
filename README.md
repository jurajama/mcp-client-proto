Prototyping MCP client implementation

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install anthropic mcp python-dotenv openai
```

## Configuration

Create a `.env` file in the project root. The client supports two model providers.

---

### Option 1: Anthropic Claude (default)

```env
MODEL_PROVIDER=anthropic
CLAUDE_API_KEY=your_api_key_here
```

Get an API key at https://console.anthropic.com. The client uses `claude-opus-4-6` by default.

---

### Option 2: Local Ollama (llama, mistral, etc.)

[Install Ollama](https://ollama.com/download), then pull a model and start the server:

```bash
ollama pull llama3.2
ollama serve
```

```env
MODEL_PROVIDER=ollama
OLLAMA_MODEL=llama3.2
# OLLAMA_BASE_URL=http://localhost:11434/v1  # optional, this is the default
```

For tool/function calling to work, use a model that supports it (e.g. `llama3.1`, `llama3.2`, `mistral-nemo`).

---

## Running

Make sure your MCP server is running at `http://localhost:8000/mcp`, then:

```bash
python client.py
```
