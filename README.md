Prototyping MCP client implementation

## Setup

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
```

Note that Python 3.10 or later is required as `mcp` package does not work with Python 3.9 or older versions.

## Configuration

Create a `.env` file in the project root. The client supports three model providers.

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

### Option 3: LiteLLM gateway

Point the client at a LiteLLM proxy/gateway. The gateway's token is sent as a standard `Authorization: Bearer` header via the OpenAI SDK.

```env
MODEL_PROVIDER=litellm
LITELLM_BASE_URL=https://your-litellm-gateway/v1
LITELLM_API_KEY=your_gateway_token
LITELLM_MODEL=your-model-alias   # the model alias configured on your gateway
```

If your gateway expects the token under a different header instead of `Authorization: Bearer`, you'll need to adjust the `OpenAI(...)` client construction in `client.py` accordingly (e.g. via `default_headers`).

---

## Running

Make sure your MCP server is running at `http://localhost:8000/mcp`, then:

```bash
python client.py
```
