"""Simple MCP client that connects to a local MCP server and provides an interactive chat."""
import asyncio
import os
import json

from dotenv import load_dotenv
import anthropic
from openai import OpenAI

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client  # adjust if you moved to mcp 2.x's streamable_http_client

load_dotenv()

MCP_SERVER_URL = os.getenv("MCP_SERVER_URL", "http://localhost:8000/mcp")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.2")

LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")  # e.g. https://litellm.internal.example.com/v1
LITELLM_API_KEY = os.getenv("LITELLM_API_KEY")    # gateway token, sent as Bearer auth
LITELLM_MODEL = os.getenv("LITELLM_MODEL", "gpt-4o")  # the model alias configured on your gateway


def mcp_tools_to_anthropic(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to Anthropic API format."""
    return [
        {
            "name": tool.name,
            "description": tool.description or "",
            "input_schema": tool.inputSchema,
        }
        for tool in mcp_tools
    ]


def mcp_tools_to_openai(mcp_tools: list) -> list[dict]:
    """Convert MCP tool definitions to OpenAI-compatible API format (used by Ollama and LiteLLM)."""
    return [
        {
            "type": "function",
            "function": {
                "name": tool.name,
                "description": tool.description or "",
                "parameters": tool.inputSchema,
            },
        }
        for tool in mcp_tools
    ]


async def chat_loop(session: ClientSession, client: anthropic.Anthropic):
    """Run the interactive chat loop using Anthropic Claude directly."""
    tools_result = await session.list_tools()
    tools = mcp_tools_to_anthropic(tools_result.tools)

    if tools:
        print(f"\nConnected to MCP server. Available tools: {[t['name'] for t in tools]}")
    else:
        print("\nConnected to MCP server. No tools available.")

    messages: list[dict] = []
    print("\nChat started. Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going until Claude stops calling tools
        while True:
            response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=4096,
                tools=tools if tools else anthropic.NOT_GIVEN,
                messages=messages,
            )

            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nClaude: {block.text}\n")
                break

            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [Calling tool: {block.name}({json.dumps(block.input)})]")
                        try:
                            result = await session.call_tool(block.name, arguments=block.input)
                            result_text = ""
                            if result.content:
                                for item in result.content:
                                    if hasattr(item, "text"):
                                        result_text += item.text
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": result_text,
                            })
                        except Exception as e:
                            tool_results.append({
                                "type": "tool_result",
                                "tool_use_id": block.id,
                                "content": f"Error: {e}",
                                "is_error": True,
                            })
                messages.append({"role": "user", "content": tool_results})
            else:
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nClaude: {block.text}\n")
                break


async def chat_loop_openai_compat(session: ClientSession, client: OpenAI, model: str, label: str):
    """Run the interactive chat loop against any OpenAI-compatible backend (Ollama, LiteLLM gateway, ...)."""
    tools_result = await session.list_tools()
    tools = mcp_tools_to_openai(tools_result.tools)

    if tools:
        print(f"\nConnected to MCP server. Available tools: {[t['function']['name'] for t in tools]}")
    else:
        print("\nConnected to MCP server. No tools available.")

    messages: list[dict] = []
    print(f"\nChat started ({label} / {model}). Type 'quit' to exit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            print("Goodbye!")
            break

        messages.append({"role": "user", "content": user_input})

        # Agentic loop: keep going until the model stops calling tools
        while True:
            kwargs = {"model": model, "messages": messages}
            if tools:
                kwargs["tools"] = tools

            response = client.chat.completions.create(**kwargs)
            msg = response.choices[0].message
            finish_reason = response.choices[0].finish_reason

            messages.append({
                "role": "assistant",
                "content": msg.content,
                "tool_calls": msg.tool_calls,
            })

            if finish_reason == "stop" or not msg.tool_calls:
                print(f"\n{label}: {msg.content}\n")
                break

            if finish_reason == "tool_calls":
                for tc in msg.tool_calls:
                    tool_name = tc.function.name
                    tool_args = json.loads(tc.function.arguments)
                    print(f"  [Calling tool: {tool_name}({json.dumps(tool_args)})]")
                    try:
                        result = await session.call_tool(tool_name, arguments=tool_args)
                        result_text = ""
                        if result.content:
                            for item in result.content:
                                if hasattr(item, "text"):
                                    result_text += item.text
                    except Exception as e:
                        result_text = f"Error: {e}"

                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result_text,
                    })
            else:
                if msg.content:
                    print(f"\n{label}: {msg.content}\n")
                break


async def main():
    provider = os.getenv("MODEL_PROVIDER", "anthropic").lower()

    if provider == "anthropic":
        api_key = os.getenv("CLAUDE_API_KEY")
        if not api_key:
            print("Error: CLAUDE_API_KEY not found in .env file.")
            return

        client = anthropic.Anthropic(api_key=api_key)
        print("Testing Anthropic API key...")
        try:
            test_response = client.messages.create(
                model="claude-opus-4-6",
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            )
            print(f"API key works! Response: {test_response.content[0].text}")
        except anthropic.AuthenticationError:
            print("Error: Invalid API key. Check your CLAUDE_API_KEY in .env.")
            return
        except anthropic.PermissionDeniedError:
            print("Error: API key lacks permissions. You may need to purchase API credits.")
            return
        except anthropic.APIStatusError as e:
            print(f"API error ({e.status_code}): {e.message}")
            if e.status_code == 400 and "credit" in str(e.message).lower():
                print("You likely need to purchase API credits at console.anthropic.com.")
            return

        loop_fn = lambda session: chat_loop(session, client)

    elif provider == "ollama":
        client = OpenAI(base_url=OLLAMA_BASE_URL, api_key="ollama")
        print(f"Testing Ollama at {OLLAMA_BASE_URL} with model '{OLLAMA_MODEL}'...")
        try:
            test_response = client.chat.completions.create(
                model=OLLAMA_MODEL,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            )
            print(f"Ollama works! Response: {test_response.choices[0].message.content}")
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            print(f"Make sure Ollama is running and '{OLLAMA_MODEL}' is pulled.")
            print(f"  ollama pull {OLLAMA_MODEL}")
            print(f"  ollama serve")
            return

        loop_fn = lambda session: chat_loop_openai_compat(session, client, OLLAMA_MODEL, "Ollama")

    elif provider == "litellm":
        if not LITELLM_BASE_URL:
            print("Error: LITELLM_BASE_URL not found in .env file.")
            return
        if not LITELLM_API_KEY:
            print("Error: LITELLM_API_KEY not found in .env file.")
            return

        client = OpenAI(base_url=LITELLM_BASE_URL, api_key=LITELLM_API_KEY)
        print(f"Testing LiteLLM gateway at {LITELLM_BASE_URL} with model '{LITELLM_MODEL}'...")
        try:
            test_response = client.chat.completions.create(
                model=LITELLM_MODEL,
                max_tokens=50,
                messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            )
            print(f"Gateway works! Response: {test_response.choices[0].message.content}")
        except Exception as e:
            # Surfaces 401s from a bad/expired token as well as other gateway errors
            print(f"Error connecting to LiteLLM gateway: {e}")
            print("Check LITELLM_BASE_URL, LITELLM_API_KEY, and that LITELLM_MODEL is a valid model alias on the gateway.")
            return

        loop_fn = lambda session: chat_loop_openai_compat(session, client, LITELLM_MODEL, "LiteLLM")

    else:
        print(f"Error: Unknown MODEL_PROVIDER '{provider}'. Use 'anthropic', 'ollama', or 'litellm'.")
        return

    # Connect to MCP server and start chat (shared across all providers)
    print(f"\nConnecting to MCP server at {MCP_SERVER_URL}...")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await loop_fn(session)
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        print("Make sure your MCP server is running at", MCP_SERVER_URL)


if __name__ == "__main__":
    asyncio.run(main())
