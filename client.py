"""Simple MCP client that connects to a local MCP server and provides an interactive chat."""

import asyncio
import os
import json

from dotenv import load_dotenv
import anthropic
from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


load_dotenv()

MCP_SERVER_URL = "http://localhost:8000/mcp"


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


async def chat_loop(session: ClientSession, client: anthropic.Anthropic):
    """Run the interactive chat loop."""
    # Discover tools from the MCP server
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

            # Collect assistant response
            assistant_content = response.content
            messages.append({"role": "assistant", "content": assistant_content})

            # If no tool calls, print text and break
            if response.stop_reason == "end_turn":
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nClaude: {block.text}\n")
                break

            # Handle tool calls
            if response.stop_reason == "tool_use":
                tool_results = []
                for block in assistant_content:
                    if block.type == "tool_use":
                        print(f"  [Calling tool: {block.name}({json.dumps(block.input)})]")
                        try:
                            result = await session.call_tool(block.name, arguments=block.input)
                            # Extract text content from the MCP result
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
                # Unexpected stop reason, print whatever text we have and break
                for block in assistant_content:
                    if hasattr(block, "text"):
                        print(f"\nClaude: {block.text}\n")
                break


async def main():
    api_key = os.getenv("CLAUDE_API_KEY")
    if not api_key:
        print("Error: CLAUDE_API_KEY not found in .env file.")
        return

    client = anthropic.Anthropic(api_key=api_key)

    # Test API connectivity
    print("Testing API key...")
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

    # Connect to MCP server and start chat
    print(f"\nConnecting to MCP server at {MCP_SERVER_URL}...")
    try:
        async with streamablehttp_client(MCP_SERVER_URL) as (read_stream, write_stream, _):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                await chat_loop(session, client)
    except Exception as e:
        print(f"Error connecting to MCP server: {e}")
        print("Make sure your MCP server is running at", MCP_SERVER_URL)


if __name__ == "__main__":
    asyncio.run(main())
