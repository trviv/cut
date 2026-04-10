#!/usr/bin/env python3
"""
GGUF Client — CLI tool to interact with the GGUF HTTP server.

Usage:
    python client.py [--host HOST] [--port PORT] <command> [args...]

Commands:
    health                                     Check server status
    generate "prompt" [--max-tokens N]         Generate text
    chat "message" [--file path]...            Chat with file context
    read <path>                                Read a file via server
    write <path> <content>                     Write a file via server
    reset                                      Reset KV cache
"""

import argparse
import json
import sys
import urllib.request
import urllib.error


def make_request(host, port, method, path, data=None):
    """Make an HTTP request and return parsed JSON response."""
    url = f"http://{host}:{port}{path}"
    body = json.dumps(data).encode() if data else None
    req = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"} if body else {},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as e:
        try:
            err = json.loads(e.read().decode())
            print(f"Error ({e.code}): {err.get('error', 'Unknown error')}", file=sys.stderr)
        except Exception:
            print(f"Error ({e.code}): {e.reason}", file=sys.stderr)
        sys.exit(1)
    except urllib.error.URLError as e:
        print(f"Connection failed: {e.reason}", file=sys.stderr)
        print(f"Is the server running at {host}:{port}?", file=sys.stderr)
        sys.exit(1)


def cmd_health(args):
    resp = make_request(args.host, args.port, "GET", "/v1/health")
    cfg = resp.get("model_config", {})
    print(f"Status: {resp['status']}")
    print(f"Model: {resp['model_path']}")
    print(
        f"Config: dim={cfg.get('dim')} layers={cfg.get('n_layers')} "
        f"heads={cfg.get('n_heads')} vocab={cfg.get('vocab_size')}"
    )
    print(f"GPU: {resp.get('gpu_memory_mb', 0):.1f} MB ({resp.get('buffer_count', 0)} buffers)")
    print(f"Chat: {'yes' if resp.get('chat_mode') else 'no'}")
    print(f"Default max tokens: {resp.get('default_max_tokens')}")


def cmd_generate(args):
    data = {"prompt": args.prompt}
    if args.max_tokens:
        data["max_tokens"] = args.max_tokens
    if args.repeat_penalty:
        data["repeat_penalty"] = args.repeat_penalty
    if args.no_chat:
        data["chat_mode"] = False

    resp = make_request(args.host, args.port, "POST", "/v1/generate", data)
    print(resp["text"])
    print(
        f"\n--- {resp['prompt_tokens']} prompt + {resp['generated_tokens']} generated | "
        f"{resp['elapsed_ms']:.0f}ms | {resp['tokens_per_second']:.1f} tok/s ---"
    )


def cmd_chat(args):
    data = {"message": args.message}
    if args.file:
        data["files"] = args.file
    if args.max_tokens:
        data["max_tokens"] = args.max_tokens
    if args.system:
        data["system_prompt"] = args.system

    resp = make_request(args.host, args.port, "POST", "/v1/chat", data)
    print(resp["text"])
    print(
        f"\n--- {resp['prompt_tokens']} prompt + {resp['generated_tokens']} generated | "
        f"{resp['elapsed_ms']:.0f}ms | {resp['tokens_per_second']:.1f} tok/s ---"
    )


def cmd_read(args):
    resp = make_request(args.host, args.port, "POST", "/v1/read", {"path": args.path})
    print(resp["content"])


def cmd_write(args):
    content = args.content
    if content == "-":
        content = sys.stdin.read()
    resp = make_request(args.host, args.port, "POST", "/v1/write", {"path": args.path, "content": content})
    print(f"Wrote {resp['bytes_written']} bytes to {resp['path']}")


def cmd_reset(args):
    resp = make_request(args.host, args.port, "POST", "/v1/reset", {})
    print(resp.get("message", "OK"))


def main():
    parser = argparse.ArgumentParser(description="GGUF Server Client")
    parser.add_argument("--host", default="localhost", help="Server host (default: localhost)")
    parser.add_argument("--port", type=int, default=8080, help="Server port (default: 8080)")
    sub = parser.add_subparsers(dest="command", required=True)

    # health
    sub.add_parser("health", help="Check server status")

    # generate
    p_gen = sub.add_parser("generate", help="Generate text from a prompt")
    p_gen.add_argument("prompt", help="Text prompt")
    p_gen.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    p_gen.add_argument("--repeat-penalty", type=float, help="Repetition penalty")
    p_gen.add_argument("--no-chat", action="store_true", help="Disable ChatML wrapping")

    # chat
    p_chat = sub.add_parser("chat", help="Chat with optional file context")
    p_chat.add_argument("message", help="User message")
    p_chat.add_argument("--file", action="append", help="File path to include as context (repeatable)")
    p_chat.add_argument("--max-tokens", type=int, help="Max tokens to generate")
    p_chat.add_argument("--system", help="System prompt")

    # read
    p_read = sub.add_parser("read", help="Read a file via the server")
    p_read.add_argument("path", help="File path to read")

    # write
    p_write = sub.add_parser("write", help="Write a file via the server")
    p_write.add_argument("path", help="File path to write")
    p_write.add_argument("content", help="Content to write (use '-' for stdin)")

    # reset
    sub.add_parser("reset", help="Reset model KV cache")

    args = parser.parse_args()

    commands = {
        "health": cmd_health,
        "generate": cmd_generate,
        "chat": cmd_chat,
        "read": cmd_read,
        "write": cmd_write,
        "reset": cmd_reset,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
