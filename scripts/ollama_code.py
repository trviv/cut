#!/usr/bin/env python3
"""Send a coding plan to a local Ollama model and get implementation back.

Usage:
    # From a plan file:
    python3 scripts/ollama_code.py plan.txt

    # From stdin:
    echo "implement a function that ..." | python3 scripts/ollama_code.py

    # With a specific model (default: devstral-small-2:24b):
    python3 scripts/ollama_code.py --model devstral-small-2:24b plan.txt

    # Include file context:
    python3 scripts/ollama_code.py --context src/foo.cpp --context src/bar.h plan.txt
"""

import argparse
import json
import sys
import urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "devstral-small-2:24b"


def read_file(path):
    with open(path, "r") as f:
        return f.read()


def call_ollama(prompt, model=DEFAULT_MODEL, stream=True):
    payload = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": stream,
        "options": {
            "temperature": 0.3,
            "num_predict": 16384,
        },
    }).encode("utf-8")

    req = urllib.request.Request(
        OLLAMA_URL,
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    with urllib.request.urlopen(req, timeout=300) as resp:
        for line in resp:
            if line.strip():
                chunk = json.loads(line)
                text = chunk.get("response", "")
                if text:
                    print(text, end="", flush=True)
                if chunk.get("done"):
                    break
    print()


def main():
    parser = argparse.ArgumentParser(description="Send a plan to Ollama for code implementation")
    parser.add_argument("plan_file", nargs="?", help="File containing the plan (reads stdin if omitted)")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model (default: {DEFAULT_MODEL})")
    parser.add_argument("--context", action="append", default=[], help="Source files to include as context (repeatable)")
    parser.add_argument("--no-stream", action="store_true", help="Wait for full response instead of streaming")
    args = parser.parse_args()

    # Read the plan
    if args.plan_file:
        plan = read_file(args.plan_file)
    elif not sys.stdin.isatty():
        plan = sys.stdin.read()
    else:
        print("Error: provide a plan file or pipe input via stdin", file=sys.stderr)
        sys.exit(1)

    # Build prompt with context files
    parts = []

    if args.context:
        parts.append("# Reference Files\n")
        for path in args.context:
            content = read_file(path)
            parts.append(f"## {path}\n```\n{content}\n```\n")

    parts.append("# Implementation Plan\n")
    parts.append(plan)
    parts.append("\n\n# Instructions\n")
    parts.append(
        "Implement the plan above. Output ONLY the code — no explanations, "
        "no markdown fences unless showing multiple files. "
        "For each file, start with a comment line: // FILE: path/to/file.ext\n"
        "Then the complete file content.\n"
        "Be precise and follow the plan exactly."
    )

    prompt = "\n".join(parts)

    call_ollama(prompt, model=args.model, stream=not args.no_stream)


if __name__ == "__main__":
    main()
