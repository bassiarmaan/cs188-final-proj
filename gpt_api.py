"""Test fetch to ChatGPT API. User prompt from terminal + system prompt from code. API key from .env."""

import argparse
import json
import os
import re
import sys
from pathlib import Path
import urllib.request


def _load_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        return key.strip()
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if line.startswith("OPENAI_API_KEY=") and not line.startswith("#"):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise ValueError("OPENAI_API_KEY not found. Set it in .env or the environment.")


api_key = _load_api_key()

# System prompt for bitmap generation (5 rows x 7 cols, max 12 cubes)
BITMAP_SYSTEM_PROMPT = """You convert design descriptions into a 5x7 bitmap for a robot that arranges cubes. Output ONLY the bitmap: 5 lines, 7 characters per line. Use 1 for cube positions, 0 for empty. Use at most 12 ones (12 cubes). No other text, explanations, or markdown.

Examples:
- Smiley: 0100010, 0001000, 0000000, 1110111, 0000000
- Horizontal line: 1111110, 0000000, 0000000, 0000000, 0000000

Prioritize simple geometric shapes. Make the outline of the shape only."""

# General-purpose system prompt (for fetch())
SYSTEM_PROMPT = """You are a helpful assistant. Be concise."""


def fetch(user_prompt: str) -> str:
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    body = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 100,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    return result["choices"][0]["message"]["content"]


def _parse_bitmap_response(text: str, nrows: int = 5, ncols: int = 7) -> tuple[str, ...]:
    """Extract nrows lines of ncols 0/1 chars from GPT response. Tolerates extra text."""
    rows: list[str] = []
    for line in text.splitlines():
        # Strip whitespace, quotes, commas; keep only 0 and 1
        cleaned = re.sub(r"[^01]", "", line.strip().strip('"').strip(","))
        if len(cleaned) == ncols and all(c in "01" for c in cleaned):
            rows.append(cleaned)
            if len(rows) >= nrows:
                break
    if len(rows) != nrows:
        raise ValueError(
            f"Expected {nrows} rows of {ncols} chars (0 or 1). Got {len(rows)} valid rows. "
            f"Raw response:\n{text[:500]}"
        )
    return tuple(rows[:nrows])


def fetch_bitmap(user_prompt: str, nrows: int = 5, ncols: int = 7) -> tuple[str, ...]:
    """Call GPT with bitmap system prompt, parse response into 5x7 bitmap tuple."""
    url = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    messages = [
        {"role": "system", "content": BITMAP_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    body = {
        "model": "gpt-4o-mini",
        "messages": messages,
        "max_tokens": 150,
    }
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers=headers, method="POST")
    with urllib.request.urlopen(req) as resp:
        result = json.loads(resp.read().decode())
    raw = result["choices"][0]["message"]["content"]
    return _parse_bitmap_response(raw, nrows=nrows, ncols=ncols)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Send prompt to ChatGPT (system prompt + user prompt)")
    ap.add_argument("prompt", nargs="?", help="User prompt (or omit to be prompted interactively)")
    args = ap.parse_args()
    user_prompt = (args.prompt or input("Enter your prompt: ")).strip()
    if not user_prompt:
        print("No prompt entered. Exiting.")
        sys.exit(1)
    print("Sending query...")
    try:
        response = fetch(user_prompt)
        print("Response:", response)
    except Exception as e:
        print("Error:", e)
