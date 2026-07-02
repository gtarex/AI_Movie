"""Shared configuration for DeepSeek Novel Writer.
API key is never saved to disk — it exists only in memory during runtime."""

import os
import sys
from openai import OpenAI

_client = None
_api_key = None
_model = "deepseek-v4-pro"

STORIES_ROOT = "Stories"
DEFAULT_STORY_NAME = "Story1"
STORY_DIR = os.path.join(STORIES_ROOT, DEFAULT_STORY_NAME)
CHAR_DIR = os.path.join(STORY_DIR, "char")
MAX_HISTORY_PAIRS = 20
MAX_PLANNING_TURNS = 20

# Output token caps by response type. Keep CLI and Web UI aligned.
MAX_TOKENS_DEFAULT = 5000
MAX_TOKENS_WORLD = 400
MAX_TOKENS_CHARACTER_PROFILE = 700
MAX_TOKENS_SCENE_SETUP = 900
MAX_TOKENS_MEMORY_COMPRESSION = 800
MAX_TOKENS_SETTLEMENT = 2500
MAX_TOKENS_SCENE_MEMORY = 2000
MAX_TOKENS_FACT_PACK = 2000
MAX_TOKENS_FACT_CHECK = 900
MAX_TOKENS_PLANNING_PROPOSAL = 1200
MAX_TOKENS_PLANNING_CHAT = 1000
MAX_TOKENS_STORY_EVENT = 3000
MAX_TOKENS_SCENE_SYNOPSIS = 15000

# User-adjustable reply target (Chinese characters). The system computes
# max_tokens = ceiling(chars * TOKEN_MULTIPLIER_PER_CHAR) so the AI has
# enough headroom for the requested reply length.
DEFAULT_REPLY_CHARS = 300
TOKEN_MULTIPLIER_PER_CHAR = 3  # conservative ~3 tokens per Chinese character
DIRECTOR_TOKEN_MULTIPLIER = 1.5  # director mode gets 50 % extra token budget


def compute_max_tokens(chars_count, is_director=False):
    """Calculate max_tokens from target reply character count.
    
    Adds a floor of 500 tokens and applies director bonus when appropriate.
    """
    base = max(int(chars_count * TOKEN_MULTIPLIER_PER_CHAR), 500)
    if is_director:
        base = int(base * DIRECTOR_TOKEN_MULTIPLIER)
    return base


def get_client(api_key=None):
    """Get or create the OpenAI client for DeepSeek.

    Priority:
    1. Explicit api_key argument
    2. DEEPSEEK_API_KEY environment variable
    3. Interactive prompt (CLI only)

    The key is held in memory and never written to disk.
    """
    global _client, _api_key

    if api_key and api_key.strip():
        _api_key = api_key.strip()
        _client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com")
        return _client

    if _client is not None:
        return _client

    # Try environment variable
    _api_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()

    if not _api_key:
        # Interactive prompt
        print("\n" + "=" * 50)
        print("  DeepSeek API Key Required")
        print("=" * 50)
        print("  Enter your DeepSeek API key below.")
        print("  (Key is held in memory only — never saved to disk.)")
        print("-" * 50)
        try:
            _api_key = input("  API Key: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[Error] No API key provided. Exiting.")
            sys.exit(1)

    if not _api_key:
        print("[Error] No API key provided. Exiting.")
        sys.exit(1)

    _client = OpenAI(api_key=_api_key, base_url="https://api.deepseek.com")
    return _client


def get_model():
    """Return the configured model name."""
    return _model


def set_api_key(key):
    """Explicitly set the API key and recreate client.  Key is never saved to disk."""
    global _client, _api_key
    _api_key = key
    if key:
        _client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
    else:
        _client = None
