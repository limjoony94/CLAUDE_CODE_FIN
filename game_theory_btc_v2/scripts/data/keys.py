"""Shared API key access — Game Theory BTC v2.

Reads `api_keys.yaml` from the household-shared location
(`bingx_rl_trading_bot/config/api_keys.yaml`) so this project + the legacy
bot share a single source of truth for credentials.

Override path via env var ``API_KEYS_PATH``.

NEVER print or log full keys. Use ``mask()`` for diagnostics.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

# Resolve paths
_THIS = Path(__file__).resolve()
PROJECT_ROOT = _THIS.parents[2]  # game_theory_btc_v2/
SHARED_KEYS_PATH = PROJECT_ROOT.parent / "bingx_rl_trading_bot" / "config" / "api_keys.yaml"


def _resolve_path() -> Path:
    """Resolve api_keys.yaml location. Honors API_KEYS_PATH env override."""
    env = os.environ.get("API_KEYS_PATH")
    if env:
        p = Path(env).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"API_KEYS_PATH set but file missing: {p}")
        return p
    if SHARED_KEYS_PATH.exists():
        return SHARED_KEYS_PATH
    raise FileNotFoundError(
        f"Shared keys file not found at {SHARED_KEYS_PATH}. "
        "Set API_KEYS_PATH env var or copy api_keys.yaml to the expected location."
    )


@lru_cache(maxsize=1)
def load_keys() -> dict[str, Any]:
    """Load and cache the YAML once per process."""
    path = _resolve_path()
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"api_keys.yaml root must be a mapping; got {type(data).__name__}")
    return data


def get_bingx(use_mainnet: bool = True) -> dict[str, str]:
    """Return BingX credentials.

    Args:
        use_mainnet: True (default) returns mainnet, False returns testnet.

    Returns:
        {'api_key': str, 'secret_key': str}

    Raises:
        KeyError if BingX section missing or empty.
    """
    keys = load_keys()
    bingx = keys.get("bingx") or {}
    if not bingx:
        raise KeyError("bingx section missing in api_keys.yaml")
    section_name = "mainnet" if use_mainnet else "testnet"
    section = bingx.get(section_name) or {}
    api_key = section.get("api_key", "").strip()
    secret_key = section.get("secret_key", "").strip()
    if not api_key or not secret_key:
        raise KeyError(f"bingx.{section_name} api_key/secret_key empty")
    return {"api_key": api_key, "secret_key": secret_key}


def get_coinglass() -> str:
    """Return Coinglass API key.

    Returns:
        API key string.

    Raises:
        KeyError if not configured.
    """
    keys = load_keys()
    cg = keys.get("coinglass") or {}
    if isinstance(cg, str):
        api_key = cg
    else:
        api_key = cg.get("api_key", "")
    api_key = (api_key or "").strip()
    if not api_key:
        raise KeyError(
            "coinglass.api_key missing in api_keys.yaml. "
            "Register free tier at https://www.coinglass.com/api"
        )
    return api_key


def get_telegram() -> dict[str, str]:
    """Return Telegram bot token + chat_id (may be empty strings)."""
    keys = load_keys()
    tg = keys.get("telegram") or {}
    return {
        "bot_token": (tg.get("bot_token") or "").strip(),
        "chat_id": str(tg.get("chat_id") or "").strip(),
    }


def get_discord() -> str:
    """Return Discord webhook URL (may be empty)."""
    keys = load_keys()
    dc = keys.get("discord") or {}
    return (dc.get("webhook_url") or "").strip()


def mask(secret: str, head: int = 6, tail: int = 4) -> str:
    """Safely mask a secret for diagnostic logging."""
    if not secret:
        return "<empty>"
    if len(secret) <= head + tail:
        return "*" * len(secret)
    return f"{secret[:head]}...{secret[-tail:]} (len={len(secret)})"


def diagnose() -> dict[str, Any]:
    """Return masked status of all configured keys for sanity check."""
    out = {"path": str(_resolve_path()), "available": {}}
    try:
        b = get_bingx(use_mainnet=True)
        out["available"]["bingx_mainnet"] = {
            "api_key": mask(b["api_key"]),
            "secret_key": mask(b["secret_key"]),
        }
    except KeyError as e:
        out["available"]["bingx_mainnet"] = f"missing: {e}"
    try:
        c = get_coinglass()
        out["available"]["coinglass"] = mask(c)
    except KeyError as e:
        out["available"]["coinglass"] = f"missing: {e}"
    return out


if __name__ == "__main__":
    import json
    print(json.dumps(diagnose(), indent=2))
