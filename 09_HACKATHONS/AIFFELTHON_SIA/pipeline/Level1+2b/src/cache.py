"""
SQLite-based cache for scraped articles and LLM responses.

scrape_cache: url -> (status, text, fetched_at)
  - status: 'ok' | 'unreachable' | 'no_text'
  - text is NULL for unreachable/no_text

llm_cache: key -> (response_json, created_at)
  - key = sha1(prompt_version | target_city | target_date | sorted article texts)
"""

import hashlib
import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from src.config import PROJECT_ROOT

CACHE_PATH = PROJECT_ROOT / "logs" / "cache.sqlite"
_local = threading.local()


def _conn() -> sqlite3.Connection:
    if getattr(_local, "conn", None) is None:
        CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        c = sqlite3.connect(str(CACHE_PATH), timeout=30.0, check_same_thread=False)
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        _local.conn = c
        _init_schema(c)
    return _local.conn


def _init_schema(c: sqlite3.Connection) -> None:
    c.executescript("""
        CREATE TABLE IF NOT EXISTS scrape_cache (
            url        TEXT PRIMARY KEY,
            status     TEXT NOT NULL,
            text       TEXT,
            fetched_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS llm_cache (
            key          TEXT PRIMARY KEY,
            response     TEXT NOT NULL,
            created_at   REAL NOT NULL
        );
    """)
    c.commit()


# ───────────── scrape cache ─────────────

def get_scrape(url: str) -> Optional[dict]:
    """Return {'status', 'text'} or None if not cached."""
    row = _conn().execute(
        "SELECT status, text FROM scrape_cache WHERE url = ?", (url,)
    ).fetchone()
    if row is None:
        return None
    return {"status": row[0], "text": row[1]}


def set_scrape(url: str, status: str, text: Optional[str]) -> None:
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO scrape_cache (url, status, text, fetched_at) VALUES (?, ?, ?, ?)",
        (url, status, text, time.time()),
    )
    c.commit()


# ───────────── llm cache ─────────────

def make_llm_key(prompt_version: str, target_city: str, target_date: str, article_texts: list) -> str:
    payload = {
        "v": prompt_version,
        "city": target_city,
        "date": target_date,
        "articles": sorted(article_texts),
    }
    blob = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha1(blob).hexdigest()


def get_llm(key: str) -> Optional[dict]:
    row = _conn().execute(
        "SELECT response FROM llm_cache WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    return json.loads(row[0])


def set_llm(key: str, response: dict) -> None:
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO llm_cache (key, response, created_at) VALUES (?, ?, ?)",
        (key, json.dumps(response, ensure_ascii=False), time.time()),
    )
    c.commit()


# ───────────── stats (for debugging / logs) ─────────────

def stats() -> dict:
    c = _conn()
    n_scrape = c.execute("SELECT COUNT(*) FROM scrape_cache").fetchone()[0]
    n_llm = c.execute("SELECT COUNT(*) FROM llm_cache").fetchone()[0]
    by_status = dict(c.execute(
        "SELECT status, COUNT(*) FROM scrape_cache GROUP BY status"
    ).fetchall())
    return {"scrape_total": n_scrape, "scrape_by_status": by_status, "llm_total": n_llm}
