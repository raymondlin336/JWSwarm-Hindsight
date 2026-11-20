# coding: utf-8
"""Hindsight memory provider for JiuwenSwarm's external memory slot (longmem-bench H2, 2026-10-20).

LOCAL ADDITION, not part of openjiuwen 0.1.19. Installed only into venvs/workswarm-h2 as
openjiuwen/core/memory/external/hindsight_memory_provider.py (source of truth:
longmem-bench/jw_h2/hindsight_memory_provider.py).

A port of Hindsight's own provider for Hermes Agent (ext/hindsight 0ee8b17,
hindsight-integrations/hermes/__init__.py), which implements the same MemoryProvider
interface (prefetch / sync_turn / tools / system prompt block). Kept from it, with its defaults:
  - sync_turn -> retain: the turn's user + assistant messages ("User: ..." / "Assistant: ...")
    as a JSON message list, appended (update_mode=append) to one document per chat,
    retain_async, tag session:<chat id>, observation_scopes "shared", item timestamp = now
  - prefetch -> recall(query[:800], budget mid, max_tokens 4096) as "- <text>" lines under
    Hermes' "# Hindsight Memory (persistent cross-session context)" header
  - memory_mode hybrid: auto injection + tools hindsight_retain / hindsight_recall /
    hindsight_reflect; Hermes' system prompt block (session_search renamed to JW's
    session_list / session_read)
Changed because of JW's slot:
  - recall runs synchronously against the current message (Hermes' recall_sync=true). Hermes'
    default recalls in the background and injects on the NEXT turn; JW's rail awaits
    prefetch(query) itself (5 s timeout), and JW rebuilds the provider every new chat, so the
    background mode would never inject anything into a chat's first message.
  - recall_types: config, default None = all fact types (what H1's MCP recall used); Hermes
    narrows to observation only.
Benchmark adaptations (user decision 2026-10-20, after the first smoke):
  - "now" = the chat's dataset date: the first message of each replayed chat starts with
    "[Today is 2023/05/26 (Fri) 07:00]"; retain timestamps (and recall's query_timestamp) use
    that date + real time elapsed since, instead of the real clock (smoke 1 stored a trip from
    "last month" as September 2026). Falls back to the real clock if no prefix was seen.
  - on follow-up model calls in a turn JW's rail passes its own injected "<system-reminder>"
    message as the query; such queries are replaced by the turn's real user text.
Config: memory.external.hindsight {base_url, bank_id, budget, recall_types, memory_mode, ...}.
Every prefetch / retain is logged as one JSON line to HINDSIGHT_PROVIDER_LOG (if set).
"""

from __future__ import annotations

import json
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

from openjiuwen.core.common.logging import logger
from openjiuwen.core.memory.external.provider import MemoryProvider

RETAIN_SCHEMA = {
    "name": "hindsight_retain",
    "description": (
        "Store information to long-term memory. Hindsight automatically "
        "extracts structured facts, resolves entities, and indexes for retrieval."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "content": {"type": "string", "description": "The information to store."},
            "context": {"type": "string", "description": "Short label (e.g. 'user preference', 'project decision')."},
            "tags": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional per-call tags to merge with configured default retain tags.",
            },
            "occurred_at": {
                "type": "string",
                "description": (
                    "When the remembered event actually happened, as an ISO-8601 date "
                    "or datetime (e.g. '2026-08-20' or '2026-08-20T14:30:00+02:00'). "
                    "Pass this whenever the memory references a specific event time "
                    "('yesterday', 'last Tuesday', 'on March 3rd') so Hindsight can "
                    "anchor it on the timeline. Omit for timeless facts/preferences."
                ),
            },
        },
        "required": ["content"],
    },
}

RECALL_SCHEMA = {
    "name": "hindsight_recall",
    "description": (
        "Search long-term memory. Returns memories ranked by relevance using "
        "semantic search, keyword matching, entity graph traversal, and reranking."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "description": "What to search for."}},
    },
}

REFLECT_SCHEMA = {
    "name": "hindsight_reflect",
    "description": (
        "Synthesize a reasoned answer from long-term memories. Unlike recall, "
        "this reasons across all stored memories to produce a coherent response."
    ),
    "parameters": {
        "type": "object",
        "required": ["query"],
        "properties": {"query": {"type": "string", "description": "The question to reflect on."}},
    },
}

# Hermes' guidance, with its session_search tool swapped for JW's session tools.
_TOOL_GUIDANCE = (
    "For cross-session facts, user preferences, and past decisions, "
    "prefer hindsight_recall over session_list / session_read — it returns "
    "deduplicated, high-density observations across sessions. "
    "Use hindsight_reflect for cross-session pattern synthesis. "
    "Use session_list / session_read only when you need verbatim transcripts or "
    "exact wording from a specific conversation. "
    "Use hindsight_retain to store facts."
)
_CONTEXT_NOTE = "Relevant memories are automatically injected into context."
_SYSTEM_PROMPT_TAILS = {
    "context": _CONTEXT_NOTE,
    "tools": _TOOL_GUIDANCE,
    "hybrid": f"{_CONTEXT_NOTE} {_TOOL_GUIDANCE}",
}
_PREFETCH_HEADER = (
    "# Hindsight Memory (persistent cross-session context)\n"
    "Use this to answer questions about the user and prior sessions. "
    "Do not call tools to look up information that is already present here."
)
_RETAIN_CONTEXT = "conversation between JiuwenSwarm Agent and the User"


_TODAY_RE = re.compile(r"^\s*\[Today is (\d{4})/(\d{1,2})/(\d{1,2})(?: \(\w+\))?(?: (\d{1,2}):(\d{2}))?\]")
_REMINDER = "<system-reminder>"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _parse_today(text: str) -> datetime | None:
    m = _TODAY_RE.match(text or "")
    if not m:
        return None
    y, mo, d, hh, mm = m.groups()
    local_tz = datetime.now().astimezone().tzinfo
    return datetime(int(y), int(mo), int(d), int(hh or 0), int(mm or 0), tzinfo=local_tz)


def _types_list(value: Any) -> list[str] | None:
    if not value:
        return None
    if isinstance(value, str):
        value = [v.strip() for v in value.split(",")]
    return [str(v).strip() for v in value if str(v).strip()] or None


class HindsightMemoryProvider(MemoryProvider):
    def __init__(
        self,
        *,
        base_url: str,
        bank_id: str,
        budget: str = "mid",
        recall_max_tokens: int = 4096,
        recall_max_input_chars: int = 800,
        recall_types: Any = None,
        recall_max_chunk_tokens: int = 0,
        memory_mode: str = "hybrid",
        tools: Any = None,
        auto_recall: bool = True,
        auto_retain: bool = True,
        retain_async: bool = True,
        timeout: float = 120.0,
        log_path: str = "",
    ):
        self._base_url = base_url.rstrip("/")
        self._bank_id = bank_id
        self._budget = budget
        self._recall_max_tokens = int(recall_max_tokens)
        self._recall_max_input_chars = int(recall_max_input_chars)
        self._recall_types = _types_list(recall_types)
        # > 0: also return the raw source text the facts came from (H2 Methodology II, 2026-10-20);
        # 0 = facts only, as Hermes and the H2 Methodology I run
        self._recall_max_chunk_tokens = int(recall_max_chunk_tokens or 0)
        self._memory_mode = memory_mode if memory_mode in _SYSTEM_PROMPT_TAILS else "hybrid"
        # which tools to expose: None = all three (Hermes); the plan doc's H2 = retain + recall
        names = _types_list(tools)
        self._tools = {f"hindsight_{n}" if not n.startswith("hindsight_") else n for n in names} \
            if names else {"hindsight_retain", "hindsight_recall", "hindsight_reflect"}
        self._auto_recall = auto_recall
        self._auto_retain = auto_retain
        self._retain_async = retain_async
        self._timeout = timeout
        self._log_path = log_path or os.environ.get("HINDSIGHT_PROVIDER_LOG", "")
        self._client = None
        self._session_id = ""
        self._turn_index = 0
        self._initialized = False
        self._chat_date: datetime | None = None    # dataset date of this chat ([Today is ...])
        self._chat_date_seen = 0.0                 # monotonic time it was seen
        self._last_query = ""                      # this turn's real user text

    def _note_user_text(self, text: str) -> str:
        """Track the chat's dataset date; map JW's '<system-reminder>' pseudo-query back to
        the turn's real user text."""
        if text.lstrip().startswith(_REMINDER):
            return self._last_query
        d = _parse_today(text)
        if d is not None:
            self._chat_date, self._chat_date_seen = d, time.monotonic()
        self._last_query = text
        return text

    def _chat_now_iso(self) -> str:
        if self._chat_date is None:
            return _now_iso()
        now = self._chat_date + timedelta(seconds=time.monotonic() - self._chat_date_seen)
        return now.isoformat(timespec="seconds")

    # -- MemoryProvider basics ------------------------------------------------

    @property
    def name(self) -> str:
        return "hindsight"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def is_available(self) -> bool:
        return bool(self._base_url and self._bank_id)

    def _event(self, kind: str, **fields: Any) -> None:
        if not self._log_path:
            return
        rec = {"t": _now_iso(), "kind": kind, "bank": self._bank_id, "session": self._session_id, **fields}
        try:
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except Exception:
            pass

    async def initialize(self, **kwargs: Any) -> None:
        from hindsight_client import Hindsight

        self._session_id = str(kwargs.get("session_id") or "").strip()
        # Created inside the running loop: the client's aiohttp session is loop-bound.
        self._client = Hindsight(base_url=self._base_url, timeout=self._timeout,
                                 user_agent="hindsight-jiuwenswarm-h2/0.1")
        self._initialized = True
        self._event("init", mode=self._memory_mode, budget=self._budget, types=self._recall_types)

    def system_prompt_block(self) -> str:
        mode = self._memory_mode
        label = "" if mode == "hybrid" else f" ({mode} mode)"
        tail = _TOOL_GUIDANCE if mode == "hybrid" and not self._auto_recall else _SYSTEM_PROMPT_TAILS[mode]
        if "hindsight_reflect" not in self._tools:   # don't point the agent at a tool it lacks
            tail = tail.replace("Use hindsight_reflect for cross-session pattern synthesis. ", "")
        return f"# Hindsight Memory\nActive{label}. Bank: {self._bank_id}, budget: {self._budget}.\n{tail}"

    # -- recall -------------------------------------------------------------------

    async def _recall_full(self, query: str) -> tuple[list, list]:
        """(facts, raw source chunks); chunks only when recall_max_chunk_tokens > 0."""
        kwargs: dict[str, Any] = {"bank_id": self._bank_id, "query": query, "budget": self._budget,
                                  "max_tokens": self._recall_max_tokens}
        if self._chat_date is not None:
            kwargs["query_timestamp"] = self._chat_now_iso()
        if self._recall_types:
            kwargs["types"] = self._recall_types
        if self._recall_max_chunk_tokens > 0:
            kwargs["include_chunks"] = True
            kwargs["max_chunk_tokens"] = self._recall_max_chunk_tokens
        resp = await self._client.arecall(**kwargs)
        chunks = [c for c in (resp.chunks or {}).values() if getattr(c, "text", None)]
        return list(resp.results or []), chunks

    async def _recall(self, query: str) -> list:
        return (await self._recall_full(query))[0]

    def _format(self, results: list, chunks: list) -> str:
        text = "\n".join(f"- {r.text}" for r in results if getattr(r, "text", None))
        if chunks:
            excerpts = "\n\n".join(c.text.strip() for c in chunks)
            text = f"{text}\n\n## Source excerpts (raw conversation text the memories came from)\n\n{excerpts}"
        return text

    async def prefetch(self, query: str, **kwargs: Any) -> str:
        if not self._initialized or not query or self._memory_mode == "tools" or not self._auto_recall:
            return ""
        query = self._note_user_text(query)
        if not query:
            return ""
        q = query[: self._recall_max_input_chars] if self._recall_max_input_chars else query
        t0 = time.monotonic()
        try:
            results, chunks = await self._recall_full(q)
        except BaseException as exc:  # incl. CancelledError from the rail's 5 s timeout
            self._event("prefetch_error", query=q[:200], secs=round(time.monotonic() - t0, 2),
                        error=f"{type(exc).__name__}: {exc}"[:300])
            if getattr(exc, "status", None) == 404:   # bank nothing was retained to yet
                return ""
            raise
        text = self._format(results, chunks)
        self._event("prefetch", query=q[:200], secs=round(time.monotonic() - t0, 2),
                    n=len(results), chunks=len(chunks), chars=len(text))
        return f"{_PREFETCH_HEADER}\n\n{text}" if text else ""

    # -- retain -------------------------------------------------------------------

    def _item(self, content: str, *, context: str | None, tags: list[str] | None,
              occurred_at: str | None = None, update_mode: str | None = None) -> dict[str, Any]:
        item: dict[str, Any] = {
            "content": content,
            "metadata": {
                "retained_at": datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z"),
                "turn_index": str(self._turn_index),
                "session_id": self._session_id,
            },
            "timestamp": (occurred_at or "").strip() or self._chat_now_iso(),
            "observation_scopes": "shared",
        }
        if context is not None:
            item["context"] = context
        if tags:
            item["tags"] = tags
        if update_mode is not None:
            item["update_mode"] = update_mode
        return item

    async def sync_turn(self, user_msg: str, assistant_msg: str, **kwargs: Any) -> None:
        if not self._initialized or not self._auto_retain:
            return
        session_id = str(kwargs.get("session_id") or self._session_id or "").strip()
        if session_id:
            self._session_id = session_id
        user_msg = self._note_user_text(user_msg)
        if not user_msg:
            return
        self._turn_index += 1
        now = self._chat_now_iso()
        messages = [
            {"role": "user", "content": f"User: {user_msg}", "timestamp": now},
            {"role": "assistant", "content": f"Assistant: {assistant_msg}", "timestamp": now},
        ]
        content = json.dumps(messages, ensure_ascii=False)
        tags = [f"session:{self._session_id}"] if self._session_id else None
        # One document per chat; append sends only this turn (server >= 0.5.0).
        item = self._item(content, context=_RETAIN_CONTEXT, tags=tags, occurred_at=now,
                          update_mode="append" if self._session_id else None)
        t0 = time.monotonic()
        try:
            resp = await self._client.aretain_batch(
                bank_id=self._bank_id, items=[item], document_id=self._session_id or None,
                retain_async=self._retain_async)
        except Exception as exc:
            self._event("retain_error", user=user_msg[:200], error=f"{type(exc).__name__}: {exc}"[:300])
            raise
        self._event("retain", turn=self._turn_index, user=user_msg[:300], user_chars=len(user_msg),
                    assistant_chars=len(assistant_msg), secs=round(time.monotonic() - t0, 2),
                    op=str(getattr(resp, "operation_id", "") or ""))

    # -- tools --------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        if self._memory_mode == "context":
            return []
        return [s for s in (RETAIN_SCHEMA, RECALL_SCHEMA, REFLECT_SCHEMA) if s["name"] in self._tools]

    async def handle_tool_call(self, tool_name: str, args: dict) -> str:
        if not self._initialized:
            return json.dumps({"error": "Hindsight provider not initialized"})
        if tool_name not in self._tools:
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        try:
            if tool_name == "hindsight_retain":
                content = str(args.get("content") or "")
                if not content:
                    return json.dumps({"error": "Missing required parameter: content"})
                tags = [str(t) for t in (args.get("tags") or [])] or None
                item = self._item(content, context=args.get("context"), tags=tags,
                                  occurred_at=args.get("occurred_at"))
                await self._client.aretain_batch(bank_id=self._bank_id, items=[item],
                                                 retain_async=self._retain_async)
                self._event("tool_retain", content=content[:300])
                return json.dumps({"result": "Memory stored successfully."})
            if tool_name == "hindsight_recall":
                query = str(args.get("query") or "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                results = await self._recall(query)
                self._event("tool_recall", query=query[:200], n=len(results))
                if not results:
                    return json.dumps({"result": "No relevant memories found."})
                lines = [f"{i}. {r.text}" for i, r in enumerate(results, 1) if getattr(r, "text", None)]
                return json.dumps({"result": "\n".join(lines)}, ensure_ascii=False)
            if tool_name == "hindsight_reflect":
                query = str(args.get("query") or "")
                if not query:
                    return json.dumps({"error": "Missing required parameter: query"})
                resp = await self._client.areflect(bank_id=self._bank_id, query=query, budget=self._budget)
                self._event("tool_reflect", query=query[:200])
                return json.dumps({"result": resp.text or "No relevant memories found."}, ensure_ascii=False)
            return json.dumps({"error": f"Unknown tool: {tool_name}"})
        except Exception as exc:
            if getattr(exc, "status", None) == 404:
                return json.dumps({"result": "No relevant memories found."})
            self._event("tool_error", tool=tool_name, error=f"{type(exc).__name__}: {exc}"[:300])
            return json.dumps({"error": f"{tool_name} failed: {exc}"})

    async def shutdown(self) -> None:
        client, self._client = self._client, None
        self._initialized = False
        if client is not None:
            try:
                await client.aclose()
            except Exception as exc:
                logger.warning("[HindsightMemoryProvider] close failed: %s", exc)

    async def on_session_end(self, messages: list[dict[str, Any]]) -> None:
        # Every turn is already retained in sync_turn; nothing is buffered.
        return


__all__ = ["HindsightMemoryProvider"]
