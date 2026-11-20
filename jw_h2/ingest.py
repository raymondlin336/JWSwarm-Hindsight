"""Methodology II ingestion for H2: feed scripted history turns through the Hindsight provider.

  <venvs/workswarm-h2s>/Scripts/python jw_h2/ingest.py < history.json

stdin JSON: {"base_url", "bank_id", "budget", "recall_max_tokens", "recall_max_chunk_tokens",
             "sessions": [{"session_id", "turns": [{"user", "assistant"}, ...]}, ...]}
Each session = one provider instance (JW rebuilds the provider every chat), initialised with
that session's id, then sync_turn(user, assistant) per turn: exactly what JW's slot calls after
every turn (external_memory_rail.after_invoke), minus the agent. The first user turn carries
the "[Today is ...]" prefix, so the provider dates every turn of the session with it.
Provider events go to HINDSIGHT_PROVIDER_LOG. Prints one JSON summary line.
"""
import asyncio
import json
import sys

from openjiuwen.core.memory.external.hindsight_memory_provider import HindsightMemoryProvider


async def main() -> None:
    job = json.loads(sys.stdin.read())
    n_turns = n_skipped = 0
    for sess in job["sessions"]:
        p = HindsightMemoryProvider(
            base_url=job["base_url"], bank_id=job["bank_id"], budget=job.get("budget", "mid"),
            recall_max_tokens=job.get("recall_max_tokens", 4096),
            recall_max_chunk_tokens=job.get("recall_max_chunk_tokens", 0))
        await p.initialize(session_id=sess["session_id"])
        for t in sess["turns"]:
            # the rail skips a turn with no user text or no reply; do the same
            if not t["user"].strip() or not t["assistant"].strip():
                n_skipped += 1
                continue
            await p.sync_turn(t["user"], t["assistant"], session_id=sess["session_id"])
            n_turns += 1
        await p.shutdown()
    print(json.dumps({"sessions": len(job["sessions"]), "turns": n_turns, "skipped": n_skipped}))


if __name__ == "__main__":
    asyncio.run(main())
