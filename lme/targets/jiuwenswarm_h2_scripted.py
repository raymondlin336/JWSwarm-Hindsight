"""H2 Methodology II (scripted conversations): JW Work + Hindsight in JW's external memory slot,
history fed in directly instead of replayed by the agent (built 2026-10-20).

Same as JiuwenSwarmH2Target (bank per question, JW web UI, Work mode, JW tools as shipped) except:
  - history: no agent chats. Every history turn (user text + the dataset's own assistant reply)
    goes through the same Hindsight provider code JW's slot calls after a turn (sync_turn), one
    provider per session, run by jw_h2/ingest.py in venvs/workswarm-h2s. First user turn of each
    session keeps the "[Today is ...]" prefix, which the provider uses as "now" for that session
  - turns come from <subset>_full.bin (build_full_bundle.py): same sessions/dates/order as the
    normal bundle, plus assistant turns
  - venv venvs/workswarm-h2s, template agentbox/template-ui_h2s (Hindsight plugin defaults: budget
    mid, 4096 tok facts, no raw source text; tools retain + recall), own server --profile h2s
  - only the question is asked in JW (fresh chat); the slot injects Hindsight's recall
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import time
from pathlib import Path

from .jiuwenswarm_h2 import PROVIDER_LOG, JiuwenSwarmH2Target
from .jiuwenswarm_ui import DENY_TEXT

ROOT = Path(__file__).resolve().parents[2]
INGEST = ROOT / "jw_h2" / "ingest.py"
TASKS = "h2_tasks"
# Plan doc (10-20): block local files / ways to cheat. Same list as the file_guard denies in
# template-ui_h2s; this catches prompts the glob rules miss (relative paths got past them in H1).
BLOCKED_PATH_RE = re.compile(
    r"(/agent/sessions(/|$)|/\.checkpoint(/|$)|/h2_tasks/.*(metadata\.json|\.session_id)"
    r"|/appdata/local/dsrc(/|$)|/longmem-bench(/|$)|/ext/hs-replication(/|$)|/ext/hindsight(/|$)"
    r"|/\.pg0(/|$)|/documents/jiuwenswarm(/|$)|/obsidian vaults(/|$))")


def pair_turns(turns: list[dict]) -> list[dict]:
    """[{role, content}] -> [{user, assistant}]: each user turn with the reply that follows it."""
    out = []
    for t in turns:
        if t["role"] == "user":
            out.append({"user": t["content"], "assistant": ""})
        elif t["role"] == "assistant" and out and not out[-1]["assistant"]:
            out[-1]["assistant"] = t["content"]
    return out


class JiuwenSwarmH2ScriptedTarget(JiuwenSwarmH2Target):
    name = "jiuwenswarm_h2_scripted"
    scripted = True
    bank_prefix = "h2s_"   # the Methodology I run uses h2_<qid> on the same server

    def __init__(self, *args, full_items: dict, ingest_python: Path,
                 budget: str = "mid", recall_max_tokens: int = 4096,
                 recall_max_chunk_tokens: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.full_items = full_items
        self.ingest_python = Path(ingest_python)
        self.recall_cfg = {"budget": budget, "recall_max_tokens": recall_max_tokens,
                           "recall_max_chunk_tokens": recall_max_chunk_tokens}

    def _extra_env(self) -> dict[str, str]:
        # own tasks folder per question (H1 rule), not the shared ~/Documents/JiuwenSwarm
        tasks = self.box / TASKS
        tasks.mkdir(exist_ok=True)
        return {"JIUWENSWARM_TASKS_DIR": str(tasks)}

    def _handle_prompts(self) -> bool:
        pg = self.page
        if not pg.locator('button:has-text("Allow once") >> visible=true').count() and \
                not pg.locator('button:has-text("Allow Once") >> visible=true').count():
            return False
        tool, paths = self._pending_ask()
        norm = [p.lower().replace("\\", "/") for p in paths]
        if any(BLOCKED_PATH_RE.search(p) for p in norm):
            for t in DENY_TEXT:
                deny = pg.locator(f'button:has-text("{t}") >> visible=true')
                if deny.count():
                    deny.first.click()
                    break
            self._log(f"prompt REJECTED (blocked path, {tool}): {paths}")
            time.sleep(1)
            return True
        return super()._handle_prompts()

    def ingest(self, question_id: str, with_date) -> list[dict]:
        """Feed the question's history into the bank; returns transcript rows."""
        item = self.full_items[question_id]
        sessions, rows = [], []
        for s_idx, s in enumerate(item["sessions"]):
            turns = pair_turns(s["turns"])
            if turns:
                turns[0]["user"] = with_date(turns[0]["user"], s["date"])
            sessions.append({"session_id": f"scripted_{s['session_id']}", "turns": turns})
            for t_idx, t in enumerate(turns):
                rows.append({"phase": "history", "scripted": True, "session": s_idx,
                             "session_id": s["session_id"], "evidence": s["evidence"],
                             "turn": t_idx, "user": t["user"], "assistant": t["assistant"]})
        job = {"base_url": self.hs, "bank_id": self.bank, **self.recall_cfg, "sessions": sessions}
        env = {**os.environ, "HINDSIGHT_PROVIDER_LOG": str(self.box / PROVIDER_LOG),
               "PYTHONUTF8": "1"}
        r = subprocess.run([str(self.ingest_python), str(INGEST)], input=json.dumps(job),
                           capture_output=True, text=True, encoding="utf-8", env=env, timeout=3600)
        if r.returncode != 0:
            raise RuntimeError(f"ingest failed: {r.stderr[-2000:]}")
        summary = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else "?"
        self._log(f"scripted ingest: {summary}")
        return rows
