"""H2: JiuwenSwarm (web UI, Work mode) with Hindsight in JW's external memory slot (built 2026-10-20).

Same flow as JiuwenSwarmUITarget (Work default: shipped tools, standard ~/Documents/JiuwenSwarm
tasks dir, no prompt rejections); JW's own memory stays on (memory.engine both). Differences:
  - venv venvs/workswarm-h2: workswarm 0.2.7 at venvs/workswarm's exact versions + hindsight-client
    1.0.0, plus jw_h2/apply_patches.py (Hindsight provider, builder dispatch, envelope unwrap)
  - template agentbox/template-ui_h2 (= ui_standard); only config.yaml memory lines differ:
    engine both, external.provider hindsight + its memory.external.hindsight block
  - one Hindsight bank per question (made empty at the start, exported + deleted at the end);
    the provider prefetches (recall) before every model call and retains every turn
  - before the memory snapshot / question, wait until the bank has no pending or processing
    operations (retain extraction + consolidation run in the background)
  - every prefetch / retain the provider makes is logged to <box>/hindsight_provider.jsonl,
    copied to the instance dir

Hindsight server: hindsight_server.py --profile h2 (pg0://jw-hindsight-h2:5436, port 8893).
"""
from __future__ import annotations

import json
import re
import shutil
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path

from .jiuwenswarm_ui import JiuwenSwarmUITarget, UITargetError
from .jw_instance import JWInstance

BANK_TOKEN = "__HINDSIGHT_BANK__"
URL_TOKEN = "__HINDSIGHT_URL__"
PROVIDER_LOG = "hindsight_provider.jsonl"


class HindsightError(RuntimeError):
    pass


class JiuwenSwarmH2Target(JiuwenSwarmUITarget):
    name = "jiuwenswarm_h2"
    bank_prefix = "h2_"

    def __init__(self, *args, hindsight_url: str = "http://127.0.0.1:8893",
                 ingest_timeout: int = 1800, **kwargs):
        super().__init__(*args, use_project=False, code_space=False, **kwargs)
        self.hs = hindsight_url.rstrip("/")
        self.ingest_timeout = ingest_timeout
        self.bank: str | None = None

    # ---- Hindsight HTTP (same calls as jiuwenswarm_hindsight) -----------------

    def _hs(self, method: str, path: str, body: dict | None = None, timeout: int = 60):
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(f"{self.hs}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                raw = r.read().decode("utf-8", errors="replace")
                return json.loads(raw) if raw.strip() else {}
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return None
            raise HindsightError(f"{method} {path}: HTTP {e.code} {e.read()[:300]!r}") from e

    def _bank_path(self) -> str:
        return f"/v1/default/banks/{self.bank}"

    def _busy_ops(self) -> int:
        busy = 0
        for status in ("pending", "processing"):
            res = self._hs("GET", f"{self._bank_path()}/operations?status={status}") or {}
            ops = res.get("operations", res.get("items", [])) if isinstance(res, dict) else res
            busy += res.get("total", len(ops)) if isinstance(res, dict) else len(ops)
        return busy

    def _wait_bank_idle(self) -> None:
        t0 = time.time()
        while time.time() - t0 < self.ingest_timeout:
            if not self._busy_ops():
                self._log(f"hindsight bank idle after {time.time() - t0:.0f}s")
                return
            time.sleep(5)
        self._log(f"hindsight still busy after {self.ingest_timeout}s; continuing")

    def _extra_env(self) -> dict[str, str]:
        """Extra app environment (subclasses); the Methodology I run uses none."""
        return {}

    # ---- Target API ----------------------------------------------------------

    def reset(self, instance_dir: Path) -> None:
        self.instance_dir = Path(instance_dir)
        self.instance_dir.mkdir(parents=True, exist_ok=True)
        (self.instance_dir / "ui_events.log").unlink(missing_ok=True)
        self.bank = self.bank_prefix + re.sub(r"[^A-Za-z0-9_-]", "_", self.instance_dir.name)
        self._hs("DELETE", self._bank_path())
        # Hindsight only creates a bank on the first retain; a recall before that is a 404.
        self._hs("PUT", self._bank_path(), {})
        self.box = self.sandbox_root / uuid.uuid4().hex[:10]
        shutil.copytree(self.template_dir, self.box / "data",
                        ignore=shutil.ignore_patterns(".initialised", "init.log"))
        cfg = self.box / "data" / "config" / "config.yaml"
        text = cfg.read_text(encoding="utf-8")
        if BANK_TOKEN not in text or URL_TOKEN not in text:
            raise UITargetError(f"{BANK_TOKEN}/{URL_TOKEN} missing from {cfg}")
        cfg.write_text(text.replace(BANK_TOKEN, self.bank).replace(URL_TOKEN, self.hs),
                       encoding="utf-8", newline="")
        (self.instance_dir / "sandbox.txt").write_text(str(self.box), encoding="utf-8")
        env = {**self.model_env, "HINDSIGHT_PROVIDER_LOG": str(self.box / PROVIDER_LOG),
               **self._extra_env()}
        self.inst = JWInstance(self.scripts_dir, self.box / "data", env,
                               self.box / "launch.log").start()
        self._log(f"app started, web UI {self.inst.url}, hindsight bank {self.bank}")
        self.page = self._browser_page()
        self.page.goto(self.inst.url, wait_until="networkidle", timeout=180000)
        time.sleep(3)
        self._dismiss_guide()
        self._first_session = True

    def memory_snapshot(self) -> dict[str, str]:
        # Called right before the question chat: everything retained so far must be searchable.
        self._wait_bank_idle()
        files = super().memory_snapshot()
        listing = self._hs("GET", f"{self._bank_path()}/memories/list?limit=1000") or {}
        # not truncated: a cut JSON listing can't be parsed (smoke 2026-10-20 hit a 300k cap)
        files[f"hindsight/{self.bank}/memories.json"] = json.dumps(listing, ensure_ascii=False, indent=1)
        return files

    def finish_instance(self) -> None:
        if self.box and self.instance_dir and (self.box / PROVIDER_LOG).exists():
            shutil.copy2(self.box / PROVIDER_LOG, self.instance_dir / PROVIDER_LOG)
        if self.bank and self.instance_dir:
            try:
                self._wait_bank_idle()
                dump = self._hs("GET", f"{self._bank_path()}/export", timeout=300)
                (self.instance_dir / "hindsight_export.json").write_text(
                    json.dumps(dump, ensure_ascii=False, indent=1), encoding="utf-8")
                self._hs("DELETE", self._bank_path())
            except Exception as e:  # noqa: BLE001 - keep the JW data even if this fails
                self._log(f"hindsight export/delete failed: {e}")
        super().finish_instance()
