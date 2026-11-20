# JWSwarm-Hindsight

[Hindsight](https://github.com/vectorize-io/hindsight) as an external memory provider for JiuwenSwarm (JW), plus the
LongMemEval wiring used for the H2 (recall) and H3 (reflect) Methodology II runs.

The provider is a port of Hindsight's own Hermes Agent provider (`hindsight-integrations/hermes`) to JW's
`MemoryProvider` interface: it retains every chat turn (user + assistant) and, before every model call, injects
Hindsight's recall (or reflect) result into the prompt. The agent also gets `hindsight_retain` / `hindsight_recall` /
`hindsight_reflect` tools.

Tested with: Windows, Python 3.12, workswarm 0.2.7 (openjiuwen 0.1.19), hindsight-api + hindsight-client 1.0.0,
deepseek-v4-flash as the LLM for both JW and Hindsight.

## What's here

| path | what |
|---|---|
| `jw_h2/hindsight_memory_provider.py` | the provider (recall as the injected block) |
| `jw_h3s/hindsight_memory_provider.py` | same provider + `prefetch_method: reflect` (H3) |
| `jw_h2/ingest.py` | feeds a scripted history through the provider's `sync_turn` (benchmark only) |
| `hindsight_server.py` | starts a Hindsight server with its own embedded Postgres per profile (benchmark helper) |
| `lme/targets/jiuwenswarm_h2*.py`, `jiuwenswarm_h3_scripted.py` | LongMemEval harness targets |
| `agentbox/template-ui_h2s/config/config.yaml`, `template-ui_h3s/...` | the JW configs used for H2 / H3 |

JW itself has no Hindsight provider, so it has to be wired in by hand (part 1). The patch scripts used for the runs
aren't in this repo; the steps below do the same thing.

## Part 1: use Hindsight with JiuwenSwarm

### 1. Hindsight server

Use its own venv, separate from JW's:

```bash
python -m venv venvs/jw-hindsight
venvs/jw-hindsight/Scripts/pip install hindsight-api==1.0.0
```

Start it with an LLM for fact extraction (any provider Hindsight supports; we used DeepSeek):

```bash
HINDSIGHT_API_DATABASE_URL=pg0://jw-hindsight:5433 \
HINDSIGHT_API_PORT=8890 \
HINDSIGHT_API_LLM_PROVIDER=deepseek \
HINDSIGHT_API_LLM_MODEL=deepseek-v4-flash \
HINDSIGHT_API_LLM_API_KEY=<your key> \
venvs/jw-hindsight/Scripts/hindsight-api --host 127.0.0.1 --port 8890
```

`pg0://<name>:<port>` is an embedded Postgres that Hindsight creates on first start. Embeddings (bge-small-en-v1.5)
and the reranker (ms-marco-MiniLM-L-6-v2) run locally on CPU with Hindsight's defaults. `hindsight_server.py` does the
same thing and reads the key from an env file instead (edit its paths first).

### 2. JW venv + Hindsight client

Use a separate venv so your normal JW install stays untouched:

```bash
python -m venv venvs/workswarm-hindsight
venvs/workswarm-hindsight/Scripts/pip install workswarm==0.2.7 hindsight-client==1.0.0
```

Below, `SP` = `venvs/workswarm-hindsight/Lib/site-packages`.

### 3. Add the provider

Copy `jw_h2/hindsight_memory_provider.py` (recall) or `jw_h3s/hindsight_memory_provider.py` (recall or reflect) to:

```
SP/openjiuwen/core/memory/external/hindsight_memory_provider.py
```

### 4. Register it in JW's provider builder

File: `SP/jiuwenswarm/agents/harness/common/memory/external_memory_builder.py`

a) In the provider dispatch, after the `jiuwenmemory` branch, add:

```python
        elif provider_name == "jiuwenmemory":
            provider = _build_jiuwen_provider(ext_cfg, config)
        elif provider_name == "hindsight":
            provider = _build_hindsight_provider(config)
```

b) Add this function right above `def _load_plugin_provider(...)`:

```python
def _build_hindsight_provider(full_config: Optional[Dict[str, Any]] = None):
    """Config memory.external.hindsight (read from the full config: get_external_memory_config()
    only passes the built-in providers' blocks)."""
    from openjiuwen.core.memory.external.hindsight_memory_provider import HindsightMemoryProvider

    cfg = full_config if full_config is not None else _load_config()
    hs = (((cfg or {}).get("memory") or {}).get("external") or {}).get("hindsight") or {}
    provider = HindsightMemoryProvider(
        base_url=str(hs.get("base_url") or os.environ.get("HINDSIGHT_URL", "")),
        bank_id=str(hs.get("bank_id") or os.environ.get("HINDSIGHT_BANK_ID", "")),
        budget=str(hs.get("budget") or "mid"),
        recall_max_tokens=int(hs.get("recall_max_tokens") or 4096),
        recall_types=hs.get("recall_types") or None,
        recall_max_chunk_tokens=int(hs.get("recall_max_chunk_tokens") or 0),
        memory_mode=str(hs.get("memory_mode") or "hybrid"),
        tools=hs.get("tools") or None,
        prefetch_method=str(hs.get("prefetch_method") or "recall"),  # only with the jw_h3s provider
    )
    if not provider.is_available():
        logger.warning("[ExternalMemoryBuilder] Hindsight unavailable (no base_url / bank_id)")
        return None
    logger.info("[ExternalMemoryBuilder] Hindsight provider built (bank=%s)", hs.get("bank_id"))
    return provider
```

With the `jw_h2` provider, drop the `prefetch_method=` line (that version doesn't take it).

### 5. Fix two things in JW's external memory rail

File: `SP/openjiuwen/harness/rails/memory/external_memory_rail.py`

a) **Prefetch timeout.** JW drops any prefetch slower than 5 s. Hindsight's recall with the CPU reranker took ~13 s
on ~250 facts, so most injections were silently lost. Raise it:

```python
    PREFETCH_TIMEOUT = 60.0   # was 5.0
```

b) **Message envelope.** JW's web channel doesn't set `raw_query`, so the text the rail hands to memory is JW's
envelope (`You receive a new message:{...json...}`), not what the user typed. Rename the existing
`_resolve_user_text_for_memory` to `_resolve_user_text_for_memory_orig` and add a new one above it that unwraps the
envelope:

```python
    @staticmethod
    def _resolve_user_text_for_memory(ctx: AgentCallbackContext) -> str:
        text = ExternalMemoryRail._resolve_user_text_for_memory_orig(ctx)
        prefix = "You receive a new message:"
        if text.startswith(prefix):
            try:
                env = json.loads(text[len(prefix):].strip())
                if isinstance(env, dict) and isinstance(env.get("content"), str) and env["content"].strip():
                    return env["content"].strip()
            except Exception:
                pass
        return text
```

Keep a copy of both original files before editing.

### 6. Turn it on in JW's config

In your JW data folder's `config/config.yaml`, under `memory:` (see the template configs here for the full file):

```yaml
memory:
  engine: both                 # JW's own memory + the external provider; or set it to external only
  external:
    provider: hindsight
    hindsight:
      base_url: http://127.0.0.1:8890
      bank_id: my-bank         # one memory bank; Hindsight creates it on the first retain
      budget: mid              # recall effort: low | mid | high
      recall_max_tokens: 4096  # size of the injected block
      recall_types: ""         # empty = all fact types (world, experience, observation)
      memory_mode: hybrid      # hybrid = auto inject + tools | context | tools
      tools: retain,recall     # which tools the agent gets (retain,recall,reflect)
      # prefetch_method: reflect   # jw_h3s provider only: inject reflect's answer instead of recall
```

The external memory slot is only used in Work mode (JW's Code space has none). Start JW from this venv as usual. Each
prefetch and retain is logged as one JSON line to the file in `HINDSIGHT_PROVIDER_LOG`, if that's set.

### Things to know

- **Dates.** If a chat's first message starts with `[Today is 2023/05/26 (Fri) 07:00]`, the provider uses that date
  as "now" for retain timestamps and recall's `query_timestamp` (and passes it to reflect as context). Without the
  prefix it uses the real clock. That's for the benchmark's replayed histories; normal chats aren't affected.
- **Follow-up model calls.** Within a turn, JW passes its own `<system-reminder>` message as the prefetch query; the
  provider swaps that back to the turn's real user text.
- **Both sides of the chat are stored** (`User: ...` / `Assistant: ...`), one Hindsight document per chat.
- **Slower model calls.** Every model call waits for recall (~5-10 s with the CPU reranker) or reflect (~10-40 s).

## Part 2: rerun the LongMemEval benchmarks

These targets plug into the LongMemEval harness in
[jiuwenswarm-longmemeval](https://github.com/raymondlin336/jiuwenswarm-longmemeval): they import its
`lme/targets/jiuwenswarm_ui.py`, `jw_instance.py` and `lme.bundle`.

1. Copy this repo's `jw_h2/`, `jw_h3s/`, `lme/targets/*` and `hindsight_server.py` into the harness folder.
2. Do part 1 twice, into two venvs: `workswarm-h2s` (jw_h2 provider) and `workswarm-h3s` (jw_h3s provider).
3. JW templates: copy a stock JW data folder (the harness's `template-ui_standard`) to
   `<AGENT_SANDBOX>/template-ui_h2s` and `template-ui_h3s`, then drop in this repo's `config.yaml` for each.
   Leave `__HINDSIGHT_URL__` / `__HINDSIGHT_BANK__` as they are: the target fills them per question (one bank per
   question, emptied at the start, exported + deleted at the end). The `file_guard` denies at the end of each config
   block the agent from reading run files, the dataset, old chats and the Hindsight database.
4. Build the full bundle (keeps the dataset's assistant turns), with the harness's `build_full_bundle.py`:
   `python build_full_bundle.py subsets/shared40.bin` → `subsets/shared40_full.bin`.
5. Register the targets in the harness's `run.py` (add `jiuwenswarm_h2_scripted` / `jiuwenswarm_h3_scripted` to
   `--target` choices and build them in the target factory):

   ```python
   if args.target == "jiuwenswarm_h2_scripted":
       from lme import bundle
       from lme.targets.jiuwenswarm_h2_scripted import JiuwenSwarmH2ScriptedTarget
       env = {**dotenv_values(args.env_file), **os.environ}
       model_env = {k: env[k] for k in ("MODEL_PROVIDER", "API_BASE", "API_KEY", "MODEL_NAME") if env.get(k)}
       sandbox = Path(env["AGENT_SANDBOX"])
       full = Path(args.subset).with_name(Path(args.subset).stem + "_full.bin")
       scripts = Path(env["WORKSWARM_H2S_SCRIPTS"])          # venvs/workswarm-h2s/Scripts
       return JiuwenSwarmH2ScriptedTarget(
           scripts_dir=scripts, sandbox_root=sandbox, template_dir=sandbox / "template-ui_h2s",
           model_env=model_env, reply_timeout=args.timeout,
           hindsight_url=env.get("HINDSIGHT_H2S_URL", "http://127.0.0.1:8894"),
           full_items=bundle.load(full)["items"], ingest_python=scripts / "python.exe")
   ```

   H3 is the same with `JiuwenSwarmH3ScriptedTarget`, `template-ui_h3s`, `WORKSWARM_H3S_SCRIPTS` and
   `HINDSIGHT_H3S_URL` (default port 8896).
6. Start the matching Hindsight server and leave it running for the whole run:
   `python hindsight_server.py <env file> --profile h2s` (port 8894) or `--profile h3s` (port 8896).
   Each profile has its own embedded Postgres, so H2 and H3 never share memory.
7. Run: `python run.py --target jiuwenswarm_h2_scripted --subset subsets/shared40.bin ...`

Per question, the target feeds the whole history (user turn + the dataset's own reply) through `jw_h2/ingest.py`,
waits until Hindsight has finished processing, then asks only the question in a fresh JW chat. Results: H2 35/40,
H3 37/40 on shared40 (deepseek-v4-flash, hand-graded).
