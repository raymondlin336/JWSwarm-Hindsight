"""Start the Hindsight server for the JiuwenSwarm + Hindsight test (own database + port).

  C:/Users/A2306101770/venvs/jw-hindsight/Scripts/python hindsight_server.py [env-file] [--profile h2|h3|h2s|h3s|h2r]

Reads the DeepSeek key from the env file (default: personal-values-layer/.env, same as the
JW runs) so it never appears on a command line. Logs to runs/_hindsight_server.log.
"""
import os
import subprocess
import sys
from pathlib import Path

ARGS = sys.argv[1:]
PROFILE = ""
if "--profile" in ARGS:
    i = ARGS.index("--profile")
    PROFILE = ARGS[i + 1]
    del ARGS[i:i + 2]
# H2 (2026-10-20, Hindsight in JW's external memory slot) gets its own database + port, so it
# never shares a store with H1 (default profile) or the Hindsight Bench session (5434).
PROFILES = {
    "": {"db": "pg0://jw-hindsight:5433", "port": "8890", "log": "_hindsight_server.log"},
    "h2": {"db": "pg0://jw-hindsight-h2:5436", "port": "8893", "log": "_hindsight_server_h2.log"},
    # H3 (2026-10-20): H1 with recall swapped for reflect (Hindsight's own LLM answers from memory)
    "h3": {"db": "pg0://jw-hindsight-h3:5437", "port": "8895", "log": "_hindsight_server_h3.log",
           "tools": "retain,reflect"},
    # H2 Methodology II (2026-10-20, scripted conversations): own store, apart from the H2 I run
    "h2s": {"db": "pg0://jw-hindsight-h2s:5438", "port": "8894", "log": "_hindsight_server_h2s.log"},
    # H3 Methodology II (2026-10-20): H2s with recall swapped for reflect, own store
    "h3s": {"db": "pg0://jw-hindsight-h3s:5439", "port": "8896", "log": "_hindsight_server_h3s.log",
            "tools": "retain,reflect"},
    # H2 retrieval-only eval (2026-10-23, retrieval_eval/): fresh ingest of the H2s history, recall
    # scored against LongMemEval evidence; own store, banks kept for later knock-out recalls
    "h2r": {"db": "pg0://jw-hindsight-h2r:5440", "port": "8897", "log": "_hindsight_server_h2r.log"},
}
P = PROFILES[PROFILE]
ENV_FILE = Path(ARGS[0] if ARGS else r"C:\Users\A2306101770\Documents\personal-values-layer\.env")
EXE = Path(r"C:\Users\A2306101770\venvs\jw-hindsight\Scripts\hindsight-api.exe")
LOG = Path(__file__).resolve().parent / "runs" / P["log"]


def read_env(path: Path) -> dict[str, str]:
    out = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip().strip('"').strip("'")
    return out


def main() -> None:
    src = read_env(ENV_FILE)
    env = {
        **os.environ,
        "HINDSIGHT_API_DATABASE_URL": P["db"],
        "HINDSIGHT_API_PORT": P["port"],
        "HINDSIGHT_API_MCP_ENABLED_TOOLS": P.get("tools", "retain,recall"),
        "HINDSIGHT_API_LLM_PROVIDER": "deepseek",
        "HINDSIGHT_API_LLM_MODEL": src.get("MODEL_NAME", "deepseek-v4-flash"),
        "HINDSIGHT_API_LLM_API_KEY": src["API_KEY"],
        "PYTHONUTF8": "1",
    }
    LOG.parent.mkdir(exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as log:
        proc = subprocess.Popen([str(EXE), "--host", "127.0.0.1", "--port", P["port"]], env=env,
                                stdout=log, stderr=subprocess.STDOUT)
        print(f"hindsight-api pid {proc.pid}, log {LOG}")
        proc.wait()


if __name__ == "__main__":
    main()
