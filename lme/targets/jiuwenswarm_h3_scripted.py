"""H3 Methodology II: H2 Methodology II with Hindsight's recall swapped for reflect (built 2026-10-20).

Identical to JiuwenSwarmH2ScriptedTarget (history fed through the slot provider's sync_turn with the
dataset's replies, only the question asked in a fresh JW Work chat, file denies + prompt
rejection, own tasks folder, bank per question) except:
  - venv venvs/workswarm-h3s: same freeze as workswarm-h2s, provider from jw_h3s/ (H2's + Hermes'
    recall_prefetch_method), installed by jw_h3s/apply_patches.py
  - template agentbox/template-ui_h3s: H2s config with tools retain + reflect and
    prefetch_method reflect, i.e. the injected block is reflect's answer (budget mid = plugin default)
  - own server: hindsight_server.py --profile h3s (pg0 5439, port 8896); banks h3s_<qid>
"""
from __future__ import annotations

from .jiuwenswarm_h2_scripted import JiuwenSwarmH2ScriptedTarget


class JiuwenSwarmH3ScriptedTarget(JiuwenSwarmH2ScriptedTarget):
    name = "jiuwenswarm_h3_scripted"
    bank_prefix = "h3s_"
