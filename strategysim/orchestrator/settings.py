"""路径和启动参数。"""
from __future__ import annotations

import os
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[2]


DEFAULT_CONFIG_PATH = WORKSPACE_ROOT / "config.json"
DEFAULT_OUTPUT_DIR = WORKSPACE_ROOT / "outputs"
DEFAULT_CONCURRENCY = 30
DEFAULT_BEHAVIOR_JSON = "agent_behavior_records.json"
DEFAULT_CLOCK_MODE = "independent"
DEFAULT_STEP_MINUTES = 1

POS_RATIO_CSV = Path(os.getenv("SIM_BASELINE_RATES", "data/baseline_rates.csv"))
