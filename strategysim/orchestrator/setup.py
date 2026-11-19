"""加载策略和代表性人群，注入访问日程、用户历史与基准购买率。"""
from __future__ import annotations

import json
import csv
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from ..agents.wakeup_coupon_agent import WakeupCouponAgent
from ..envs.wakeup_coupon_env import WakeupCouponEnv
from .controller import SimController
from .settings import POS_RATIO_CSV


def load_baseline_rates() -> Dict[int, float]:
    with open(POS_RATIO_CSV, newline="", encoding="utf-8") as stream:
        return {
            int(row["user_id"]): float(row["baseline_rate"])
            for row in csv.DictReader(stream)
        }


def load_init_config(config_path: str) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def create_env_from_config(env_config: dict) -> WakeupCouponEnv:
    kwargs = env_config.get("kwargs", {})
    return WakeupCouponEnv(**kwargs)


def create_agent_from_config(agent_config: dict) -> WakeupCouponAgent:
    kwargs = agent_config.get("kwargs", {})
    return WakeupCouponAgent(**kwargs)


def _init_env_and_agents(
    config: dict,
    output_path: Path,
) -> Tuple[WakeupCouponEnv, SimController, List[Tuple[int, WakeupCouponAgent]]]:
    """
    从 init_config 创建 Env、SimController 和所有 Agent 实例。
    两种时钟模式共用此初始化逻辑。
    """
    env_configs = config.get("env_modules", [])
    if not env_configs:
        raise ValueError("init_config.json 中没有 env_modules")
    pos_ratio_map = load_baseline_rates()
    env = create_env_from_config(env_configs[0])
    controller = SimController(env=env, output_dir=output_path, pos_ratio_map=pos_ratio_map)

    active_user_ids = set(env.active_user_ids())
    agent_configs = config.get("agents", [])

    agents_to_run: List[Tuple[int, WakeupCouponAgent]] = []
    skipped_no_pos_ratio = 0
    for ac in agent_configs:
        kwargs = dict(ac.get("kwargs", {}))
        profile = dict(kwargs.get("profile", {}))
        kwargs["profile"] = profile
        kwargs.setdefault("include_domain_knowledge", config.get("include_domain_knowledge", True))
        uid = int(profile.get("user_id", ac.get("agent_id", 0)))
        if uid not in active_user_ids:
            continue
        if uid not in pos_ratio_map:
            skipped_no_pos_ratio += 1
            continue
        agent = create_agent_from_config({"kwargs": kwargs})
        agent.bind_environment(env)

        wakeup_seq = env.get_wakeup_sequence(uid)
        agent.set_wakeup_sequence(wakeup_seq)
        history = env.get_user_actions(uid)
        if history:
            agent.load_history_actions(history)

        if config.get("log_model_messages", False):
            agent.set_llm_io_dir(output_path / "llm_io")
        agents_to_run.append((uid, agent))

    if skipped_no_pos_ratio:
        print(f"  跳过无 pos_ratio 用户: {skipped_no_pos_ratio}")

    return env, controller, agents_to_run


def _print_env_info(env: WakeupCouponEnv, agents_to_run, agent_configs):
    active_user_ids = set(env.active_user_ids())
    total_events = sum(len(env.get_user_timeline(uid)) for uid in active_user_ids)
    print(f"  活跃用户数: {len(active_user_ids)}")
    print(f"  总事件数: {total_events}")
    print(f"  需要运行的 Agent: {len(agents_to_run)} "
          f"（共 {len(agent_configs)} 个配置，"
          f"{len(agent_configs) - len(agents_to_run)} 个无事件跳过）")
    print()
