"""仿真命令行与原有公共调用入口。"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
env_file = WORKSPACE_ROOT / ".env"
if env_file.exists():
    try:
        from dotenv import load_dotenv
        load_dotenv(env_file)
    except ImportError:
        pass

sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(WORKSPACE_ROOT / "strategysim"))


from strategysim.orchestrator.settings import (
    DEFAULT_CLOCK_MODE, DEFAULT_CONCURRENCY, DEFAULT_CONFIG_PATH,
    DEFAULT_OUTPUT_DIR, DEFAULT_STEP_MINUTES,
)
from strategysim.orchestrator.setup import (
    create_agent_from_config, create_env_from_config, load_init_config, load_baseline_rates,
)
from strategysim.orchestrator.aggregation import aggregate_wakeup_from_visits, compute_rollout_weight
from strategysim.orchestrator.controller import SimController
from strategysim.orchestrator.simulation import run_independent_simulation, run_single_agent, run_sync_simulation


def main():
    parser = argparse.ArgumentParser(
        description="购物优惠券消费仿真（三层解耦，支持独立/同步两种时钟模式）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
时钟模式说明：
  independent（默认）：每个 Agent 独立遍历自身事件时间线，不同 Agent 完全并行。
                       适合离线仿真、真实行为回放、模型校准。
  sync               ：全局统一时钟按固定步长推进，每 tick 内并行处理命中 Agent。
                       不同 tick 之间串行。适合多模块联合仿真、策略实验、时间轴对齐。

示例:
  # 独立时钟（默认）
  python strategysim/run_independent.py
  python strategysim/run_independent.py --mode independent --concurrency 20

  # 同步时钟，步长 1 分钟
  python strategysim/run_independent.py --mode sync --step 1

  # 同步时钟，步长 5 分钟，自定义输出目录
  python strategysim/run_independent.py --mode sync --step 5 --output-dir outputs/sync
        """,
    )
    parser.add_argument("--config", "-c", type=str, default=str(DEFAULT_CONFIG_PATH),
                        help="初始化配置文件路径（默认：config.json）")
    parser.add_argument("--concurrency", "-n", type=int, default=DEFAULT_CONCURRENCY,
                        help="最大并发 Agent 数（默认：30）")
    parser.add_argument("--output-dir", "-o", type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="结果输出目录（默认：outputs）")
    parser.add_argument(
        "--mode", "-m",
        type=str,
        default=DEFAULT_CLOCK_MODE,
        choices=["independent", "sync"],
        help="时钟模式：independent=独立时钟（默认），sync=同步时钟",
    )
    parser.add_argument(
        "--step", "-s",
        type=int,
        default=DEFAULT_STEP_MINUTES,
        metavar="MINUTES",
        help="同步时钟步长（分钟，仅 --mode sync 有效，默认：1）",
    )
    args = parser.parse_args()

    if args.mode == "sync":
        result = asyncio.run(run_sync_simulation(
            config_path=args.config,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
            step_minutes=args.step,
        ))
    else:
        result = asyncio.run(run_independent_simulation(
            config_path=args.config,
            concurrency=args.concurrency,
            output_dir=args.output_dir,
        ))

    failed = result.get("failed_agents", 0)
    if failed > 0:
        print(f"\n⚠ 有 {failed} 个 Agent 执行失败，请检查日志。")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
