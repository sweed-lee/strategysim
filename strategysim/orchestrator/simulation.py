"""独立时钟与同步时钟的调度和续跑。"""
from __future__ import annotations

import asyncio
import time
from typing import Dict, List
from datetime import datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.wakeup_coupon_agent import WakeupCouponAgent
    from ..envs.wakeup_coupon_env import WakeupCouponEnv
from .controller import SimController
from .setup import _init_env_and_agents, _print_env_info, load_init_config
from .reporting import (
    _append_wakeup_record, _collect_summary_stats, _load_results_from_records,
    _print_summary, _write_env_final_status, _write_outputs,
)
from .settings import (
    DEFAULT_BEHAVIOR_JSON, DEFAULT_CONFIG_PATH, DEFAULT_CONCURRENCY, DEFAULT_OUTPUT_DIR,
    DEFAULT_STEP_MINUTES,
)

try:
    from tqdm.asyncio import tqdm_asyncio
except ImportError:
    tqdm_asyncio = None


async def run_single_agent(
    agent: WakeupCouponAgent,
    controller: SimController,
    semaphore: asyncio.Semaphore,
    output_dir: Path,
) -> Dict:
    """在 Semaphore 保护下执行单个 Agent 的完整独立时钟仿真。"""
    async with semaphore:
        profile = agent.get_profile() or {}
        user_id = agent.user_id
        env = controller.env
        start_time = time.time()

        try:

            wakeup_seq = env.get_wakeup_sequence(user_id)
            agent.set_wakeup_sequence(wakeup_seq)


            history = env.get_user_actions(user_id)
            if history:
                agent.load_history_actions(history)


            timeline = env.get_user_timeline(user_id)
            if not timeline:
                return {
                    "user_id": user_id,
                    "status": "skip",
                    "reason": "no_events",
                    "wakeup_count": 0, "order_count": 0,
                    "total_spend": 0.0, "total_gtv": 0.0, "coupon_used_count": 0,
                    "coupon_saved": 0.0, "events_processed": 0,
                    "wakeup_records": [],
                }


            wakeup_count = 0
            order_count = 0
            wakeup_records: List[Dict] = []

            for event in timeline:
                event_type = event["type"]
                event_time: datetime = event["time"]
                event_idx = event.get("event_idx", -1)


                if env.is_event_consumed(user_id, event_idx):
                    continue
                env.mark_event_consumed(user_id, event_idx)

                if event_type == "coupon":

                    controller.advance_time(event_time, user_id=user_id)

                elif event_type == "wakeup":
                    wakeup_time: datetime = event_time
                    wakeup_count += 1
                    env.record_wakeup(user_id)


                    record = await controller.run_single_wakeup_flow(agent, wakeup_time)
                    wakeup_records.append(record)

                    if record.get("consumed"):
                        order_count += float(record.get("rollout_weight", 1.0))


            elapsed = time.time() - start_time
            total_spend = sum(
                float(r.get("final_price", 0))
                for r in wakeup_records if r.get("consumed")
            )
            total_gtv = sum(
                float(r.get("order_final_price", r.get("final_price", 0)))
                for r in wakeup_records if r.get("consumed")
            )
            coupon_used_count = sum(
                1 for r in wakeup_records if r.get("coupon_used")
            )
            coupon_saved = sum(
                float(r.get("coupon_amount", 0))
                for r in wakeup_records if r.get("coupon_used")
            )

            result = {
                "user_id": user_id,
                "agent_name": agent.display_name,
                "status": "success",
                "wakeup_count": wakeup_count,
                "order_count": order_count,
                "total_spend": round(total_spend, 2),
                "total_gtv": round(total_gtv, 2),
                "coupon_used_count": coupon_used_count,
                "coupon_saved": round(coupon_saved, 2),
                "events_processed": len(timeline),
                "elapsed_seconds": round(elapsed, 2),
                "wakeup_records": wakeup_records,
            }


            _append_wakeup_record(output_dir, user_id, result)

            print(
                f"  ✓ User_{user_id}: "
                f"wakeups={wakeup_count}, orders={order_count}, "
                f"coupons_used={coupon_used_count}, "
                f"spend=¥{total_spend:.2f}, gtv=¥{total_gtv:.2f}, "
                f"time={elapsed:.1f}s"
            )
            return result

        except Exception as e:
            elapsed = time.time() - start_time
            error_result = {
                "user_id": user_id,
                "agent_name": agent.display_name,
                "status": "error",
                "error": str(e),
                "elapsed_seconds": round(elapsed, 2),
                "wakeup_count": 0, "order_count": 0,
                "total_spend": 0.0, "total_gtv": 0.0, "coupon_used_count": 0,
                "coupon_saved": 0.0, "events_processed": 0,
                "wakeup_records": [],
            }
            print(f"  ✗ User_{user_id}: ERROR {e} ({elapsed:.1f}s)")
            return error_result


async def run_independent_simulation(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    concurrency: int = DEFAULT_CONCURRENCY,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
) -> Dict:
    """独立时钟模式主流程。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("仿真启动（独立时钟模式）")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  并发上限: {concurrency}")
    print(f"  输出目录: {output_dir}")
    print()

    config = load_init_config(config_path)

    print("[1/4] 初始化环境和 Agent...")
    env, controller, agents_to_run = _init_env_and_agents(config, output_path)
    agent_configs = config.get("agents", [])
    _print_env_info(env, agents_to_run, agent_configs)


    records_dir = output_path / "records"
    completed_uids = set()
    if records_dir.exists():
        for f in records_dir.iterdir():
            if f.name.startswith("user_") and f.suffix == ".json":
                try:
                    uid = int(f.stem.replace("user_", ""))
                    completed_uids.add(uid)
                except ValueError:
                    pass

    pending = [(uid, agent) for uid, agent in agents_to_run if uid not in completed_uids]
    skipped = len(agents_to_run) - len(pending)
    if skipped > 0:
        print(f"[续跑] 已完成 {skipped} 个用户，跳过；剩余 {len(pending)} 个")
        print()

    if not pending:
        print("  所有用户已完成，仿真结束。")
        results = _load_results_from_records(records_dir, completed_uids)
        stats = _collect_summary_stats(results)
        summary, summary_path, detail_path, flow_log_path = _write_outputs(
            output_path, controller, env, results, stats,
            mode_tag="independent",
            concurrency=concurrency,
            config_path=config_path,
            overall_elapsed=0.0,
            agents_to_run=agents_to_run,
            agent_configs=agent_configs,
        )
        return summary

    print(f"[3/4] 开始并发仿真（并发上限={concurrency}）...")
    overall_start = time.time()
    semaphore = asyncio.Semaphore(concurrency)

    async def _run_all():
        coros = [
            run_single_agent(agent, controller, semaphore, output_path)
            for _, agent in pending
        ]
        if tqdm_asyncio:
            return list(await tqdm_asyncio.gather(
                *coros, desc="  Agents", unit="user", total=len(coros)
            ))
        return list(await asyncio.gather(*coros))

    results_new = await _run_all()
    overall_elapsed = time.time() - overall_start
    print()


    results_cached = _load_results_from_records(records_dir, completed_uids)
    results = results_new + results_cached

    print("[4/4] 仿真完成，生成摘要...")
    stats = _collect_summary_stats(results)
    summary, summary_path, detail_path, flow_log_path = _write_outputs(
        output_path, controller, env, results, stats,
        mode_tag="independent",
        concurrency=concurrency,
        config_path=config_path,
        overall_elapsed=overall_elapsed,
        agents_to_run=agents_to_run,
        agent_configs=agent_configs,
    )
    final_status_path = await _write_env_final_status(env, output_path)
    _print_summary(
        mode_label="独立时钟（三层解耦）",
        concurrency=concurrency,
        overall_elapsed=overall_elapsed,
        agents_to_run=agents_to_run,
        stats=stats,
        env=env,
        summary=summary,
        summary_path=summary_path,
        detail_path=detail_path,
        flow_log_path=flow_log_path,
        final_status_path=final_status_path,
    )
    return summary


async def run_sync_simulation(
    config_path: str = str(DEFAULT_CONFIG_PATH),
    concurrency: int = DEFAULT_CONCURRENCY,
    output_dir: str = str(DEFAULT_OUTPUT_DIR),
    step_minutes: int = DEFAULT_STEP_MINUTES,
) -> Dict:
    """同步时钟模式主流程。"""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("仿真启动（同步时钟模式）")
    print("=" * 60)
    print(f"  配置文件: {config_path}")
    print(f"  时钟步长: {step_minutes} 分钟")
    print(f"  并发上限: {concurrency}")
    print(f"  输出目录: {output_dir}")
    print()

    config = load_init_config(config_path)

    print("[1/4] 初始化环境和 Agent...")
    env, controller, agents_to_run = _init_env_and_agents(config, output_path)
    agent_configs = config.get("agents", [])
    _print_env_info(env, agents_to_run, agent_configs)


    records_dir = output_path / "records"
    completed_uids_sync = set()
    if records_dir.exists():
        for f in records_dir.iterdir():
            if f.name.startswith("user_") and f.suffix == ".json":
                try:
                    uid = int(f.stem.replace("user_", ""))
                    completed_uids_sync.add(uid)
                except ValueError:
                    pass

    pending_sync = [(uid, agent) for uid, agent in agents_to_run if uid not in completed_uids_sync]
    skipped_sync = len(agents_to_run) - len(pending_sync)
    if skipped_sync > 0:
        print(f"[续跑] 已完成 {skipped_sync} 个用户，跳过；剩余 {len(pending_sync)} 个")
        print()

    if not pending_sync:
        print("  所有用户已完成，仿真结束。")
        results = _load_results_from_records(records_dir, completed_uids_sync)
        stats = _collect_summary_stats(results)
        summary, summary_path, detail_path, flow_log_path = _write_outputs(
            output_path, controller, env, results, stats,
            mode_tag="sync",
            concurrency=concurrency,
            config_path=config_path,
            overall_elapsed=0.0,
            agents_to_run=agents_to_run,
            agent_configs=agent_configs,
        )
        return summary


    all_agents = [a for _, a in pending_sync]
    all_wakeup_times: List[datetime] = []
    for a in all_agents:
        all_wakeup_times.extend(a.get_wakeup_sequence())

    for uid in env.active_user_ids():
        tl = env.get_user_timeline(uid)
        for ev in tl:
            all_wakeup_times.append(ev["time"])

    if not all_wakeup_times:
        print("  没有可处理的事件，仿真结束。")
        return {"total_agents": 0, "results": []}

    sim_start = min(all_wakeup_times)
    sim_end = max(all_wakeup_times)
    step = timedelta(minutes=step_minutes)
    total_ticks = int((sim_end - sim_start).total_seconds() / step.total_seconds()) + 1

    print(f"[2/4] 时间轴: {sim_start.strftime('%Y-%m-%d %H:%M')} ~ "
          f"{sim_end.strftime('%Y-%m-%d %H:%M')}")
    print(f"       共 {total_ticks} 个 tick（步长 {step_minutes} 分钟）")
    print()


    print(f"[3/4] 开始同步时钟仿真...")
    overall_start = time.time()
    semaphore = asyncio.Semaphore(concurrency)


    agent_records: Dict[int, Dict] = {
        uid: {
            "user_id": uid,
            "agent_name": agent.display_name,
            "status": "success",
            "wakeup_count": 0,
            "order_count": 0,
            "total_spend": 0.0,
            "total_gtv": 0.0,
            "coupon_used_count": 0,
            "coupon_saved": 0.0,
            "events_processed": 0,
            "elapsed_seconds": 0.0,
            "wakeup_records": [],
        }
        for uid, agent in pending_sync
    }
    agent_map: Dict[int, WakeupCouponAgent] = {uid: a for uid, a in pending_sync}

    tick_time = sim_start
    tick_num = 0
    total_wakeups_sync = 0

    try:
        from tqdm import tqdm as _tqdm
        tick_iter = _tqdm(range(total_ticks), desc="  Ticks", unit="tick")
    except ImportError:
        tick_iter = range(total_ticks)

    for _ in tick_iter:
        tick_records = await controller.run_sync_tick(
            tick_time, all_agents, semaphore, step_minutes
        )


        for record in tick_records:

            sid = record.get("session_id", "")
            try:
                uid = int(sid.split("_")[0])
            except (ValueError, IndexError):
                continue
            if uid in agent_records:
                ar = agent_records[uid]
                ar["wakeup_records"].append(record)
                ar["wakeup_count"] += 1
                total_wakeups_sync += 1
                if record.get("consumed"):
                    w = float(record.get("rollout_weight", 1.0))
                    ar["order_count"] += w
                    ar["total_spend"] += float(record.get("final_price", 0))
                    ar["total_gtv"] += float(record.get("order_final_price", record.get("final_price", 0)))
                if record.get("coupon_used"):
                    ar["coupon_used_count"] += 1
                    ar["coupon_saved"] += float(record.get("coupon_amount", 0))

        tick_time += step
        tick_num += 1

    overall_elapsed = time.time() - overall_start
    print()


    results: List[Dict] = []
    for uid, ar in agent_records.items():
        ar["total_spend"] = round(ar["total_spend"], 2)
        ar["total_gtv"] = round(ar["total_gtv"], 2)
        ar["coupon_saved"] = round(ar["coupon_saved"], 2)
        ar["events_processed"] = len(ar["wakeup_records"])
        results.append(ar)
        _append_wakeup_record(output_path, uid, ar)


    if completed_uids_sync:
        results = results + _load_results_from_records(records_dir, completed_uids_sync)

    print("[4/4] 仿真完成，生成摘要...")
    stats = _collect_summary_stats(results)
    summary, summary_path, detail_path, flow_log_path = _write_outputs(
        output_path, controller, env, results, stats,
        mode_tag="sync",
        concurrency=concurrency,
        config_path=config_path,
        overall_elapsed=overall_elapsed,
        agents_to_run=agents_to_run,
        agent_configs=agent_configs,
        step_minutes=step_minutes,
        total_ticks=tick_num,
    )
    final_status_path = await _write_env_final_status(env, output_path)
    _print_summary(
        mode_label=f"同步时钟（步长 {step_minutes} 分钟，共 {tick_num} tick）",
        concurrency=concurrency,
        overall_elapsed=overall_elapsed,
        agents_to_run=agents_to_run,
        stats=stats,
        env=env,
        summary=summary,
        summary_path=summary_path,
        detail_path=detail_path,
        flow_log_path=flow_log_path,
        final_status_path=final_status_path,
        extra_lines=[f"时钟步长: {step_minutes} 分钟", f"总 tick 数: {tick_num}"],
    )
    return summary
