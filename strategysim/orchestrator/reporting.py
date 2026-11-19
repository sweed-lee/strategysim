"""逐次访问记录、续跑读取、实验摘要与原项目指标适配。"""
from __future__ import annotations

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.wakeup_coupon_agent import WakeupCouponAgent
    from ..envs.wakeup_coupon_env import WakeupCouponEnv
from .controller import SimController
from .settings import DEFAULT_BEHAVIOR_JSON


def _load_results_from_records(records_dir: Path, uids: set) -> List[Dict]:
    """从 per-user 文件加载已完成的结果，用于续跑时合并。"""
    results = []
    for uid in uids:
        user_path = records_dir / f"user_{uid}.json"
        if user_path.exists():
            try:
                with open(user_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                recs = data.get("wakeup_records", [])
                results.append({
                    "user_id": uid,
                    "status": "success",
                    "wakeup_count": sum(r.get("type", "wakeup") == "wakeup" for r in recs),
                    "order_count": sum(
                        float(r.get("rollout_weight", 1.0)) for r in recs if r.get("consumed")
                    ),
                    "total_spend": sum(float(r.get("final_price", 0)) for r in recs if r.get("consumed")),
                    "total_gtv": sum(
                        float(r.get("order_final_price", r.get("final_price", 0)))
                        for r in recs if r.get("consumed")
                    ),
                    "coupon_used_count": sum(1 for r in recs if r.get("coupon_used")),
                    "coupon_saved": sum(float(r.get("coupon_amount", 0)) for r in recs if r.get("coupon_used")),
                    "events_processed": len(recs),
                    "elapsed_seconds": 0,
                    "wakeup_records": recs,
                })
            except Exception:
                continue
    return results


def _append_wakeup_record(output_dir: Path, user_id: int, result: Dict):
    """
    将单个 Agent 的 wakeup 行为记录写入独立的 per-user JSON 文件。

    改进：不再读写全局大文件（原方案每个 Agent 完成时全量读写 agent_behavior_records.json，
    导致 O(N²) 的 I/O——第 N 个 Agent 完成时要读写 N 份已有数据）。
    现改为每个 User 独立写入 records/user_{uid}.json，互不干扰，I/O 恒定 O(1)。
    """
    records_dir = output_dir / "records"
    records_dir.mkdir(exist_ok=True)
    user_path = records_dir / f"user_{user_id}.json"
    wakeup_records = result.get("wakeup_records", [])
    tmp_path = user_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump({"user_id": user_id, "wakeup_records": wakeup_records}, f, ensure_ascii=False, indent=2)
    tmp_path.replace(user_path)


def _collect_summary_stats(results: List[Dict]) -> Dict:
    """从结果列表收集汇总统计"""
    success_results = [r for r in results if r.get("status") == "success"]
    error_results = [r for r in results if r.get("status") == "error"]
    return {
        "success_results": success_results,
        "error_results": error_results,
        "total_wakeups": sum(r.get("wakeup_count", 0) for r in success_results),
        "total_orders": sum(r.get("order_count", 0) for r in success_results),
        "total_spend": sum(r.get("total_spend", 0.0) for r in success_results),

        "total_gtv": sum(r.get("total_gtv", r.get("total_spend", 0.0)) for r in success_results),
        "total_coupon_used": sum(r.get("coupon_used_count", 0) for r in success_results),
        "total_saved": sum(r.get("coupon_saved", 0.0) for r in success_results),
        "total_events_processed": sum(r.get("events_processed", 0) for r in success_results),
        "total_elapsed_agent": sum(r.get("elapsed_seconds", 0) for r in results),
    }


def _write_outputs(
    output_path: Path,
    controller: SimController,
    env: WakeupCouponEnv,
    results: List[Dict],
    stats: Dict,
    mode_tag: str,
    concurrency: int,
    config_path: str,
    overall_elapsed: float,
    agents_to_run,
    agent_configs,
    **extra_summary_fields,
) -> Tuple[Dict, Path, Path]:
    """
    写入摘要、详情、流程日志三个输出文件（同步，不含 Env 最终状态）。

    返回: (summary, summary_path, detail_path, flow_log_path)
    Env 最终状态由调用方在 async 上下文中 await 后单独写入。
    """
    s = stats
    summary = {
        "mode": mode_tag,
        "config_path": config_path,
        "concurrency": concurrency,
        "timestamp": datetime.now().isoformat(),
        "overall_elapsed_seconds": round(overall_elapsed, 2),
        "total_agent_configs": len(agent_configs),
        "active_agents_ran": len(agents_to_run),
        "successful_agents": len(s["success_results"]),
        "failed_agents": len(s["error_results"]),
        "total_wakeups": s["total_wakeups"],
        "total_orders": s["total_orders"],
        "total_spend": round(s["total_spend"], 2),

        "total_gtv": round(s["total_gtv"], 2),
        "total_coupon_used": s["total_coupon_used"],
        "total_saved_by_coupons": round(s["total_saved"], 2),
        "total_events_processed": s["total_events_processed"],
        "order_rate_pct": round(s["total_orders"] / max(s["total_wakeups"], 1) * 100, 2),
        "coupon_use_rate_pct": round(
            s["total_coupon_used"] / max(env.total_coupon_issues, 1) * 100, 2
        ),
        "avg_spend_per_order": round(s["total_spend"] / max(s["total_orders"], 1), 2),
        "avg_gtv_per_order": round(s["total_gtv"] / max(s["total_orders"], 1), 2),
        "total_agent_cpu_time": round(s["total_elapsed_agent"], 2),
        "effective_parallelism": round(s["total_elapsed_agent"] / max(overall_elapsed, 1), 2),
        **extra_summary_fields,
    }

    summary_path = output_path / f"{mode_tag}_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    detail_path = output_path / f"{mode_tag}_results.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    flow_log_path = controller.dump_flow_log(output_path / "flow_log.jsonl")

    return summary, summary_path, detail_path, flow_log_path


async def _write_env_final_status(env: WakeupCouponEnv, output_path: Path) -> Optional[Path]:
    """在 async 上下文中 await Env 最终状态并写入文件，返回文件路径（失败返回 None）。"""
    try:
        final_status = await env.get_statistics()
        final_status_path = output_path / "env_final_status.json"
        with open(final_status_path, "w", encoding="utf-8") as f:
            json.dump(final_status, f, ensure_ascii=False, indent=2)
        return final_status_path
    except Exception as e:
        print(f"  （Env 最终状态保存失败: {e}）")
        return None


def _print_summary(
    mode_label: str,
    concurrency: int,
    overall_elapsed: float,
    agents_to_run,
    stats: Dict,
    env: WakeupCouponEnv,
    summary: Dict,
    summary_path: Path,
    detail_path: Path,
    flow_log_path: Path,
    final_status_path: Optional[Path],
    extra_lines: List[str] = None,
):
    s = stats
    print()
    print("=" * 60)
    print("仿真结果摘要")
    print("=" * 60)
    print(f"  运行模式:       {mode_label}")
    print(f"  并发上限:       {concurrency}")
    print(f"  总耗时:         {overall_elapsed:.1f}s")
    print(f"  有效并行度:     {summary['effective_parallelism']:.1f}x")
    if extra_lines:
        for line in extra_lines:
            print(f"  {line}")
    print()
    print(f"  活跃用户:       {len(agents_to_run)}")
    print(f"  成功/失败:      {len(s['success_results'])}/{len(s['error_results'])}")
    print(f"  总事件数:       {s['total_events_processed']}")
    print()
    print(f"  唤醒次数:       {s['total_wakeups']}")
    print(f"  下单次数:       {s['total_orders']}")
    print(f"  下单率:         {summary['order_rate_pct']}%")
    print(f"  总实付金额:     ¥{s['total_spend']:.2f}")
    print(f"  均单实付:       ¥{summary['avg_spend_per_order']:.2f}")
    print(f"  总商品实付:          ¥{s['total_gtv']:.2f}")
    print(f"  均单商品实付:        ¥{summary['avg_gtv_per_order']:.2f}")
    print()
    print(f"  发券总数:       {env.total_coupon_issues}")
    print(f"  用券次数:       {s['total_coupon_used']}")
    print(f"  用券率:         {summary['coupon_use_rate_pct']}%")
    print(f"  优惠总额:       ¥{s['total_saved']:.2f}")
    print()
    print(f"  结果文件:")
    print(f"    摘要:     {summary_path}")
    print(f"    详情:     {detail_path}")
    print(f"    流程日志: {flow_log_path}")
    if final_status_path:
        print(f"    Env状态:  {final_status_path}")
    print("=" * 60)
