"""高保真交易环境的 AgentSociety2 入口（式 7）。"""
from __future__ import annotations

import time
import threading
from typing import ClassVar, Dict, List, Optional, Tuple
from datetime import datetime
from agentsociety2.storage import ColumnDef
from agentsociety2.env import EnvBase, tool
import pandas as pd
from .config import WakeupCouponEnvConfig
from .data import ShoppingContext
from .coupons import CouponInventory
from .catalog import MerchantCatalog
from .transactions import Transactions


class WakeupCouponEnv(ShoppingContext, CouponInventory, MerchantCatalog, Transactions, EnvBase):
    """购物优惠券消费仿真环境。"""


    _agent_state_columns: ClassVar[list] = [
        ColumnDef("user_id", "INTEGER", nullable=False, description="用户ID"),
        ColumnDef("wakeup_count", "INTEGER", nullable=False, description="累计唤醒次数"),
        ColumnDef("order_count", "INTEGER", nullable=False, description="累计下单次数"),
        ColumnDef("total_spend", "REAL", nullable=False, description="累计实付金额"),
        ColumnDef("coupon_used_count", "INTEGER", nullable=False, description="累计用券次数"),
        ColumnDef("coupon_saved", "REAL", nullable=False, description="累计优惠金额"),
    ]
    _env_state_columns: ClassVar[list] = [
        ColumnDef("sim_time", "TEXT", nullable=False, description="仿真时间字符串"),
        ColumnDef("total_wakeups", "INTEGER", nullable=False, description="累计唤醒次数"),
        ColumnDef("total_orders", "INTEGER", nullable=False, description="累计下单次数"),
        ColumnDef("total_coupon_issues", "INTEGER", nullable=False, description="累计发券次数"),
        ColumnDef("total_coupon_used", "INTEGER", nullable=False, description="累计用券次数"),
    ]

    def __init__(
        self,

        wakeup_samples_json: str = "data/visits.json",
        coupon_dataset_json: str = "data/coupons.json",
        shops_json: str = "data/shops.json",
        menus_json: str = "data/menus.json",
        user_csv: str = "data/profiles.csv",
        user_actions_json: str = "data/history.json",

        coupon_csv: str = "",
        coupon_json: str = "",
        real_shops_json: str = "",
        wakeup_json: str = "",

        simulate_start: str = "2000-01-01 00:00:00",
        simulate_end: str = "2000-01-07 23:59:59",
        max_shop_display: int = 5,
        max_shop_retry: int = 3,
        max_checkout_retry: int = 2,
        refresh_memory_on_coupon: bool = True,
        refresh_memory_on_wakeup: bool = True,
        config: "WakeupCouponEnvConfig | dict | None" = None,
    ):
        super().__init__()
        if config is not None:
            if isinstance(config, dict):
                config = WakeupCouponEnvConfig(**config)
            self._config = config
        else:
            self._config = WakeupCouponEnvConfig(
                wakeup_samples_json=wakeup_samples_json,
                coupon_dataset_json=coupon_dataset_json,
                shops_json=shops_json,
                menus_json=menus_json,
                user_csv=user_csv,
                user_actions_json=user_actions_json,

                coupon_csv=coupon_csv,
                coupon_json=coupon_json,
                real_shops_json=real_shops_json,
                wakeup_json=wakeup_json,
                simulate_start=simulate_start,
                simulate_end=simulate_end,
                max_shop_display=max_shop_display,
                max_shop_retry=max_shop_retry,
                max_checkout_retry=max_checkout_retry,
                refresh_memory_on_coupon=refresh_memory_on_coupon,
                refresh_memory_on_wakeup=refresh_memory_on_wakeup,
            )


        self._coupon_df: Optional[pd.DataFrame] = None

        self._wakeup_sequences: Dict[int, List[datetime]] = {}

        self._real_shops: Dict[int, Dict[str, str]] = {}

        self._shop_info: Dict[str, Dict] = {}

        self._shop_menus: Dict[str, Dict] = {}


        self._shop_avg_price_cache: Dict[str, float] = {}

        self._dish_price_index: Dict[str, Dict[str, float]] = {}

        self._user_actions: Dict[str, List[str]] = {}

        self._negative_shops: Dict[int, Dict[str, List[str]]] = {}


        self._coupon_inventory: Dict[int, List[Dict]] = {}


        self._user_timelines: Dict[int, List[Dict]] = {}
        self._consumed_events: Dict[int, set] = {}


        self._carts: Dict[str, Dict] = {}


        self._checkout_states: Dict[str, Dict] = {}


        self._order_records: List[Dict] = []


        self._user_stats: Dict[int, Dict] = {}
        self._total_wakeups: int = 0
        self._total_orders: int = 0
        self._total_coupon_issues: int = 0
        self._total_coupon_used: int = 0
        self._current_time: Optional[datetime] = None
        self._step_index: int = 0


        self._lock = threading.RLock()


        self._load_all_data()

    @classmethod
    def mcp_description(cls) -> str:
        return "购物交易环境"

    async def init(self, start_datetime: datetime):
        self._current_time = start_datetime
        self._step_index = 0
        print(f"[WakeupCouponEnv] 初始化完成，仿真时间: {start_datetime}")

    async def step(self, tick: int, t: datetime):
        self.t = t
        self._current_time = t
        self._step_index += 1
        await self._persist_step(t)

    async def close(self):
        pass

    def get_simulate_range(self) -> Tuple[datetime, datetime]:
        """获取仿真起止时间"""
        return (
            datetime.strptime(self._config.simulate_start, "%Y-%m-%d %H:%M:%S"),
            datetime.strptime(self._config.simulate_end, "%Y-%m-%d %H:%M:%S"),
        )

    def get_wakeup_sequence(self, user_id: int) -> List[datetime]:
        """
        获取指定用户的唤醒时间序列（属于 Agent，由外部控制脚本分发给 Agent）。

        返回已排序的 datetime 列表，不在仿真范围内的已被过滤。
        """
        return list(self._wakeup_sequences.get(user_id, []))

    def get_user_timeline(self, user_id: int) -> List[Dict]:
        """
        获取用户的完整事件时间线（独立时钟模式，供外部控制脚本遍历）。

        每项格式：
          {"type": "wakeup" | "coupon", "time": datetime, "event_idx": int, "data": {...}}
        """
        with self._lock:
            return list(self._user_timelines.get(user_id, []))

    def mark_event_consumed(self, user_id: int, event_idx: int):
        """标记事件已处理（防重复）"""
        with self._lock:
            self._consumed_events.setdefault(user_id, set()).add(event_idx)

    def is_event_consumed(self, user_id: int, event_idx: int) -> bool:
        with self._lock:
            return event_idx in self._consumed_events.get(user_id, set())

    def record_wakeup(self, user_id: int):
        """外部控制脚本触发唤醒时通知 Env 更新统计"""
        with self._lock:
            self._total_wakeups += 1
            stats = self._user_stats.setdefault(user_id, {
                "wakeup_count": 0, "order_count": 0,
                "total_spend": 0.0, "coupon_used_count": 0, "coupon_saved": 0.0,
            })
            stats["wakeup_count"] += 1

    def get_user_actions(self, user_id: int) -> List[str]:
        """获取用户历史行为列表（初始化时注入 Agent）"""
        return list(self._user_actions.get(str(user_id), []))

    @tool(readonly=True, kind="observe")
    async def get_env_config(self) -> dict:
        """获取环境配置参数（含外部控制脚本决策参数）"""
        return {
            "max_shop_display": self._config.max_shop_display,
            "max_shop_retry": self._config.max_shop_retry,
            "max_checkout_retry": self._config.max_checkout_retry,
            "refresh_memory_on_coupon": self._config.refresh_memory_on_coupon,
            "refresh_memory_on_wakeup": self._config.refresh_memory_on_wakeup,
            "simulate_start": self._config.simulate_start,
            "simulate_end": self._config.simulate_end,
        }

    @tool(readonly=True, kind="observe")
    async def get_env_status(self) -> dict:
        """获取当前仿真统计状态"""
        with self._lock:
            return {
                "sim_time": self._current_time.isoformat() if self._current_time else None,
                "step_index": self._step_index,
                "total_wakeups": self._total_wakeups,
                "total_orders": self._total_orders,
                "total_coupon_issues": self._total_coupon_issues,
                "total_coupon_used": self._total_coupon_used,
                "order_rate": round(self._total_orders / max(self._total_wakeups, 1) * 100, 2),
                "coupon_use_rate": round(self._total_coupon_used / max(self._total_coupon_issues, 1) * 100, 2),
                "active_users": len(self._user_stats),
            }

    @tool(readonly=True, kind="statistics")
    async def get_statistics(self) -> dict:
        """获取仿真统计摘要"""
        with self._lock:
            total_spend = sum(s["total_spend"] for s in self._user_stats.values())
            total_saved = sum(s["coupon_saved"] for s in self._user_stats.values())
            return {
                "sim_time": self._current_time.isoformat() if self._current_time else None,
                "total_wakeups": self._total_wakeups,
                "total_orders": self._total_orders,
                "total_users_with_orders": sum(
                    1 for s in self._user_stats.values() if s["order_count"] > 0
                ),
                "total_coupon_issues": self._total_coupon_issues,
                "total_coupon_used": self._total_coupon_used,
                "total_spend": round(total_spend, 2),
                "total_saved_by_coupons": round(total_saved, 2),
                "avg_spend_per_order": round(total_spend / max(self._total_orders, 1), 2),
                "coupon_use_rate_pct": round(
                    self._total_coupon_used / max(self._total_coupon_issues, 1) * 100, 2
                ),
            }

    @tool(readonly=True)
    async def get_order_records(self, limit: int = 100) -> dict:
        """获取最近订单记录"""
        with self._lock:
            recent = self._order_records[-limit:]
            return {"total_orders": len(self._order_records), "records": recent}

    async def _persist_step(self, t: datetime):
        step = self._step_index
        await self._write_env_state(
            step=step, t=t,
            sim_time=t.strftime("%Y-%m-%d %H:%M:%S"),
            total_wakeups=self._total_wakeups,
            total_orders=self._total_orders,
            total_coupon_issues=self._total_coupon_issues,
            total_coupon_used=self._total_coupon_used,
        )
        if self._user_stats:
            rows = []
            for uid, stats in self._user_stats.items():
                rows.append({
                    "agent_id": uid,
                    "user_id": uid,
                    "wakeup_count": stats["wakeup_count"],
                    "order_count": stats["order_count"],
                    "total_spend": stats["total_spend"],
                    "coupon_used_count": stats["coupon_used_count"],
                    "coupon_saved": stats["coupon_saved"],
                })
            if rows:
                await self._write_agent_state_batch(step=step, t=t, records=rows)

    def _dump_state(self) -> dict:
        with self._lock:
            return {
                "step_index": self._step_index,
                "current_time": self._current_time.isoformat() if self._current_time else None,
                "coupon_issue_cursor": getattr(self, "_coupon_issue_cursor", 0),
                "user_stats": {str(k): v for k, v in self._user_stats.items()},
                "total_wakeups": self._total_wakeups,
                "total_orders": self._total_orders,
                "total_coupon_issues": self._total_coupon_issues,
                "total_coupon_used": self._total_coupon_used,
                "order_records": self._order_records,
                "consumed_events": {str(k): list(v) for k, v in self._consumed_events.items()},

                "coupon_inventory": {str(k): v for k, v in self._coupon_inventory.items()},
            }

    def _load_state(self, state: dict):
        with self._lock:
            self._step_index = state.get("step_index", 0)
            ct = state.get("current_time")
            self._current_time = datetime.fromisoformat(ct) if ct else None
            self._coupon_issue_cursor = state.get("coupon_issue_cursor", 0)
            self._user_stats = {int(k): v for k, v in state.get("user_stats", {}).items()}
            self._total_wakeups = state.get("total_wakeups", 0)
            self._total_orders = state.get("total_orders", 0)
            self._total_coupon_issues = state.get("total_coupon_issues", 0)
            self._total_coupon_used = state.get("total_coupon_used", 0)
            self._order_records = state.get("order_records", [])
            consumed = state.get("consumed_events", {})
            self._consumed_events = {int(k): set(v) for k, v in consumed.items()}
            inv = state.get("coupon_inventory", {})
            self._coupon_inventory = {int(k): v for k, v in inv.items()}

    def advance_coupon_clock(self, time: datetime, user_id: Optional[int] = None):
        self._current_time = time
        self.update_coupon_inventory_by_time(time, user_id=user_id)

    def active_user_ids(self):
        return tuple(self._user_timelines)

    @property
    def total_coupon_issues(self) -> int:
        return self._total_coupon_issues
