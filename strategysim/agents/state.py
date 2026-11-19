"""用户画像、心理状态和访问内历史（式 5）。"""
from __future__ import annotations

from typing import Dict, List
from datetime import datetime


class UserState:
    def get_wakeup_sequence(self) -> List[datetime]:
        """获取唤醒时间序列（属于 Agent 内部状态）。"""
        return list(self._wakeup_sequence)

    def set_wakeup_sequence(self, sequence: List[datetime]):
        """注入唤醒时间序列（由外部控制脚本初始化时调用）"""
        self._wakeup_sequence = sorted(sequence)

    def append_behavior_record(self, behavior_text: str):
        """追加一条历史行为记录（由外部控制脚本根据决策结果调用）"""
        self._action_history.append(behavior_text)
        if len(self._action_history) > 200:
            self._action_history = self._action_history[-200:]

    def load_history_actions(self, actions: List[str]):
        """批量加载历史行为（初始化时由外部控制脚本从 Env 获取后注入）"""
        self._initial_action_history = list(actions[:50])
        self._action_history = list(self._initial_action_history)

    def reset_before_wakeup(self, sim_time: str = "") -> None:
        """每次 wakeup 开始前重置短期记忆与历史行为为初始态。"""
        self._short_memory = {
            "updated_at": sim_time or "",
            "mood": "",
            "consumption_plan": "",
            "today_summary": "",
        }
        self._action_history = list(self._initial_action_history)

    def format_short_memory_before_wakeup(self, sim_time: str = "") -> None:
        """兼容旧调用名，等同 reset_before_wakeup。"""
        self.reset_before_wakeup(sim_time)

    def _profile_text(self) -> str:
        profile = self.get_profile()
        if not profile:
            return "无个人信息"
        labels = {
            "order_frequency": "历史下单频次",
            "average_spend": "历史平均实付",
            "preferences": "消费偏好",
            "discount_preference": "优惠偏好",
        }
        lines = [f"- {label}: {profile[field]}" for field, label in labels.items()
                 if profile.get(field) is not None]
        return "\n".join(lines) or "无个人信息"

    def _short_memory_text(self, sim_time: str = "") -> str:
        sm = self._short_memory
        parts = []
        updated_at = sm.get("updated_at", "")
        if updated_at:
            parts.append(f"[更新于 {updated_at}]")
        if sm.get("mood"):
            parts.append(f"当前情绪与感受：{sm['mood']}")
        if sm.get("consumption_plan"):
            parts.append(f"消费规划：{sm['consumption_plan']}")
        if sm.get("today_summary"):
            parts.append(f"本次唤醒摘要：{sm['today_summary']}")
        if not parts:
            return "（暂无短期记忆）"
        return "\n".join(parts)

    def _update_short_memory(self, sim_time: str, updates: Dict):
        """用 LLM 输出的结构化字段更新短期记忆，只覆盖非空字段。"""
        self._short_memory["updated_at"] = sim_time
        for key in ("mood", "consumption_plan", "today_summary"):
            v = updates.get(key, "")
            if v and isinstance(v, str):
                self._short_memory[key] = v.strip()

    def _recent_history_text(self, n: int = 50) -> str:
        if not self._action_history:
            return "暂无历史行为记录"
        recent = self._action_history[-n:]
        return f"最近 {len(recent)} 条历史行为：\n" + "\n".join(f"  {a}" for a in recent)
