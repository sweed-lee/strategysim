"""代表性用户 Agent 的框架入口。模型连接由运行环境配置。"""
from __future__ import annotations

import json
from typing import Any, Dict, List
from datetime import datetime
from types import SimpleNamespace
import os

for suffix in ("MODEL", "API_BASE", "API_KEY"):
    value = os.environ.get(f"AGENTSOCIETY_LLM_{suffix}")
    if value:
        os.environ.setdefault(f"AGENTSOCIETY_NANO_LLM_{suffix}", value)

from agentsociety2.agent.base import AgentBase

from .model_io import ModelIO
from .stages import StagedDecisions
from .state import UserState


class WakeupCouponAgent(StagedDecisions, UserState, ModelIO, AgentBase):
    def __init__(self, id: int, profile: Any, name: str = None, *, include_domain_knowledge: bool = True):
        super().__init__(id, profile, name)
        self.include_domain_knowledge = include_domain_knowledge


        self._short_memory: Dict = {
            "updated_at":        "",
            "mood":              "",
            "consumption_plan":  "",
            "today_summary":     "",
        }

        self._action_history: List[str] = []
        self._initial_action_history: List[str] = []

        self._wakeup_sequence: List[datetime] = []


        self._llm_io_dir = None

        self._current_session_id: str = ""

    async def step(self, tick: int, t: datetime) -> str:
        """step。"""
        return "staged mode: driven by SimController"

    async def ask(self, message: str, readonly: bool = True) -> str:
        return json.dumps({"status": "ok", "message": "WakeupCouponAgent : use SimController"}, ensure_ascii=False)

    def _get_env(self):
        if self._env is None:
            return None
        env_modules = getattr(self._env, "env_modules", [])
        for mod in env_modules:
            if type(mod).__name__ == "WakeupCouponEnv":
                return mod
        if env_modules:
            return env_modules[0]
        return None

    async def dump(self) -> dict:
        return {
            "id": self._id,
            "profile": self.get_profile(),
            "name": self._name,
            "short_memory": self._short_memory,
            "action_history": self._action_history[-100:],
            "wakeup_sequence": [t.isoformat() for t in self._wakeup_sequence],
        }

    async def load(self, dump_data: dict):
        self._id = dump_data.get("id", self._id)
        profile = dump_data.get("profile")
        if profile:
            self._profile = profile
        self._name = dump_data.get("name", self._name)
        raw_sm = dump_data.get("short_memory", {})

        if "content" in raw_sm and "mood" not in raw_sm:
            self._short_memory = {
                "updated_at":       "",
                "mood":             raw_sm.get("content", ""),
                "consumption_plan": "",
                "today_summary":    "",
            }
        else:
            self._short_memory = raw_sm if raw_sm else {
                "updated_at": "", "mood": "", "consumption_plan": "", "today_summary": "",
            }
        self._action_history = dump_data.get("action_history", [])
        ws = dump_data.get("wakeup_sequence", [])
        self._wakeup_sequence = [datetime.fromisoformat(t) for t in ws]

    @classmethod
    def mcp_description(cls) -> str:
        return "消费决策智能体"

    @property
    def user_id(self) -> int:
        return int((self.get_profile() or {}).get("user_id", self._id))

    @property
    def display_name(self) -> str:
        return self._name or f"User_{self.user_id}"

    def bind_environment(self, environment):
        """兼容 AgentSociety2 的环境路由接口。"""
        self._env = SimpleNamespace(env_modules=[environment])
