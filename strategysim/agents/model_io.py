"""模型调用、响应校验重试与逐次调用日志。"""
from __future__ import annotations

import json
from typing import Dict, Optional
from datetime import datetime
from json_repair import repair_json as _repair_json


class ModelIO:
    def set_current_session_id(self, session_id: str):
        """设置当前 session_id（由外部控制脚本在每次唤醒/进店时调用，用于 LLM IO 归因）"""
        self._current_session_id = session_id

    def set_llm_io_dir(self, path):
        """注入 LLM IO 日志目录（由外部控制脚本在初始化时调用）。

        Args:
            path: str 或 pathlib.Path，日志目录路径（不存在时自动创建）
        """
        from pathlib import Path as _Path
        self._llm_io_dir = _Path(path)
        self._llm_io_dir.mkdir(parents=True, exist_ok=True)

    async def _llm_call_with_retry(
        self,
        method: str,
        msgs: list,
        default: Dict,
        max_retries: int = 3,
        validator=None,
        validator_kwargs: Dict = None,
    ) -> Dict:
        """带重试机制的 LLM 调用：请求 -> 解析 JSON -> 校验，全部失败则返回默认值。

        每次尝试中，JSON 提取失败（:meth:`_extract_json` 返回 ``None``）与
        校验函数返回 ``False`` 都会触发下一轮重试，是本方法内建的错误迭代
        机制；只有 ``max_retries`` 次全部失败才会退回 ``default``，作为最终
        安全兜底而非常规路径。

        :param method: 方法名，仅用于日志与 IO 记录标识调用来源。
        :param msgs: 传给 LLM 的 messages 列表。
        :param default: 全部重试失败后返回的默认 dict。
        :param max_retries: 最大重试次数。
        :param validator: 可选校验函数，签名为 ``(result, **kwargs) -> bool``，
            返回 ``False`` 视为本次响应不合法，触发重试。
        :param validator_kwargs: 透传给 ``validator`` 的额外关键字参数。
        :returns: 解析并校验通过的 dict；全部重试失败则返回 ``default`` 的浅拷贝。
        """
        validator_kwargs = validator_kwargs or {}
        for attempt in range(max_retries):
            try:


                assert self._router is not None and self._model_name is not None, "LLM is not initialized"
                resp = await self._router.acompletion(
                    model=self._model_name,
                    messages=msgs,
                    max_tokens=16384,
                )
                try:
                    self._record_token_usage(resp)
                except (AttributeError, NotImplementedError):
                    pass
                self._write_llm_io(method, msgs, resp)

                msg_obj = resp.choices[0].message if resp.choices else None
                content = getattr(msg_obj, "content", "") or ""
                reasoning_content = getattr(msg_obj, "reasoning_content", None) or ""


                extracted_reasoning = ""
                text = content
                if reasoning_content:

                    extracted_reasoning = reasoning_content.strip()
                else:

                    think_end = content.find("</think>")
                    if think_end != -1:

                        think_start = content.find("<think>")
                        if think_start != -1:
                            extracted_reasoning = content[think_start + len("<think>"):think_end].strip()
                        text = content[think_end + len("</think>"):]

                result = self._extract_json(text)
                if result is None:
                    print(f"  [retry] {method}({attempt + 1}/{max_retries}): "
                          f"未能从响应中提取合法 JSON 对象，重试...")
                    continue

                if extracted_reasoning and (not result.get("reasoning") or result.get("reasoning") == "default"):
                    result["reasoning"] = extracted_reasoning
                if validator is not None:
                    ok = validator(result, **validator_kwargs)
                    if not ok:
                        print(f"  [retry] {method}({attempt + 1}/{max_retries}): 校验失败，重试...")
                        continue
                return result
            except Exception as e:
                print(f"  [retry] {method}({attempt + 1}/{max_retries}): 异常 {e}，重试...")
        print(f"  [retry] {method}: {max_retries} 次重试均失败，返回默认值")
        return dict(default)

    def _write_llm_io(self, method: str, messages: list, resp):
        """将 LLM 调用的 IO 写入 JSONL 日志。"""
        try:
            response_text = ""
            input_tokens = 0
            output_tokens = 0
            model = ""
            if hasattr(resp, "choices") and resp.choices:
                msg_obj = resp.choices[0].message
                content = getattr(msg_obj, "content", "") or ""
                reasoning_content = getattr(msg_obj, "reasoning_content", None) or ""

                if reasoning_content:

                    response_text = f"<think>{reasoning_content}</think>{content}"
                else:

                    response_text = content

            if hasattr(resp, "usage") and resp.usage:
                input_tokens = getattr(resp.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(resp.usage, "completion_tokens", 0) or 0
            model = getattr(resp, "model", "") or ""
            profile = self.get_profile() or {}
            entry = {
                "user_id": int(profile.get("user_id", self._id)),
                "session_id": self._current_session_id or "",
                "method": method,
                "model": model,
                "timestamp": datetime.now().isoformat(),
                "messages": messages,
                "response_text": response_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
            }
            if self._llm_io_dir is None:
                return
            fpath = self._llm_io_dir / f"{method}.jsonl"
            with open(fpath, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception:
            pass

    @staticmethod
    def _extract_json(text: str) -> Optional[Dict]:
        """从 LLM 原始响应文本中提取一个合法的 JSON 对象。"""

        try:
            parsed = json.loads(text.strip())
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass


        try:
            repaired = _repair_json(text, return_objects=True)
            if isinstance(repaired, dict):
                return repaired
        except Exception:
            pass


        start_idx = -1
        depth = 0
        for i, ch in enumerate(text):
            if ch == '{':
                if depth == 0:
                    start_idx = i
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0 and start_idx >= 0:
                    candidate = text[start_idx:i + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except json.JSONDecodeError:
                        try:
                            repaired = _repair_json(candidate, return_objects=True)
                            if isinstance(repaired, dict):
                                return repaired
                        except Exception:
                            pass
                    start_idx = -1

        return None
