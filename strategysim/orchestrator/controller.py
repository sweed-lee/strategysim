"""按访问时间推进交易与三个阶段的调用。"""
from __future__ import annotations

import json
import asyncio
from typing import Dict, List, Optional
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..agents.wakeup_coupon_agent import WakeupCouponAgent
    from ..envs.wakeup_coupon_env import WakeupCouponEnv

from .observations import checkout_observation, held_coupon_text, perception_view
from .aggregation import aggregate_wakeup_from_visits, compute_rollout_weight


class SimController:
    """仿真控制器。"""

    def __init__(
        self,
        env: WakeupCouponEnv,
        output_dir: Path,
        pos_ratio_map: Optional[Dict[int, float]] = None,
    ):
        self.env = env
        self.output_dir = output_dir
        self._pos_ratio_map = pos_ratio_map or {}
        self.sim_time: Optional[datetime] = None


        self._flow_log_path: Optional[Path] = None
        self._flow_log_file = None
        self._flow_log_lock = asyncio.Lock()


    def advance_time(self, new_time: datetime, user_id: int = None):
        """
        推进仿真时钟到 new_time，同时触发 Env 的券库存更新。
        在独立时钟模式下，由 run_single_agent 在事件时间点直接调用。

        Args:
            new_time : 目标仿真时间
            user_id  : 可选，独立时钟模式下只推进该用户的券（防止跨用户时序错乱）。
                       同步时钟模式下为 None，推进所有用户（全局 cursor）。
        """
        self.sim_time = new_time


        self.env.advance_coupon_clock(new_time, user_id=user_id)


    def check_wakeup_agents(
        self,
        sim_time: datetime,
        agents: List[WakeupCouponAgent],
        step_minutes: int = 1,
    ) -> List[WakeupCouponAgent]:
        """
        同步时钟模式专用：检查哪些 Agent 的唤醒时间序列命中当前 tick。

        命中条件：唤醒时刻落在 (sim_time - step, sim_time] 窗口内。
        这样每次步长推进后，恰好覆盖该步长内发生的全部唤醒事件。

        Args:
            sim_time     : 当前 tick 的全局仿真时间（推进后）
            agents       : 所有 Agent 列表
            step_minutes : 时钟步长（分钟），默认 1

        Returns:
            本 tick 内需要被唤醒的 Agent 列表（含对应唤醒时刻）
        """
        step_sec = step_minutes * 60
        triggered = []
        for agent in agents:
            for wt in agent.get_wakeup_sequence():

                delta = (sim_time - wt).total_seconds()
                if 0 <= delta < step_sec:
                    triggered.append((agent, wt))
                    break
        return triggered

    async def run_sync_tick(
        self,
        tick_time: datetime,
        agents: List[WakeupCouponAgent],
        semaphore: asyncio.Semaphore,
        step_minutes: int = 1,
    ) -> List[Dict]:
        """
        同步时钟模式专用：执行一个 tick 内的所有唤醒流程。

        步骤：
          1. advance_time → 更新 Env 券库存
          2. check_wakeup_agents → 找出本 tick 命中的 Agent
          3. 同 tick 内所有命中 Agent 并行执行 run_single_wakeup_flow
             （Semaphore 限制全局并发上限，相同 Agent 不会同时命中同一 tick）

        Returns:
            本 tick 内所有唤醒记录列表（可能为空）
        """
        self.advance_time(tick_time)
        triggered = self.check_wakeup_agents(tick_time, agents, step_minutes)

        if not triggered:
            return []

        async def _run_one(agent: WakeupCouponAgent, wakeup_time: datetime) -> Dict:
            async with semaphore:
                return await self.run_single_wakeup_flow(agent, wakeup_time)

        records = await asyncio.gather(
            *[_run_one(agent, wt) for agent, wt in triggered]
        )
        return list(records)


    async def run_single_wakeup_flow(
        self,
        agent: WakeupCouponAgent,
        wakeup_time: datetime,
    ) -> Dict:
        """执行一次完整的用户 App 访问流程。"""
        user_id = agent.user_id
        session_id = f"{user_id}_{wakeup_time.strftime('%Y%m%d_%H%M%S')}"
        wakeup_time_str = wakeup_time.strftime("%Y-%m-%d %H:%M:%S")
        weekday = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"][wakeup_time.weekday()]

        await self._log_flow(session_id, "session_start", {"user_id": user_id, "wakeup_time": wakeup_time_str})


        self.advance_time(wakeup_time, user_id=user_id)


        agent.reset_before_wakeup(wakeup_time_str)

        wakeup_record = {
            "session_id": session_id,
            "time": wakeup_time_str,
            "consumed": False,
            "original_price": 0.0,
            "final_price": 0.0,
            "order_final_price": 0.0,
            "coupon_used": False,
            "coupon_id": None,
            "coupon_amount": 0.0,
            "shop_name": None,
            "shop_id": None,
            "dish_name": None,
        }


        coupon_view = self.env.get_all_homepage_coupons(user_id, wakeup_time)
        coupon_count = coupon_view.get("coupon_count", 0)
        env_view_observe = perception_view(coupon_view, wakeup_time_str)
        await self._log_flow(session_id, "expose_homepage_coupons", {
            "coupon_count": coupon_count,
        })

        agent.set_current_session_id(session_id)
        await agent.observe_homepage_coupons(env_view_observe)
        await self._log_flow(session_id, "agent_observe_homepage_coupons", {"coupon_count": coupon_count})


        all_candidates = self.env.get_all_candidate_shops(user_id, wakeup_time_str)
        await self._log_flow(session_id, "get_all_candidate_shops", {
            "candidate_count": len(all_candidates),
            "candidates": [(c.get("shop_id"), c.get("label")) for c in all_candidates],
        })

        if not all_candidates:

            agent.append_behavior_record(
                f"{wakeup_time_str}（{weekday}）打开了模拟商店，浏览后无消费"
            )
            await self._log_flow(session_id, "session_end", {"reason": "no_candidates"})
            return wakeup_record


        all_visit_records: List[Dict] = []

        for candidate_idx, candidate in enumerate(all_candidates):
            label = candidate.get("label", "unknown")

            visit_session_id = f"{session_id}_visit{candidate_idx+1}_{label}"
            visit_record = await self._run_single_visit_flow(
                user_id=user_id,
                visit_session_id=visit_session_id,
                wakeup_time=wakeup_time,
                wakeup_time_str=wakeup_time_str,
                weekday=weekday,
                agent=agent,
                base_session_id=session_id,
                candidate=candidate,
            )
            visit_record["label"] = label
            all_visit_records.append(visit_record)


        wakeup_record["visits"] = all_visit_records
        wakeup_record["candidate_count"] = len(all_candidates)
        aggregate_wakeup_from_visits(wakeup_record)

        await self._log_flow(session_id, "session_end", {
            "reason": "all_visits_done",
            "visit_count": len(all_visit_records),
        })
        return wakeup_record

    async def _run_single_visit_flow(
        self,
        user_id: int,
        visit_session_id: str,
        wakeup_time: datetime,
        wakeup_time_str: str,
        weekday: str,
        agent: WakeupCouponAgent,
        base_session_id: str,
        candidate: Dict,
    ) -> Dict:
        """
        对单个候选店铺执行完整的进店→选菜→结算→订单流程。

        由 run_single_wakeup_flow 对每个候选店铺调用。

        Returns:
            visit_record dict（单次进店的完整记录，不含 label 字段，由调用方补充）
        """
        shop_id = candidate["shop_id"]
        shop_name = candidate["shop_name"]

        event_identifier = candidate.get("event_identifier", "") or ""
        visit_record = {
            "visit_session_id": visit_session_id,
            "event_identifier": event_identifier,
            "time": wakeup_time_str,
            "consumed": False,
            "original_price": 0.0,
            "final_price": 0.0,
            "order_final_price": 0.0,
            "coupon_used": False,
            "coupon_id": None,
            "coupon_amount": 0.0,
            "shop_name": shop_name,
            "shop_id": shop_id,
            "dish_name": None,
            "cart_items": [],
        }

        await self._log_flow(visit_session_id, "visit_start", {
            "user_id": user_id,
            "shop_id": shop_id,
            "shop_name": shop_name,
        })

        agent.append_behavior_record(
            f"{wakeup_time_str}（{weekday}）浏览了「{shop_name}」"
        )

        agent.set_current_session_id(visit_session_id)


        self.env.create_cart(user_id, visit_session_id, shop_id, wakeup_time=wakeup_time_str)


        shop_view = self.env.get_shop_view(shop_id)
        menu_view = self.env.get_menu_view(shop_id)
        dishes = menu_view.get("dishes", [])

        dishes_text = "\n".join(
            f"  [{i+1}] {d['dish_name']}，¥{d['price']}"
            for i, d in enumerate(dishes[:50])
        )
        if not dishes_text:
            dishes_text = "  【本店暂无菜品出售】"

        valid_dish_names = {d["dish_name"] for d in dishes[:50]}
        dish_price_map = {d["dish_name"]: float(d["price"]) for d in dishes[:50]}

        await self._log_flow(visit_session_id, "expose_shop_and_menu", {
            "shop_id": shop_id,
            "dish_count": len(dishes),
        })


        MAX_RESELECT = 3
        reselect_count = 0


        held_coupons = self.env.get_all_homepage_coupons(user_id, wakeup_time)["coupons"]
        all_held_coupons_text = held_coupon_text(held_coupons)

        while True:

            reselect_hint = (
                f"（第 {reselect_count + 1} 次选品，你刚从结算页返回，请根据短期记忆重新选购）"
                if reselect_count > 0 else ""
            )
            env_view_product = {
                "sim_time": wakeup_time_str,
                "shop_name": shop_name,
                "avg_price": shop_view.get("avg_price", 0),
                "dishes_text": dishes_text,
                "valid_dish_names": valid_dish_names,
                "reselect_hint": reselect_hint,
                "all_held_coupons_text": all_held_coupons_text,
            }

            agent.set_current_session_id(visit_session_id)
            product_decision = await agent.decide_products(env_view_product)
            await self._log_flow(visit_session_id, "agent_decide_products", {
                "action": product_decision.get("action"),
                "item_count": len(product_decision.get("items", [])),
                "reselect_round": reselect_count,
                "short_memory_preview": (product_decision.get("short_memory") or "")[:80],
            })


            action = product_decision.get("action", "add_to_cart")
            items = product_decision.get("items") or []
            if action != "add_to_cart" or not items:
                if valid_dish_names:
                    first_name = next(iter(valid_dish_names))
                    product_decision["action"] = "add_to_cart"
                    product_decision["items"] = [{
                        "dish_name": first_name,
                        "price": dish_price_map.get(first_name, 10.0),
                        "count": 1,
                    }]
                else:
                    visit_record["reject_stage"] = "empty_menu"
                    self.env.clear_session(user_id, visit_session_id)
                    await self._log_flow(visit_session_id, "visit_end", {"reason": "empty_menu"})
                    return visit_record

            items = product_decision.get("items", [])
            if not items:
                agent.append_behavior_record(
                    f"{wakeup_time_str}（{weekday}）在「{shop_name}」浏览菜单后未选择商品"
                )
                visit_record["reject_stage"] = "no_items_selected"
                self.env.clear_session(user_id, visit_session_id)
                await self._log_flow(visit_session_id, "visit_end", {"reason": "no_items_selected"})
                return visit_record


            items = self.env.lookup_dish_prices(shop_id, items)


            self.env.update_cart(user_id, visit_session_id, items, sim_time=wakeup_time_str)
            cart_view = self.env.get_cart_view(user_id, visit_session_id)

            dish_names = "、".join(it.get("dish_name", "?") for it in items[:3])
            original_price = cart_view.get("original_price", 0.0) if cart_view else 0.0
            agent.append_behavior_record(
                f"{wakeup_time_str}（{weekday}）在「{shop_name}」选择了商品：{dish_names}，"
                f"原价¥{original_price}"
                + (f"（第{reselect_count + 1}次选品）" if reselect_count > 0 else "")
            )

            visit_record["original_price"] = original_price
            visit_record["shop_name"] = shop_name
            visit_record["shop_id"] = shop_id
            visit_record["dish_name"] = dish_names
            visit_record["cart_items"] = [
                {"dish_name": it.get("dish_name", "?"), "price": it.get("price", 0), "count": it.get("count", 1)}
                for it in items
            ]


            self.env.generate_checkout_state(user_id, visit_session_id, wakeup_time)
            checkout_view = self.env.get_checkout_view(user_id, visit_session_id)

            env_view_checkout = checkout_observation(cart_view, checkout_view, shop_name, wakeup_time_str)
            usable_coupons = env_view_checkout["usable_coupons"]
            original_price_cp = env_view_checkout["original_price"]

            await self._log_flow(visit_session_id, "expose_checkout_view", {
                "original_price": original_price_cp,
                "usable_coupon_count": len(usable_coupons),
                "reselect_round": reselect_count,
            })

            agent.set_current_session_id(visit_session_id)
            checkout_decision = await agent.decide_checkout(env_view_checkout)
            await self._log_flow(visit_session_id, "agent_decide_checkout", {
                "pay": checkout_decision.get("pay"),
                "use_coupon": checkout_decision.get("use_coupon"),
                "final_price": checkout_decision.get("final_price"),
                "coupon_purchase_prob_delta": checkout_decision.get("coupon_purchase_prob_delta"),
                "no_pay_action": checkout_decision.get("no_pay_action"),
                "reselect_round": reselect_count,
                "short_memory_preview": (checkout_decision.get("short_memory") or "")[:80],
            })


            if checkout_decision.get("pay", False):
                order_result = self.env.submit_order(user_id, visit_session_id, checkout_decision)
                self.env.update_after_checkout(user_id, visit_session_id, checkout_decision)


                total_payment = float(checkout_decision.get("total_payment", original_price_cp))
                order_final_price = float(checkout_decision.get("order_final_price", original_price_cp))

                final_price = order_final_price
                use_coupon = bool(checkout_decision.get("use_coupon", False))
                coupon_id = checkout_decision.get("coupon_id")

                payment_note = ""
                if use_coupon and coupon_id:
                    coupon_amount = original_price_cp - order_final_price
                    payment_note = f"，使用优惠券¥{round(coupon_amount, 2)}"

                agent.append_behavior_record(
                    f"{wakeup_time_str}（{weekday}）完成支付：「{shop_name}」"
                    f"实付¥{total_payment}{payment_note}"
                )

                if use_coupon:
                    actual_coupon_amt = round(original_price_cp - order_final_price, 2)
                else:
                    actual_coupon_amt = 0.0

                pos_ratio = self._pos_ratio_map.get(user_id, 1.0)
                delta = checkout_decision.get("coupon_purchase_prob_delta")
                rw = compute_rollout_weight(True, delta, pos_ratio)
                rollout_weight = 0.0 if rw is None else rw

                visit_record.update({
                    "consumed": True,
                    "final_price_raw": total_payment,
                    "order_final_price_raw": order_final_price,
                    "final_price": round(total_payment * rollout_weight, 4),
                    "order_final_price": round(order_final_price * rollout_weight, 4),
                    "rollout_weight": round(rollout_weight, 6),
                    "pos_ratio": pos_ratio,
                    "coupon_purchase_prob_delta": delta,
                    "coupon_used": use_coupon,
                    "coupon_id": coupon_id,
                    "coupon_amount": actual_coupon_amt,
                })

                await self._log_flow(visit_session_id, "visit_end", {
                    "reason": "paid",
                    "order_id": order_result.get("order_id"),
                    "final_price": total_payment,
                    "order_final_price": order_final_price,
                    "reselect_rounds": reselect_count,
                })

                break

            else:

                abandon_reason = checkout_decision.get("abandon_reason", "")
                no_pay_action = checkout_decision.get("no_pay_action") or "leave_shop"

                if no_pay_action == "leave_shop":
                    agent.append_behavior_record(
                        f"{wakeup_time_str}（{weekday}）在「{shop_name}」结算页放弃支付并离店"
                        + (f"（{abandon_reason}）" if abandon_reason else "")
                    )
                    visit_record["reject_stage"] = "checkout_abandon"
                    self.env.clear_session(user_id, visit_session_id)
                    await self._log_flow(visit_session_id, "visit_end", {
                        "reason": "checkout_abandon",
                        "abandon_reason": abandon_reason,
                        "reselect_rounds": reselect_count,
                    })
                    return visit_record

                reselect_count += 1
                agent.append_behavior_record(
                    f"{wakeup_time_str}（{weekday}）在「{shop_name}」结算页未支付，返回重新选购"
                    + (f"（{abandon_reason}）" if abandon_reason else "")
                )
                await self._log_flow(visit_session_id, "checkout_reselect", {
                    "reselect_round": reselect_count,
                    "abandon_reason": abandon_reason,
                })


                if reselect_count > MAX_RESELECT:
                    agent.append_behavior_record(
                        f"{wakeup_time_str}（{weekday}）在「{shop_name}」多次重选仍未支付，最终放弃"
                    )
                    visit_record["reject_stage"] = "max_reselect_exceeded"
                    self.env.clear_session(user_id, visit_session_id)
                    await self._log_flow(visit_session_id, "visit_end", {
                        "reason": "max_reselect_exceeded",
                        "reselect_rounds": reselect_count,
                    })
                    return visit_record


        env_view_extra = {"sim_time": wakeup_time_str}
        extra_decision = await agent.decide_extra_needs(env_view_extra)
        await self._log_flow(visit_session_id, "agent_decide_extra_needs", {
            "has_extra_needs": extra_decision.get("has_extra_needs", False),
        })

        return visit_record


    async def _log_flow(self, session_id: str, event: str, data: Dict = None):
        """流式写入一条控制流程日志。"""
        async with self._flow_log_lock:
            if self._flow_log_file is None:
                if self._flow_log_path is None:
                    self._flow_log_path = self.output_dir / "flow_log.jsonl"
                self._flow_log_file = open(self._flow_log_path, "a", encoding="utf-8", buffering=65536)
            entry = {
                "ts": datetime.now().isoformat(),
                "session_id": session_id,
                "event": event,
                "data": data or {},
            }
            self._flow_log_file.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def dump_flow_log(self, path: Path):
        """刷新并关闭流程日志文件（流式写入模式下 path 参数忽略，实际写入 flow_log.jsonl）。"""
        if self._flow_log_file is not None:
            self._flow_log_file.flush()
            self._flow_log_file.close()
            self._flow_log_file = None

        actual_path = self._flow_log_path or path
        return actual_path
