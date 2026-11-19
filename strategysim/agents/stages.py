"""激励感知、商品选择与结算评估的决策接口（式 6）。"""
from __future__ import annotations

from typing import Dict, List
from .prompting import render_stage


class StagedDecisions:
    async def observe_homepage_coupons(self, env_view: Dict) -> None:
        """
        看到主页持有券后更新短期记忆。

        不做复杂感知建模，只是让 Agent 知道"我现在持有哪些券"，
        并将这一信息写入短期记忆，供后续决策使用。

        Args:
            env_view:
              - sim_time: str    （当前仿真时间）
              - coupon_desc: str （持有券描述文本）
        """
        sim_time = env_view.get("sim_time", "")
        coupon_desc = env_view.get("coupon_desc", "当前账户中没有可用优惠券。")
        has_coupons = bool(coupon_desc and "没有可用优惠券" not in coupon_desc)

        coupon_section = (
            f"\n【主页为您推送了一部分你的可用券】\n{coupon_desc}\n"
            if has_coupons else ""
        )

        prompt = render_stage(
            "incentive_perception",
            include_knowledge=self.include_domain_knowledge,
            sim_time=sim_time,
            profile=self._profile_text(),
            psychological_state=self._short_memory_text(sim_time),
            coupon_section=coupon_section,
        )


        msgs = [
            {"role": "system", "content": "你扮演模拟商店的顾客，正在使用模拟商店。输出严格JSON格式。"},
            {"role": "user", "content": prompt},
        ]
        default = {
            "reasoning": "default",
            "mood": f"在 {sim_time} 打开主页浏览",
            "consumption_plan": "打算浏览商品，再决定是否购买或使用优惠券",
            "today_summary": "",
        }
        result = await self._llm_call_with_retry(
            "observe_homepage_coupons", msgs, default, max_retries=2,
        )
        self._update_short_memory(sim_time, result)

    async def decide_products(self, env_view: Dict) -> Dict:
        """
        商品选择函数：模拟用户进入店铺后选择商品。

        Args:
            env_view: 环境侧信息，只含 Env 侧字段：
              - sim_time: str        （当前仿真时间）
              - shop_name: str       （店铺名）
              - avg_price: float     （店铺均价）
              - dishes_text: str     （菜品列表文本，由外部脚本从 menu_view 格式化）
              - valid_dish_names: set（合法菜品名集合，用于校验）

        Returns:
            {
              "action": "add_to_cart",
              "items": [{"dish_name": str, "price": float, "count": int}, ...],
              "short_memory": str,               # ← 与决策同步输出
              "reasoning": str,
            }
        """
        sim_time = env_view.get("sim_time", "")
        shop_name = env_view.get("shop_name", "?")
        avg_price = env_view.get("avg_price", 0)
        dishes_text = env_view.get("dishes_text", "  （暂无菜品信息）")
        all_held_coupons_text = env_view.get("all_held_coupons_text", "")


        coupon_section = (
            f"\n{all_held_coupons_text}\n"
            if all_held_coupons_text else ""
        )

        prompt = render_stage(
            "item_selection",
            include_knowledge=self.include_domain_knowledge,
            sim_time=sim_time,
            profile=self._profile_text(),
            psychological_state=self._short_memory_text(sim_time),
            history=self._recent_history_text(20),
            shop_name=shop_name,
            avg_price=avg_price,
            dishes_text=dishes_text,
            coupon_section=coupon_section,
        )

        msgs = [
            {"role": "system", "content": "你扮演模拟商店的顾客，正在店铺内浏览菜单。输出严格JSON格式。"},
            {"role": "user", "content": prompt},
        ]

        valid_dish_names = env_view.get("valid_dish_names", set())


        fallback_items = []
        if valid_dish_names:
            first_name = next(iter(valid_dish_names))
            fallback_items = [{"dish_name": first_name, "price": 10.0, "count": 1}]
        default = {
            "action": "add_to_cart",
            "items": fallback_items,
            "mood": f"在「{shop_name}」浏览菜单",
            "consumption_plan": "LLM 决策异常后的安全兜底选品",
            "today_summary": "",
            "reasoning": "default",
        }
        result = await self._llm_call_with_retry(
            "decide_products", msgs, default, max_retries=3,
            validator=self._validate_product_decision,
            validator_kwargs={"valid_dish_names": valid_dish_names},
        )

        self._update_short_memory(sim_time, result)
        return result

    def _validate_product_decision(self, result: Dict, valid_dish_names: set = None) -> bool:
        """校验商品选择决策

        校验规则：
          1. action 必须是 add_to_cart
          2. items 非空
          3. 每个 item 的 dish_name 非空字符串
          4. dish_name 必须在菜单中（若提供了 valid_dish_names）
          5. price > 0
        """
        action = result.get("action", "")
        if action != "add_to_cart":
            result["action"] = "add_to_cart"
        items = result.get("items", [])
        if not isinstance(items, list) or len(items) == 0:
            return False
        for item in items:
            dish_name = item.get("dish_name", "")
            if not dish_name or not isinstance(dish_name, str):
                return False

            if valid_dish_names and dish_name not in valid_dish_names:

                fuzzy_match = any(
                    dish_name in menu_name or menu_name in dish_name
                    for menu_name in valid_dish_names
                )
                if not fuzzy_match:
                    return False
            try:
                p = float(item.get("price", 0))
                if p <= 0:
                    return False
            except (TypeError, ValueError):
                return False
        return True

    async def decide_extra_needs(self, env_view: Dict) -> Dict:
        """额外需求决策函数。"""


        return {"has_extra_needs": False}

    async def decide_checkout(self, env_view: Dict) -> Dict:
        """结算选择函数。"""
        sim_time = env_view.get("sim_time", "")
        shop_name = env_view.get("shop_name", "?")
        original_price = env_view.get("original_price", 0.0)
        cart_text = env_view.get("cart_text", "  （无商品信息）")
        coupon_text = env_view.get("coupon_text", "当前订单无可用优惠券。")
        unusable_coupon_text = env_view.get("unusable_coupon_text", "")

        prompt = render_stage(
            "checkout_evaluation",
            include_knowledge=self.include_domain_knowledge,
            sim_time=sim_time,
            profile=self._profile_text(),
            psychological_state=self._short_memory_text(sim_time),
            history=self._recent_history_text(20),
            shop_name=shop_name,
            cart_text=cart_text,
            original_price=original_price,
            coupon_text=coupon_text,
            unusable_coupon_text=unusable_coupon_text,
        )


        msgs = [
            {"role": "system", "content": "你扮演模拟商店的顾客，正在结算订单。输出严格JSON格式。"},
            {"role": "user", "content": prompt},
        ]

        valid_coupon_ids = env_view.get("valid_coupon_ids", [])
        usable_coupons = env_view.get("usable_coupons", [])

        default = {
            "pay": True,
            "use_coupon": False,
            "coupon_id": None,
            "order_final_price": original_price,
            "total_payment": original_price,
            "final_price": original_price,
            "coupon_purchase_prob_delta": 0.0,
            "no_pay_action": None,
            "abandon_reason": "",
            "mood": f"在「{shop_name}」完成了结算决策",
            "consumption_plan": f"实付《{shop_name}》订单",
            "today_summary": "",
            "reasoning": "default",
        }
        result = await self._llm_call_with_retry(
            "decide_checkout", msgs, default, max_retries=3,
            validator=self._validate_checkout_decision,
            validator_kwargs={
                "original_price": original_price,
                "valid_coupon_ids": valid_coupon_ids,
                "usable_coupons": usable_coupons,
            },
        )

        self._update_short_memory(sim_time, result)
        return result

    def _validate_checkout_decision(
        self,
        result: Dict,
        original_price: float = 0.0,
        valid_coupon_ids: List[str] = None,
        usable_coupons: List[Dict] = None,
    ) -> bool:
        """校验结算决策，并修正不合理字段。"""
        valid_coupon_ids = valid_coupon_ids or []
        usable_coupons = usable_coupons or []


        if not isinstance(result.get("pay"), bool):
            try:
                result["pay"] = bool(result.get("pay", True))
            except Exception:
                return False

        pay = bool(result["pay"])
        use_coupon = bool(result.get("use_coupon", False))
        if pay:
            delta_raw = result.get("coupon_purchase_prob_delta")
            if delta_raw is None:
                result["coupon_purchase_prob_delta"] = 0.0
            else:
                try:
                    delta = float(delta_raw)
                except (TypeError, ValueError):
                    return False
                if not -1.0 <= delta <= 1.0:
                    return False
                result["coupon_purchase_prob_delta"] = round(delta, 4)
            result["no_pay_action"] = None
        else:
            result["coupon_purchase_prob_delta"] = None
            npa = result.get("no_pay_action") or "leave_shop"
            if npa not in ("reselect", "leave_shop"):
                npa = "leave_shop"
            result["no_pay_action"] = npa


        if "order_final_price" in result or "total_payment" in result:

            try:
                ofp = float(result.get("order_final_price", original_price))
                if ofp < 0 or ofp > original_price * 2:
                    return False
                result["order_final_price"] = round(ofp, 2)
            except (TypeError, ValueError):
                return False
            try:
                tp = float(result.get("total_payment", original_price))
                if tp < 0 or tp > original_price * 3:
                    return False
                result["total_payment"] = round(tp, 2)
            except (TypeError, ValueError):
                return False

            result["final_price"] = result["order_final_price"]
        else:

            try:
                fp = float(result.get("final_price", original_price))
                if fp < 0 or fp > original_price * 2:
                    return False
                result["final_price"] = fp

                if "order_final_price" not in result:
                    result["order_final_price"] = fp
                if "total_payment" not in result:
                    result["total_payment"] = fp
            except (TypeError, ValueError):
                return False

        if not use_coupon:
            result["coupon_id"] = None
            if result.get("pay"):
                result["order_final_price"] = original_price
                result["total_payment"] = original_price
                result["final_price"] = original_price
            return True


        cid = str(result.get("coupon_id", ""))
        if cid not in valid_coupon_ids:
            if usable_coupons:
                best = max(usable_coupons, key=lambda c: c.get("coupon_amount", 0))
                result["coupon_id"] = str(best["id"])
                ofp = max(0.0, original_price - best["coupon_amount"])
                result["order_final_price"] = round(ofp, 2)
                result["total_payment"] = round(ofp, 2)
                result["final_price"] = ofp
            else:
                result["use_coupon"] = False
                result["coupon_id"] = None
                result["order_final_price"] = original_price
                result["total_payment"] = original_price
                result["final_price"] = original_price
        else:
            for c in usable_coupons:
                if str(c.get("id")) == cid:
                    if original_price < c.get("order_amount_limit", 0):
                        result["use_coupon"] = False
                        result["coupon_id"] = None
                        result["order_final_price"] = original_price
                        result["total_payment"] = original_price
                        result["final_price"] = original_price
                    else:
                        ofp = max(0.0, original_price - c["coupon_amount"])
                        result["order_final_price"] = round(ofp, 2)
                        result["total_payment"] = round(ofp, 2)
                        result["final_price"] = ofp
                    break

        return True
