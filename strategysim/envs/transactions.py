"""购物车、结算状态、订单提交和会话清理。"""
from __future__ import annotations

from typing import Dict, List, Optional
from datetime import datetime


class Transactions:
    def create_cart(self, user_id: int, session_id: str, shop_id: str, wakeup_time: str = ""):
        """
        创建购物车。外部控制脚本在确认进入店铺后调用。
        """
        with self._lock:
            shop_id_str = str(shop_id)
            info = self._shop_info.get(shop_id_str, {})
            self._carts[session_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "shop_id": shop_id_str,
                "shop_name": info.get("name", f"店铺{shop_id_str}"),
                "items": [],
                "item_total": 0.0,
                "packing_fee": 0.0,
                "delivery_fee": 0.0,
                "original_price": 0.0,
                "updated_at": "",
            }
            if wakeup_time:
                self._carts[session_id]["wakeup_time"] = wakeup_time

    def update_cart(self, user_id: int, session_id: str, selected_items: List[Dict], sim_time: str = ""):
        """
        将 Agent 的商品选择意图写入购物车（外部控制脚本调用）。

        selected_items 格式：[{"dish_name": str, "price": float, "count": int}, ...]
        或兼容旧格式：[{"dish": str, "price": float}, ...]
        """
        with self._lock:
            cart = self._carts.get(session_id)
            if cart is None:
                return
            items = []
            item_total = 0.0
            for item in selected_items:

                dish_name = item.get("dish_name") or item.get("dish", "?")
                price = float(item.get("price", 0))
                count = int(item.get("count", 1))
                items.append({"dish_name": dish_name, "price": price, "count": count})
                item_total += price * count
            cart["items"] = items
            cart["item_total"] = round(item_total, 2)
            cart["original_price"] = round(item_total + cart["packing_fee"] + cart["delivery_fee"], 2)
            cart["updated_at"] = sim_time

    def get_cart_state(self, user_id: int, session_id: str) -> Optional[Dict]:
        """获取完整购物车内部状态（外部控制脚本内部使用）"""
        with self._lock:
            return dict(self._carts[session_id]) if session_id in self._carts else None

    def get_cart_view(self, user_id: int, session_id: str) -> Optional[Dict]:
        """
        获取购物车的用户可见视图（外部控制脚本曝光给 Agent）。

        返回格式与 get_cart_state 相同，但 summary 字段为文字摘要。
        """
        with self._lock:
            cart = self._carts.get(session_id)
            if cart is None:
                return None
            view = dict(cart)
            item_desc = "、".join(
                f"{it['dish_name']}×{it['count']}(¥{it['price']})"
                for it in cart["items"]
            )
            view["summary"] = (
                f"{cart['shop_name']} | {item_desc} | 原价¥{cart['original_price']}"
                if cart["items"] else f"{cart['shop_name']} | 空购物车"
            )
            return view

    def clear_cart(self, user_id: int, session_id: str):
        """清除购物车"""
        with self._lock:
            self._carts.pop(session_id, None)

    def generate_checkout_state(self, user_id: int, session_id: str, sim_time: datetime):
        """
        根据当前购物车和用户持有券生成结算页状态（外部控制脚本调用）。

        结算页状态存储在 _checkout_states[session_id]，后续通过
        get_checkout_view 取出曝光给 Agent。
        """
        cart = self.get_cart_state(user_id, session_id)
        if cart is None:
            return
        coupon_view = self.get_checkout_coupon_view(user_id, session_id, sim_time)


        best_coupon = None
        best_price = cart["original_price"]
        for c in coupon_view["usable_coupons"]:
            price_after = max(0.0, cart["original_price"] - c["coupon_amount"])
            if price_after < best_price:
                best_price = price_after
                best_coupon = c

        with self._lock:
            self._checkout_states[session_id] = {
                "user_id": user_id,
                "session_id": session_id,
                "shop_id": cart["shop_id"],
                "shop_name": cart["shop_name"],
                "items": cart["items"],
                "item_total": cart["item_total"],
                "packing_fee": cart["packing_fee"],
                "delivery_fee": cart["delivery_fee"],
                "original_price": cart["original_price"],
                "usable_coupons": coupon_view["usable_coupons"],
                "unusable_coupons": coupon_view["unusable_coupons"],
                "best_coupon": best_coupon,
                "best_price_with_coupon": round(best_price, 2),
                "generated_at": sim_time.strftime("%Y-%m-%d %H:%M:%S"),
            }

    def get_checkout_view(self, user_id: int, session_id: str) -> Optional[Dict]:
        """
        获取结算页的用户可见视图（外部控制脚本曝光给 Agent）。
        """
        with self._lock:
            state = self._checkout_states.get(session_id)
            if state is None:
                return None
            return dict(state)

    def submit_order(self, user_id: int, session_id: str, checkout_decision: Dict) -> Dict:
        """
        提交订单（外部控制脚本调用，非 async，直接写入）。

        checkout_decision 格式（Agent 结算决策输出）：
          {
            "pay": bool,
            "use_coupon": bool,
            "coupon_id": str | None,
            "final_price": float,
            "reasoning": str,   # 可选
            ...
          }

        返回：{"success": bool, "order_id": int}
        """
        cart = self.get_cart_state(user_id, session_id)
        if cart is None:
            return {"success": False, "error": "cart not found"}

        use_coupon = bool(checkout_decision.get("use_coupon", False))
        coupon_id = checkout_decision.get("coupon_id") if use_coupon else None
        final_price = float(checkout_decision.get("final_price", cart["original_price"]))
        coupon_amount_used = 0.0


        if use_coupon and coupon_id:
            with self._lock:
                inventory = self._coupon_inventory.get(user_id, [])
                for c in inventory:
                    if str(c["id"]) == str(coupon_id):
                        coupon_amount_used = c["coupon_amount"]
                        break

        with self._lock:
            self._total_orders += 1
            stats = self._user_stats.setdefault(user_id, {
                "wakeup_count": 0, "order_count": 0,
                "total_spend": 0.0, "coupon_used_count": 0, "coupon_saved": 0.0,
            })
            stats["order_count"] += 1
            stats["total_spend"] += final_price
            if use_coupon:
                self._total_coupon_used += 1
                stats["coupon_used_count"] += 1
                stats["coupon_saved"] += coupon_amount_used

            dish_names = "、".join(it["dish_name"] for it in cart["items"])
            sim_time_str = self._current_time.strftime("%Y-%m-%d %H:%M:%S") if self._current_time else ""
            record = {
                "user_id": user_id,
                "session_id": session_id,
                "wakeup_time": cart.get("wakeup_time", ""),
                "shop_id": cart["shop_id"],
                "shop_name": cart["shop_name"],
                "dish_name": dish_names,
                "original_price": round(cart["original_price"], 2),
                "final_price": round(final_price, 2),
                "use_coupon": 1 if use_coupon else 0,
                "coupon_id": coupon_id or "",
                "coupon_amount": round(coupon_amount_used, 2),
                "cart": cart["items"],
                "item_count": len(cart["items"]),
                "sim_time": sim_time_str,
            }
            self._order_records.append(record)
            order_id = len(self._order_records)

        return {"success": True, "order_id": order_id}

    def update_after_checkout(self, user_id: int, session_id: str, checkout_decision: Dict):
        """已支付订单使用的券从可用库存中移除。"""
        if checkout_decision.get("pay") and checkout_decision.get("use_coupon"):
            coupon_id = str(checkout_decision.get("coupon_id"))
            with self._lock:
                self._coupon_inventory[user_id] = [
                    c for c in self._coupon_inventory.get(user_id, [])
                    if str(c["id"]) != coupon_id
                ]

    def clear_session(self, user_id: int, session_id: str):
        """
        清理 session（外部控制脚本在 session 结束时调用）：
          - 清除购物车
          - 清除结算页状态
        """
        with self._lock:
            self._carts.pop(session_id, None)
            self._checkout_states.pop(session_id, None)
