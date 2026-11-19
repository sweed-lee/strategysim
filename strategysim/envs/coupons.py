"""按访问时间发券、维护库存并计算优惠券可用性。"""
from __future__ import annotations

from typing import Dict
from datetime import datetime


class CouponInventory:
    def update_coupon_inventory_by_time(self, sim_time: datetime, user_id: int = None):
        """
        按当前仿真时间推进券库存：将所有 ctime <= sim_time 且尚未入库的券
        发放到对应用户的 _coupon_inventory。

        由外部控制脚本在每个时钟步开始时调用。

        Args:
            sim_time : 当前仿真时间
            user_id  : 可选，指定只推进该用户的券库存（独立时钟模式）。
                       为 None 时推进所有用户（同步时钟模式，向后兼容）。
        """
        with self._lock:
            if user_id is not None:

                self._advance_user_coupons(user_id, sim_time)
            else:

                if not hasattr(self, "_coupon_issue_cursor"):
                    self._coupon_issue_cursor = 0
                events = getattr(self, "_raw_coupon_events", [])
                while self._coupon_issue_cursor < len(events):
                    ce = events[self._coupon_issue_cursor]
                    if ce["ctime_dt"] > sim_time:
                        break
                    self._issue_coupon_to_user(ce)
                    self._coupon_issue_cursor += 1

    def _advance_user_coupons(self, user_id: int, sim_time: datetime):
        """独立时钟模式：只推进指定用户的券库存。"""
        user_events = self._user_coupon_events.get(user_id, [])
        cursor = self._user_coupon_cursor.get(user_id, 0)
        while cursor < len(user_events):
            global_idx, ce = user_events[cursor]
            if ce["ctime_dt"] > sim_time:
                break
            self._issue_coupon_to_user(ce)
            cursor += 1
        self._user_coupon_cursor[user_id] = cursor

    def _issue_coupon_to_user(self, ce: Dict):
        """将一张券发到对应用户的库存（内部方法，调用前需持有 _lock）。"""
        uid = ce["user_id"]
        inventory = self._coupon_inventory.setdefault(uid, [])
        existing_ids = {c["id"] for c in inventory}
        if ce["id"] not in existing_ids:
            coupon = {
                "id": ce["id"],
                "user_id": uid,
                "ctime": ce["ctime"],
                "expire_time": ce["expire_time"],
                "coupon_amount": ce["coupon_amount"],
                "order_amount_limit": ce["order_amount_limit"],
                "strategy_type": ce.get("strategy_type", ""),
                "used": False,
                "used_in_order": None,
            }
            inventory.append(coupon)

    def get_homepage_coupon_view(self, user_id: int, sim_time: datetime) -> Dict:
        """返回本次主页展示的优惠券。"""
        view = self.get_all_homepage_coupons(user_id, sim_time)
        view["coupons"] = view["coupons"][:3]
        view["coupon_count"] = len(view["coupons"])
        return view

    def get_all_homepage_coupons(self, user_id: int, sim_time: datetime) -> Dict:
        """查询未到期的优惠券。"""
        with self._lock:
            inventory = self._coupon_inventory.get(user_id, [])
            all_visible = []
            for c in inventory:
                try:
                    exp = datetime.strptime(c["expire_time"], "%Y-%m-%d %H:%M:%S")
                    if exp <= sim_time:
                        continue
                except (ValueError, KeyError):
                    pass
                all_visible.append({
                    "id": c["id"],
                    "coupon_amount": c["coupon_amount"],
                    "order_amount_limit": c["order_amount_limit"],
                    "expire_time": c["expire_time"],
                    "strategy_type": c.get("strategy_type", ""),
                })

            all_visible.sort(key=lambda x: x["order_amount_limit"])
            return {
                "user_id": user_id,
                "sim_time": sim_time.strftime("%Y-%m-%d %H:%M:%S"),
                "coupons": all_visible,
                "coupon_count": len(all_visible),
            }

    def get_checkout_coupon_view(self, user_id: int, session_id: str, sim_time: datetime) -> Dict:
        """
        返回结算时可用的券视图（外部控制脚本在生成结算页时调用）。

        需要结合购物车原价计算可用性。
        """
        cart = self._carts.get(session_id, {})
        original_price = cart.get("original_price", 0.0)

        with self._lock:
            inventory = self._coupon_inventory.get(user_id, [])
            usable = []
            unusable = []
            for c in inventory:
                try:
                    exp = datetime.strptime(c["expire_time"], "%Y-%m-%d %H:%M:%S")
                    if exp <= sim_time:
                        continue
                except (ValueError, KeyError):
                    pass
                view = {
                    "id": c["id"],
                    "coupon_amount": c["coupon_amount"],
                    "order_amount_limit": c["order_amount_limit"],
                    "expire_time": c["expire_time"],
                    "strategy_type": c.get("strategy_type", ""),
                    "price_after_use": max(0.0, original_price - c["coupon_amount"]),
                }
                if original_price >= c["order_amount_limit"]:
                    usable.append(view)
                else:
                    gap = c["order_amount_limit"] - original_price
                    view["gap_to_threshold"] = round(gap, 2)
                    unusable.append(view)

            usable.sort(key=lambda x: x.get("price_after_use", 9999))
            return {
                "user_id": user_id,
                "session_id": session_id,
                "original_price": original_price,
                "usable_coupons": usable,
                "unusable_coupons": unusable,
            }
