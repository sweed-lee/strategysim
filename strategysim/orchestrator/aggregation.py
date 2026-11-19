"""保留访问的连续订单权重与单次唤醒汇总。"""
from __future__ import annotations

from typing import Dict, Optional


def compute_rollout_weight(
    pay: bool,
    coupon_purchase_prob_delta: Optional[float],
    pos_ratio: float,
) -> Optional[float]:
    """pay=true 时 w=(1+delta)*pos_ratio；pay=false 时 None。"""
    if not pay:
        return None
    delta = 0.0 if coupon_purchase_prob_delta is None else float(coupon_purchase_prob_delta)
    delta = max(-1.0, min(1.0, delta))
    return (1.0 + delta) * pos_ratio


def aggregate_wakeup_from_visits(wakeup_record: Dict) -> None:
    """将 visits 中 rollout 加权字段汇总到 wakeup 顶层（供 O/P 汇总与 DOC 读取）。"""
    visits = wakeup_record.get("visits") or []
    paid = [v for v in visits if v.get("consumed")]
    if not paid:
        return

    first = paid[0]
    wakeup_record["consumed"] = True
    wakeup_record["rollout_weight"] = round(
        sum(float(v.get("rollout_weight", 1.0)) for v in paid), 6
    )
    wakeup_record["order_final_price"] = round(
        sum(float(v.get("order_final_price", 0.0)) for v in paid), 4
    )
    wakeup_record["final_price"] = round(
        sum(float(v.get("final_price", 0.0)) for v in paid), 4
    )
    wakeup_record["order_final_price_raw"] = round(
        sum(
            float(v.get("order_final_price_raw", v.get("order_final_price", 0.0)))
            for v in paid
        ),
        4,
    )
    wakeup_record["final_price_raw"] = round(
        sum(
            float(v.get("final_price_raw", v.get("final_price", 0.0)))
            for v in paid
        ),
        4,
    )
    wakeup_record["pos_ratio"] = first.get("pos_ratio")
    wakeup_record["coupon_purchase_prob_delta"] = first.get("coupon_purchase_prob_delta")
    wakeup_record["original_price"] = first.get("original_price", 0.0)
    wakeup_record["coupon_used"] = first.get("coupon_used", False)
    wakeup_record["coupon_id"] = first.get("coupon_id")
    wakeup_record["coupon_amount"] = first.get("coupon_amount", 0.0)
    wakeup_record["shop_name"] = first.get("shop_name")
    wakeup_record["shop_id"] = first.get("shop_id")
    wakeup_record["dish_name"] = first.get("dish_name")
