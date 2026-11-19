"""预处理后的代表性访问、商品、商家及策略数据载入。"""
from __future__ import annotations

import json
from typing import Dict, List, Tuple
from datetime import datetime, timedelta
from pathlib import Path
import pandas as pd


class ShoppingContext:
    def _load_all_data(self):
        """_load_all_data。"""
        cfg = self._config
        sim_start = datetime.strptime(cfg.simulate_start, "%Y-%m-%d %H:%M:%S")
        sim_end = datetime.strptime(cfg.simulate_end, "%Y-%m-%d %H:%M:%S")


        wakeup_path = Path(cfg.wakeup_samples_json)
        if wakeup_path.exists():
            with open(wakeup_path, encoding="utf-8") as f:
                raw = json.load(f)
            self._wakeup_sequences, self._real_shops = self._parse_wakeup_samples(raw, sim_start, sim_end)
            total_wakeups = sum(len(v) for v in self._wakeup_sequences.values())
            total_shops = sum(len(v) for v in self._real_shops.values())
            print(f"[WakeupCouponEnv] 唤醒样本: {len(self._wakeup_sequences)} 用户, "
                  f"{total_wakeups} 次唤醒, {total_shops} 条真实店铺记录")
        else:
            print(f"[WakeupCouponEnv] 警告: 唤醒样本不存在 {wakeup_path}")
            self._wakeup_sequences = {}
            self._real_shops = {}


        self._load_coupon_data(sim_start, sim_end)


        poi_path = Path(cfg.shops_json)
        if poi_path.exists():
            with open(poi_path, encoding="utf-8") as f:
                self._shop_info = json.load(f)
            print(f"[WakeupCouponEnv] 店铺信息: {len(self._shop_info)} 家")
        else:
            print(f"[WakeupCouponEnv] 警告: 店铺信息不存在 {poi_path}")


        dish_path = Path(cfg.menus_json)
        if dish_path.exists():
            with open(dish_path, encoding="utf-8") as f:
                self._shop_menus = json.load(f)
            print(f"[WakeupCouponEnv] 菜品数据: {len(self._shop_menus)} 家店")
        else:
            print(f"[WakeupCouponEnv] 警告: 菜品数据不存在 {dish_path}")


        actions_path = Path(cfg.user_actions_json)
        if actions_path.exists():
            with open(actions_path, encoding="utf-8") as f:
                self._user_actions = json.load(f)
            print(f"[WakeupCouponEnv] 历史行为: {len(self._user_actions)} 用户")
        else:
            print(f"[WakeupCouponEnv] 警告: 历史行为数据不存在 {actions_path}")


        self._build_user_timelines()

    def _load_coupon_data(self, sim_start: datetime, sim_end: datetime):
        """读取按用户分组的优惠券并建立时间索引。"""
        cfg = self._config
        coupon_path = Path(cfg.coupon_dataset_json)
        if not coupon_path.exists():
            print(f"[WakeupCouponEnv] 警告: 发券数据不存在 {coupon_path}")
            self._raw_coupon_events = []
            self._total_coupon_issues = 0
            return

        with open(coupon_path, encoding="utf-8") as f:
            raw = json.load(f)

        all_events: List[Dict] = []
        initial_coupons: Dict[int, List[Dict]] = {}

        for uid_str, coupons in raw.items():
            try:
                uid = int(uid_str)
            except (ValueError, TypeError):
                continue
            for cid, cinfo in coupons.items():
                try:
                    get_time = datetime.strptime(str(cinfo["get_time"]), "%Y-%m-%d %H:%M:%S")
                    expire_time = datetime.strptime(str(cinfo["expire_time"]), "%Y-%m-%d %H:%M:%S")
                except (KeyError, ValueError, TypeError):
                    continue

                amount = float(cinfo.get("amount", 0))
                threshold = float(cinfo.get("threshold", 0))


                if expire_time < sim_start:
                    continue

                coupon_record = {
                    "id": str(cid),
                    "user_id": uid,
                    "ctime": cinfo["get_time"],
                    "ctime_dt": get_time,
                    "expire_time": cinfo["expire_time"],
                    "coupon_amount": round(amount, 2),
                    "order_amount_limit": round(threshold, 2),
                    "strategy_type": str(cinfo.get("strategy_type", "")),

                }


                if get_time <= sim_start:
                    inventory = initial_coupons.setdefault(uid, [])
                    existing_ids = {c["id"] for c in inventory}
                    if cid not in existing_ids:
                        inventory.append({
                            **coupon_record,
                            "used": False,
                            "used_in_order": None,
                        })

                    coupon_record["_initial"] = True
                    all_events.append(coupon_record)
                elif get_time <= sim_end:

                    all_events.append(coupon_record)

        all_events.sort(key=lambda x: x["ctime_dt"])
        self._raw_coupon_events = all_events
        self._total_coupon_issues = len(all_events)


        self._user_coupon_events: Dict[int, List[Tuple[int, Dict]]] = {}
        for idx, ce in enumerate(all_events):
            uid = ce["user_id"]
            self._user_coupon_events.setdefault(uid, []).append((idx, ce))

        self._user_coupon_cursor: Dict[int, int] = {}


        for uid, coupons in initial_coupons.items():
            self._coupon_inventory[uid] = list(coupons)

        initial_total = sum(len(v) for v in initial_coupons.values())
        future_total = len(all_events) - initial_total
        print(f"[WakeupCouponEnv] 优惠券数据: {len(all_events)} 张券 "
              f"(初始持有: {initial_total}, 仿真期间发放: {future_total})")

    def _parse_wakeup_samples(
        self,
        raw: Dict[str, List[Dict]],
        sim_start: datetime,
        sim_end: datetime,
    ) -> tuple:
        """_parse_wakeup_samples。"""
        wakeup_seqs: Dict[int, List[datetime]] = {}
        real_shops: Dict[int, Dict[str, str]] = {}
        negative_shops: Dict[int, Dict[str, List[str]]] = {}


        event_id_map: Dict[int, Dict[str, str]] = {}
        neg_event_id_map: Dict[int, Dict[str, str]] = {}

        for uid_str, samples in raw.items():
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            dts = []
            shops_for_uid: Dict[str, str] = {}
            neg_for_uid: Dict[str, List[str]] = {}
            eid_for_uid: Dict[str, str] = {}
            neg_eid_for_uid: Dict[str, str] = {}
            for sample in samples:
                time_str = sample.get("time", "")
                try:
                    dt = datetime.strptime(time_str, "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue
                if not (sim_start <= dt <= sim_end):
                    continue
                dts.append(dt)
                event_id = sample.get("event_identifier", "") or ""

                positive = sample.get("positive") or {}
                shop_id = positive.get("shop_id", "")
                if shop_id:
                    shops_for_uid[time_str] = str(shop_id)
                    eid_for_uid[time_str] = str(event_id)

                negatives = sample.get("negatives", [])
                neg_ids = []
                for neg in negatives:
                    neg_shop_id = neg.get("shop_id", "")
                    if neg_shop_id:
                        neg_ids.append(str(neg_shop_id))
                if neg_ids:
                    neg_for_uid[time_str] = neg_ids
                    neg_eid_for_uid[time_str] = str(event_id)
            if dts:
                wakeup_seqs[uid] = sorted(dts)
            if shops_for_uid:
                real_shops[uid] = shops_for_uid
            if neg_for_uid:
                negative_shops[uid] = neg_for_uid
            if eid_for_uid:
                event_id_map[uid] = eid_for_uid
            if neg_eid_for_uid:
                neg_event_id_map[uid] = neg_eid_for_uid

        self._negative_shops = negative_shops
        self._event_id_map = event_id_map
        self._neg_event_id_map = neg_event_id_map
        return wakeup_seqs, real_shops

    def _parse_wakeup_sequences(
        self,
        raw: Dict[str, List[str]],
        sim_start: datetime,
        sim_end: datetime,
    ) -> Dict[int, List[datetime]]:
        """
        解析唤醒序列数据。

        期望格式：{"uid_str": ["YYYY-MM-DD HH:MM:SS", ...], ...}
        返回：{uid_int: [datetime, ...]}（在仿真时间范围内，已排序）
        """
        result: Dict[int, List[datetime]] = {}
        for uid_str, times in raw.items():
            try:
                uid = int(uid_str)
            except ValueError:
                continue
            dts = []
            for t in times:
                try:
                    dt = datetime.strptime(str(t), "%Y-%m-%d %H:%M:%S")
                    if sim_start <= dt <= sim_end:
                        dts.append(dt)
                except (ValueError, TypeError):
                    continue
            if dts:
                result[uid] = sorted(dts)
        return result

    def _build_user_timelines(self):
        """
        构建用户级事件时间线（外部控制脚本使用，用于按时间检查唤醒）。

        格式：
          _user_timelines[user_id] = [
            {"type": "wakeup", "time": datetime, "event_idx": int, "data": {...}},
            {"type": "coupon", "time": datetime, "event_idx": int, "data": {...}},
            ...
          ]
        按时间升序排列。
        """
        user_events: Dict[int, List[Dict]] = {}


        for uid, times in self._wakeup_sequences.items():
            for wt in times:
                user_events.setdefault(uid, []).append({
                    "type": "wakeup",
                    "time": wt,
                    "data": {"user_id": uid, "wakeup_time": wt.strftime("%Y-%m-%d %H:%M:%S")},
                })


        for ce in getattr(self, "_raw_coupon_events", []):
            uid = ce["user_id"]
            user_events.setdefault(uid, []).append({
                "type": "coupon",
                "time": ce["ctime_dt"],
                "data": {k: v for k, v in ce.items() if k != "ctime_dt"},
            })


        for uid, events in user_events.items():
            events.sort(key=lambda e: e["time"])
            for i, ev in enumerate(events):
                ev["event_idx"] = i
            self._user_timelines[uid] = events

        self._consumed_events = {uid: set() for uid in self._user_timelines}

        total = sum(len(e) for e in self._user_timelines.values())
        print(f"[WakeupCouponEnv] 用户事件时间线: {len(self._user_timelines)} 用户, {total} 个事件")
