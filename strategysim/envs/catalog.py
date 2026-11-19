"""候选店铺、菜单与商品价格查询。"""
from __future__ import annotations

from typing import Dict, List, Optional


class MerchantCatalog:
    def get_real_visited_shop(self, user_id: int, wakeup_time: str) -> Optional[Dict]:
        """
        根据用户 ID 和唤醒时刻，返回本次真实消费的店铺信息。

        使用 real_shops_json 数据（格式：{uid: {wakeup_time_str: shop_id}}）。
        若无匹配记录，返回 None（表示该次唤醒没有真实消费）。

        返回格式：
          {"shop_id": str, "shop_name": str, "cate": str, "avg_price": float}
        """
        shops_for_user = self._real_shops.get(user_id, {})
        shop_id = shops_for_user.get(wakeup_time)
        if not shop_id:
            return None
        shop_id_str = str(shop_id)
        info = self._shop_info.get(shop_id_str, {})


        avg_price = self.get_shop_view(shop_id_str).get("avg_price", 0)
        return {
            "shop_id": shop_id_str,
            "shop_name": info.get("name", f"店铺{shop_id_str}"),
            "cate": info.get("cate", ""),
            "avg_price": avg_price,
        }

    def get_all_candidate_shops(self, user_id: int, wakeup_time: str) -> List[Dict]:
        """
        获取某次唤醒的所有候选店铺（正样本 + 负样本），按「正样本优先」排列。

        返回格式：
          [
            {"shop_id": str, "shop_name": str, "cate": str, "avg_price": float,
             "label": "positive", "event_identifier": str},
            {"shop_id": str, "shop_name": str, "cate": str, "avg_price": float,
             "label": "negative", "event_identifier": str},
            ...
          ]
        正样本（真实消费店铺）始终排在第一位。
        event_identifier 来自原始 CSV 行，可直接用于回查真值。
        """
        candidates: List[Dict] = []


        positive_shop = self.get_real_visited_shop(user_id, wakeup_time)
        if positive_shop:
            pos_event_id = (self._event_id_map.get(user_id) or {}).get(wakeup_time, "")
            candidates.append({**positive_shop, "label": "positive",
                                "event_identifier": pos_event_id})


        neg_for_user = self._negative_shops.get(user_id, {})
        negative_ids = neg_for_user.get(wakeup_time, [])
        neg_event_id = (self._neg_event_id_map.get(user_id) or {}).get(wakeup_time, "")
        for neg_id in negative_ids:
            neg_id_str = str(neg_id)
            info = self._shop_info.get(neg_id_str, {})

            avg_price = self.get_shop_view(neg_id_str).get("avg_price", 0)
            candidates.append({
                "shop_id": neg_id_str,
                "shop_name": info.get("name", f"店铺{neg_id_str}"),
                "cate": info.get("cate", ""),
                "avg_price": avg_price,
                "label": "negative",
                "event_identifier": neg_event_id,
            })

        return candidates

    def get_shop_view(self, shop_id: str) -> Dict:
        """返回店铺基本信息。"""
        shop_id_str = str(shop_id)
        info = self._shop_info.get(shop_id_str, {})
        cached = self._shop_avg_price_cache.get(shop_id_str)
        if cached is not None:
            avg_price = cached
        else:
            avg_price = info.get("avg_price")
            if avg_price is None:
                dish_data = self._shop_menus.get(shop_id_str, {})
                raw_dishes = dish_data.get("dishes", [])
                valid_prices = [float(d[1]) for d in raw_dishes
                                if isinstance(d, (list, tuple)) and len(d) >= 2 and float(d[1]) > 0]
                avg_price = round(sum(valid_prices) / len(valid_prices), 2) if valid_prices else 0.0
            self._shop_avg_price_cache[shop_id_str] = avg_price
        return {
            "shop_id": shop_id_str,
            "shop_name": info.get("name", f"店铺{shop_id_str}"),
            "cate": info.get("cate", ""),
            "avg_price": avg_price,
        }

    def get_menu_view(self, shop_id: str) -> Dict:
        """
        返回店铺菜品列表（外部控制脚本曝光给 Agent）。

        过滤规则：
          - 删除价格 <= 0 的菜品
          - 至多返回前 50 道菜品

        返回格式：
          {"shop_id": str, "shop_name": str, "avg_price": float,
           "dishes": [{"dish_name": str, "price": float}, ...]}
        """
        shop_id_str = str(shop_id)
        info = self._shop_info.get(shop_id_str, {})
        dish_data = self._shop_menus.get(shop_id_str, {})
        raw_dishes = dish_data.get("dishes", [])

        dishes = []
        for d in raw_dishes:
            if isinstance(d, (list, tuple)) and len(d) >= 2:
                price = float(d[1])
                if price <= 0:
                    continue
                dishes.append({"dish_name": str(d[0]), "price": price})
            elif isinstance(d, dict):
                price = float(d.get("price", 0))
                if price <= 0:
                    continue
                dishes.append({
                    "dish_name": str(d.get("dish_name", d.get("name", "?"))),
                    "price": price,
                })

        dishes = dishes[:50]
        return {
            "shop_id": shop_id_str,
            "shop_name": info.get("name", f"店铺{shop_id_str}"),
            "cate": info.get("cate", ""),
            "avg_price": info.get("avg_price", 0),
            "dishes": dishes,
        }

    def lookup_dish_prices(self, shop_id: str, selected_items: List[Dict]) -> List[Dict]:
        """根据 LLM 选择的菜品名，从真实菜单中查找价格，返回修正后的 items 列表。"""
        shop_id_str = str(shop_id)


        menu_price_map = self._dish_price_index.get(shop_id_str)
        if menu_price_map is None:
            dish_data = self._shop_menus.get(shop_id_str, {})
            raw_dishes = dish_data.get("dishes", [])
            menu_price_map = {}
            for d in raw_dishes:
                if isinstance(d, (list, tuple)) and len(d) >= 2:
                    name = str(d[0]).strip()
                    try:
                        price = float(d[1])
                        if price > 0:
                            menu_price_map[name] = price
                    except (TypeError, ValueError):
                        pass
                elif isinstance(d, dict):
                    name = str(d.get("dish_name", d.get("name", ""))).strip()
                    try:
                        price = float(d.get("price", 0))
                        if price > 0 and name:
                            menu_price_map[name] = price
                    except (TypeError, ValueError):
                        pass
            self._dish_price_index[shop_id_str] = menu_price_map

        if not menu_price_map:

            return selected_items

        corrected = []
        for item in selected_items:
            dish_name = str(item.get("dish_name", "")).strip()
            count = int(item.get("count", 1))
            llm_price = float(item.get("price", 0))


            real_price = menu_price_map.get(dish_name)


            if real_price is None:
                for menu_name, p in menu_price_map.items():
                    if dish_name in menu_name or menu_name in dish_name:
                        real_price = p
                        break

            corrected.append({
                "dish_name": dish_name,
                "price": real_price if real_price is not None else llm_price,
                "count": count,
            })

        return corrected
