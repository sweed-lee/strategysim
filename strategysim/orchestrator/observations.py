"""把交易环境的结构化观测转换为各阶段可见的输入。"""


def perception_view(coupon_view: dict, sim_time: str) -> dict:
    coupons = coupon_view.get("coupons", [])

    if not coupons:
        coupon_desc = "当前账户中没有可用优惠券。"
    else:
        valid_coupons = [c for c in coupons if c['coupon_amount'] > 0]
        if not valid_coupons:
            coupon_desc = "当前账户中没有可用优惠券。"
        else:
            coupon_desc = f"当前持有 {len(valid_coupons)} 张可用优惠券，以下为门槛较低的若干张：\n" + "\n".join(
                f"  - ¥{c['coupon_amount']} 优惠券（满¥{c['order_amount_limit']}减）"
                for c in valid_coupons[:3]
            )


    return {
        "sim_time": sim_time,
        "coupon_desc": coupon_desc,
    }


def held_coupon_text(held_coupons: list) -> str:
    all_held_coupons_text = ""
    if held_coupons:
        lines = []
        for c in held_coupons:
            amt = c.get("coupon_amount", 0)
            threshold = c.get("order_amount_limit", 0)
            if threshold > 0:
                lines.append(f"  - ¥{amt} 优惠券（满¥{threshold}减）")
            else:
                lines.append(f"  - ¥{amt} 无门槛优惠券")
        all_held_coupons_text = "【账户持有券】\n" + "\n".join(lines)

    return all_held_coupons_text


def checkout_observation(cart_view: dict, checkout_view: dict, shop_name: str, sim_time: str) -> dict:
    usable_coupons = checkout_view.get("usable_coupons", []) if checkout_view else []
    unusable_coupons = checkout_view.get("unusable_coupons", []) if checkout_view else []
    original_price_cp = checkout_view.get("original_price", 0.0) if checkout_view else 0.0

    cart_items_cv = cart_view.get("items", []) if cart_view else []
    cart_text = "\n".join(
        f"  - {it.get('dish_name', '?')} ×{it.get('count', 1)}，¥{it.get('price', 0)}"
        for it in cart_items_cv
    )
    if not cart_text:
        cart_text = "  （无商品信息）"

    if usable_coupons:
        coupon_text = (f"可用优惠券（共 {len(usable_coupons)} 张，每单只能使用 **1 张**）：\n"
                      + "\n".join(
                          f"  - 优惠券ID={c.get('id','?')}，面额¥{c.get('coupon_amount','?')}"
                          f"（满¥{c.get('order_amount_limit','?')}减）"
                          f"，使用后实付¥{c.get('price_after_use', original_price_cp - c.get('coupon_amount', 0))}"
                          for c in usable_coupons
                      ))
    else:
        coupon_text = "当前订单无可用优惠券。"

    if unusable_coupons:
        unusable_text = "不可用优惠券（门槛未达）：\n" + "\n".join(
            f"  - ¥{c.get('coupon_amount','?')} 优惠券，还差¥{c.get('gap_to_threshold', '?')} 达到门槛"
            for c in unusable_coupons[:3]
        )
    else:
        unusable_text = ""

    valid_coupon_ids = [str(c.get("id")) for c in usable_coupons]

    return {
        "sim_time": sim_time,
        "shop_name": shop_name,
        "cart_text": cart_text,
        "original_price": original_price_cp,
        "coupon_text": coupon_text,
        "unusable_coupon_text": unusable_text,
        "valid_coupon_ids": valid_coupon_ids,
        "usable_coupons": usable_coupons,
    }
