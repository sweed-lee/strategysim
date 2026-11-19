"""交易环境配置及原有初始化字段。"""
from __future__ import annotations

from pydantic import BaseModel, Field


class WakeupCouponEnvConfig(BaseModel):
    """唤醒驱动优惠券仿真环境配置。"""


    wakeup_samples_json: str = Field(
        default="data/visits.json",
        description="唤醒样本 JSON（含正/负样本店铺，合并原 wakeup + real_shops）"
    )
    coupon_dataset_json: str = Field(
        default="data/coupons.json",
        description="发券数据 JSON（按用户分组：{uid: {cid: {get_time, expire_time, amount, threshold}}}）"
    )
    shops_json: str = Field(
        default="data/shops.json",
        description="店铺基本信息 JSON {shop_id: {name, cate}}"
    )
    menus_json: str = Field(
        default="data/menus.json",
        description="店铺菜品 JSON {shop_id: {shop_name, dishes: [[name, price], ...]}}"
    )
    user_csv: str = Field(
        default="data/profiles.csv",
        description="用户画像 CSV"
    )
    user_actions_json: str = Field(
        default="data/history.json",
        description="用户历史行为 JSON {uid: [action_str, ...]}"
    )

    coupon_csv: str = Field(default="", description="(废弃) 旧版发券 CSV")
    coupon_json: str = Field(default="", description="(废弃) 旧版发券 JSON")
    real_shops_json: str = Field(default="", description="(废弃) 已合并入 wakeup_samples_json")
    wakeup_json: str = Field(default="", description="(废弃) 已合并入 wakeup_samples_json")

    simulate_start: str = Field(
        default="2000-01-01 00:00:00",
        description="仿真开始时间"
    )
    simulate_end: str = Field(
        default="2000-01-07 23:59:59",
        description="仿真结束时间"
    )

    max_shop_display: int = Field(
        default=5,
        description="（外部控制脚本参数）每次唤醒可查看的最大店铺数量"
    )
    max_shop_retry: int = Field(
        default=3,
        description="（外部控制脚本参数）单次唤醒最多重选店铺次数"
    )
    max_checkout_retry: int = Field(
        default=2,
        description="（外部控制脚本参数）单次唤醒最多返回结算重选次数"
    )
    refresh_memory_on_coupon: bool = Field(
        default=True,
        description="（外部控制脚本参数）收到券时是否触发短时记忆刷新"
    )
    refresh_memory_on_wakeup: bool = Field(
        default=True,
        description="（外部控制脚本参数）唤醒时是否触发短时记忆刷新"
    )
