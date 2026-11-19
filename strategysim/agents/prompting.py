"""按阶段将领域知识、用户状态和交易观测装配为模型输入。"""
from functools import lru_cache
from pathlib import Path
from string import Template

_RESOURCE_DIR = Path(__file__).resolve().parent
STAGE_KNOWLEDGE = {
    "incentive_perception": (
        "coupon_threshold_reference",
        "perception_habits",
    ),
    "item_selection": (
        "basket_discount_reference",
        "basket_budget",
        "menu_price_expectations",
    ),
    "checkout_evaluation": (
        "basket_discount_reference",
        "checkout_affordability",
        "checkout_coupon_use",
        "incentive_response",
    ),
}


@lru_cache(maxsize=None)
def _text(folder: str, name: str) -> str:
    return (_RESOURCE_DIR / folder / f"{name}.txt").read_text(encoding="utf-8")


def render_stage(stage: str, *, include_knowledge: bool = True, **observations) -> str:
    blocks = {
        name: _text("knowledge", name) if include_knowledge else ""
        for name in STAGE_KNOWLEDGE[stage]
    }
    template = Template(_text("prompts", stage))
    return template.substitute(**observations, **blocks)
