from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class PromptVariant:
    name: str
    prompt: str
    factor: str


def _base_fields(item: Mapping[str, object]) -> str:
    return (
        f"电影：{item['title']}\n"
        f"评价维度：{item['aspect']}\n"
        f"情感类别：{item['sentiment']}\n"
        f"情感强度（1-5）：{item['intensity']}"
    )


def build_prompt_variants(item: Mapping[str, object]) -> list[PromptVariant]:
    fields = _base_fields(item)
    baseline = f"""你是一位资深豆瓣影评人。
任务：根据给定的结构化信息，写一条 300-450 字的中文影评，风格、情感、关键词必须与信息完全一致，信息如下：
{fields}

输出要求：
必须覆盖全部关键词，不得新增电影里没有的情节；
直接输出正文，不要标题。

影评：
"""
    structured = f"""你是一位严格执行约束的中文影评写作者。
请只依据下面的信息生成影评，不要补充演员、剧情结局、票房、导演等未给出的事实。

{fields}

硬性要求：
1. 正文 300-450 个中文字符。
2. 必须自然覆盖“评价维度”中的每个关键词。
3. 情绪必须与“情感类别”和“情感强度”一致。
4. 只输出正文，不要标题、列表、解释或“影评：”前缀。
"""
    fewshot = f"""你是一位资深豆瓣影评人，写作要像真实用户评论，避免模板腔。

示例：
电影：示例电影
评价维度：摄影 节奏 情绪
情感类别：正面
情感强度（1-5）：4
输出：这部电影最打动我的是摄影和节奏的配合，情绪推进不急不躁，让人愿意跟着角色慢慢进入故事。

现在请根据以下信息写 300-450 字中文影评：
{fields}

只输出正文。
"""
    anti_hallucination = f"""请完成受控影评生成任务。

已知信息：
{fields}

生成策略：
- 可以评价观感、节奏、表达效果和关键词对应维度。
- 不可以编造具体剧情、演员表现、导演背景、获奖信息或现实事件。
- 如果信息不足，用概括性观感表达，不要新增事实。
- 目标长度 300-450 字。
- 只输出正文。
"""
    return [
        PromptVariant("baseline", baseline, "baseline"),
        PromptVariant("structured_constraints", structured, "constraint_strength"),
        PromptVariant("fewshot_style", fewshot, "few_shot"),
        PromptVariant("anti_hallucination", anti_hallucination, "hallucination_control"),
    ]
