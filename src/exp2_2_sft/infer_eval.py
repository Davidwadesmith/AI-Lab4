from __future__ import annotations

from src.exp2_1_prompt.metrics import calculate_keyword_coverage, calculate_repetition, calculate_text_length


def build_copywriting_eval_set() -> list[dict[str, str]]:
    return [
        {
            "id": "perfume_weibo",
            "platform": "微博",
            "product": "清新花果香调女士香水，前调是葡萄柚和梨，中调是茉莉与牡丹，尾调带有一丝麝香。",
            "keywords": "葡萄柚 梨 茉莉 牡丹 麝香",
            "max_chars": "100",
        },
        {
            "id": "perfume_xhs",
            "platform": "小红书",
            "product": "清新花果香调女士香水，前调是葡萄柚和梨，中调是茉莉与牡丹，尾调带有一丝麝香。",
            "keywords": "葡萄柚 梨 茉莉 牡丹 麝香",
            "max_chars": "150",
        },
    ]


def evaluate_copywriting(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    metrics = []
    for row in rows:
        text = row.get("generation", "")
        max_chars = int(row.get("max_chars", "150"))
        metrics.append(
            {
                "id": row.get("id", ""),
                "platform": row.get("platform", ""),
                "length": calculate_text_length(text),
                "length_compliant": calculate_text_length(text) <= max_chars,
                "keyword_coverage": calculate_keyword_coverage(text, row.get("keywords", "")),
                "repetition_3gram": calculate_repetition(text, 3),
            }
        )
    return metrics
