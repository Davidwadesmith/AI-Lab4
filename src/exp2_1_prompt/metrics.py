from __future__ import annotations

import re
from collections import Counter
from statistics import mean
from typing import Mapping, Sequence


def remove_thought_chain(text: str) -> str:
    start_marker = "[unused16]"
    end_marker = "[unused17]"
    if start_marker in text and end_marker in text:
        start = text.find(start_marker)
        end = text.find(end_marker, start) + len(end_marker)
        if start < end:
            return (text[:start] + text[end:]).strip()
    return text.strip()


def calculate_text_length(text: str) -> int:
    return len(re.sub(r"\s+", "", text))


def _tokenize(text: str) -> list[str]:
    try:
        import jieba

        tokens = [token.strip() for token in jieba.cut(text) if token.strip()]
    except Exception:
        tokens = [token for token in re.split(r"\s+", text) if token]
        if len(tokens) <= 1:
            tokens = [char for char in text if not char.isspace()]
    return tokens


def calculate_repetition(text: str, n: int = 3) -> float:
    tokens = _tokenize(text)
    if n <= 0 or len(tokens) < n:
        return 0.0
    ngrams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    counts = Counter(ngrams)
    return sum(count - 1 for count in counts.values()) / len(ngrams)


def calculate_keyword_coverage(text: str, keywords: str) -> float:
    keyword_list = [keyword.strip() for keyword in re.split(r"[\s,，/]+", keywords) if keyword.strip()]
    if not keyword_list:
        return 1.0
    hits = sum(1 for keyword in keyword_list if keyword in text)
    return hits / len(keyword_list)


def rough_sentiment_match(text: str, expected: str) -> float:
    positive = ["好", "出色", "动人", "惊喜", "优秀", "喜欢", "真诚", "震撼"]
    negative = ["差", "失望", "混乱", "薄弱", "尴尬", "空洞", "生硬", "遗憾"]
    pos_hits = sum(word in text for word in positive)
    neg_hits = sum(word in text for word in negative)
    if expected == "正面":
        return 1.0 if pos_hits >= neg_hits else 0.0
    if expected == "负面":
        return 1.0 if neg_hits >= pos_hits else 0.0
    return 0.5


def evaluate_generations(
    items: Sequence[Mapping[str, object]],
    generations: Sequence[str],
    *,
    min_chars: int = 300,
    max_chars: int = 450,
) -> dict[str, object]:
    rows = []
    for item, generation in zip(items, generations):
        cleaned = remove_thought_chain(generation)
        length = calculate_text_length(cleaned)
        rows.append(
            {
                "title": item["title"],
                "length": length,
                "length_compliant": min_chars <= length <= max_chars,
                "repetition_2gram": calculate_repetition(cleaned, 2),
                "repetition_3gram": calculate_repetition(cleaned, 3),
                "keyword_coverage": calculate_keyword_coverage(cleaned, str(item["aspect"])),
                "rough_sentiment_match": rough_sentiment_match(cleaned, str(item["sentiment"])),
            }
        )
    if not rows:
        return {"count": 0, "rows": []}
    return {
        "count": len(rows),
        "length_compliance_rate": mean(row["length_compliant"] for row in rows),
        "avg_repetition_2gram": mean(row["repetition_2gram"] for row in rows),
        "avg_repetition_3gram": mean(row["repetition_3gram"] for row in rows),
        "avg_keyword_coverage": mean(row["keyword_coverage"] for row in rows),
        "avg_rough_sentiment_match": mean(row["rough_sentiment_match"] for row in rows),
        "rows": rows,
    }
