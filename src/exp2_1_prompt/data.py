from __future__ import annotations


def build_demo_inputs() -> list[dict[str, object]]:
    return [
        {"title": "流浪地球", "aspect": "特效 科幻 视觉", "sentiment": "正面", "intensity": 2},
        {"title": "霸王别姬", "aspect": "表演 剧情 情感", "sentiment": "负面", "intensity": 5},
        {"title": "战狼", "aspect": "动作 爱国 军事", "sentiment": "负面", "intensity": 3},
        {"title": "我不是药神", "aspect": "社会 现实 医疗", "sentiment": "正面", "intensity": 5},
        {"title": "让子弹飞", "aspect": "黑色幽默 隐喻 剧情", "sentiment": "正面", "intensity": 4},
    ]
