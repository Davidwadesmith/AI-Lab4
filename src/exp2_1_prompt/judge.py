from __future__ import annotations

import json
import os
import urllib.request
from typing import Mapping


def judge_enabled(env: Mapping[str, str] | None = None) -> bool:
    values = env or os.environ
    return bool(values.get("JUDGE_API_BASE") and values.get("JUDGE_API_KEY") and values.get("JUDGE_MODEL"))


def build_judge_prompt(row: Mapping[str, object]) -> str:
    return f"""你是一位严格、公正的影评评分专家。
任务：对下面生成的影评进行逐项量化评分，严格遵守规则。
标注情感：{row['sentiment']}
待评影评：
{row['generation']}
输出格式（仅 JSON，不要多余字）：
{{"总分":<0-100>, "情感一致":<0/20>, "风格契合":<0/8/15>, "事实一致":<0-15>, "去AI化":<0-15>, "流畅度":<0-10>, "长度":<0/5/10>, "多样性":<0/10>, "报告":5}}
"""


def call_openai_compatible_judge(row: Mapping[str, object], env: Mapping[str, str] | None = None) -> dict[str, object]:
    values = env or os.environ
    url = values["JUDGE_API_BASE"].rstrip("/") + "/chat/completions"
    payload = {
        "model": values["JUDGE_MODEL"],
        "messages": [{"role": "user", "content": build_judge_prompt(row)}],
        "temperature": 0,
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {values['JUDGE_API_KEY']}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=120) as response:
        body = json.loads(response.read().decode("utf-8"))
    content = body["choices"][0]["message"]["content"]
    try:
        score = json.loads(content)
    except json.JSONDecodeError:
        score = {"raw": content}
    return {"title": row["title"], "prompt_variant": row["prompt_variant"], "judge": score}
