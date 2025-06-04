from typing import Dict, Any
import json
from openai import OpenAI
from Retrieval.config import OPENAI_API_KEY, PROMPT_TEMPLATE  # PROMPT_TEMPLATE contains "{full_description}" placeholder

_client = OpenAI(api_key=OPENAI_API_KEY)


def _llm(full_description: str) -> str:
    prompt: str = PROMPT_TEMPLATE.replace("{full_description}", full_description)
    resp = _client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
    )
    txt: str = resp.choices[0].message.content
    try:
        return json.loads(txt)["query_full"]
    except Exception:
        return txt  


def agentic(full_description: str) -> Dict[str, Any]:
    q: str = _llm(full_description)
    return {
        "query": q,
        "full_query": q,
        "top_k1": 500,
        "search_type": "frame",
        "search_method": "clip",
        "enable_s2": True,
        "top_m2": 20,
        "enable_s3": True,
        "top_k3": 50,
    }