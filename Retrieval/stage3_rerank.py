from __future__ import annotations
import json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict
import streamlit as st
from openai import OpenAI
from Retrieval.config import OPENAI_API_KEY, RE_RANK_PROMPT

openai_client = OpenAI(api_key=OPENAI_API_KEY)

def _print_timing(label: str, t0: float) -> None:
    dt = time.time() - t0
    print(f"[⏱️] {label:<32}: {dt:7.4f}s")


# ---------- helper ---------- #
def _format_batch(batch: List[Dict]) -> str:
    """Thu gọn batch về JSON-string (path, caption)."""
    return json.dumps(
        [
            {
                "path": item.get("shot_path") or item.get("frame_path"),
                "llm_caption": item["llm_caption"],
                "blip_caption": item["blip_caption"],
            }
            for item in batch
        ],
        ensure_ascii=False,
        indent=2,
    )


def _build_prompt(query: str, items_json: str, search_type: str) -> str:
    prompt = RE_RANK_PROMPT.replace("{query}", query)
    prompt = prompt.replace("{items_json}", items_json)
    # print(f"Re rank Prompt: {prompt}")
    return prompt


def _call_openai(query: str, batch: List[Dict], bid: int, search_type: str) -> List[Dict]:
    prompt = _build_prompt(query, _format_batch(batch), search_type)
    try:
        t0 = time.time()
        resp = openai_client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
        )
        _print_timing(f"Batch {bid} – OpenAI call", t0)

        content = resp.choices[0].message.content.strip()
        # print(f"Response: {content}")
        if not content.startswith("["):
            print(f"[] Batch {bid}: Unexpected output, skip.")
            return []
        return json.loads(content)
    except Exception as e:
        print(f" Batch {bid} failed: {e}")
        return []


def rerank_with_openai_parallel(
    query: str,
    items: List[Dict],
    *,
    top_k_rerank: int,
    max_workers: int,
    search_type: str
) -> List[Dict]:

    t_total = time.time()

    candidates = items[:top_k_rerank]
    batches = [candidates[i : i + 5] for i in range(0, len(candidates), 5)]
    print(f"[🚀] Rerank {len(candidates)} items → {len(batches)} batches")

    merged: List[Dict] = []
    t_api = time.time()
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {}

        for i, batch in enumerate(batches):
            future = ex.submit(_call_openai, query, batch, i, search_type)
            futs[future] = i

        for fut in as_completed(futs):
            merged.extend(fut.result())
    _print_timing("All OpenAI API calls (total)", t_api)

    path2orig = {it.get("shot_path") or it.get("frame_path"): it for it in items}

    t_merge = time.time()
    final: List[Dict] = []
    for entry in sorted(merged, key=lambda x: x["score"], reverse=True):
        orig = path2orig.get(entry["path"])
        if orig:
            enriched = orig.copy()
            enriched["bge_score"] = entry["score"]
            enriched["bge_explanation"] = entry.get("explanation", "")
            final.append(enriched)
    _print_timing("Merge & sort results", t_merge)

    _print_timing("Stage 3 – TOTAL", t_total)
    return final
