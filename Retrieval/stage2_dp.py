from __future__ import annotations
import time
from typing import List, Dict

import numpy as np
from tqdm import tqdm

from Retrieval.cache_loader import shot_cache, embedding_cache, core_cache
from Retrieval.embedder import CLIPEmbedder

def _print_timing(label: str, t0: float) -> None:
    """In nhãn + thời gian, format gọn đẹp."""
    dt = time.time() - t0
    print(f"[⏱️] {label:<32}: {dt:7.4f}s")

def refine_shots_with_dp(
    shot_paths: List[str],
    query: str,
    embedder: CLIPEmbedder,
) -> List[Dict]:

    # ---------- 1. Embed các sub-query ---------- #
    start_total = time.time()
    t_embed = time.time()
    sub_queries = [s.strip() for s in query.split('.') if s.strip()]
    sub_embs = [embedder.encode_text(sq).cpu().numpy() for sq in sub_queries]
    M = len(sub_embs)
    print(f"[⏱️] Embed subqueries ({M} parts): {time.time() - t_embed:.4f}s")

    results: List[Dict] = []
    t_stage = time.time()

    # ---------- 2. Lặp qua từng shot ---------- #
    for shot_path in tqdm(shot_paths, desc="DP refinement"):
        try:
            t0 = time.time()

            # === a) Parse Lxx, Vyyy, shot_id
            parts = shot_path.split('/')
            Lxx, Vyyy = parts[-3], parts[-2]
            shot_id = int(parts[-1].replace("Shot_", "").replace(".mp4", ""))

            # === b) Load frame_paths from shot_cache
            t_json = time.time()
            frame_paths = shot_cache[shot_path]["frame_paths"]
            # print(f"[⏱ JSON ] {shot_path}: {time.time()-t_json:.4f}s")

            # === c) Load NPZ vectors
            t_npz = time.time()
            key = f"{Lxx}/{Vyyy}"
            npz_entry = embedding_cache[key]
            path_to_idx = {p: i for i, p in enumerate(npz_entry["paths"])}
            vectors = npz_entry["vectors"]
            frame_embs = np.stack([vectors[path_to_idx[p]] for p in frame_paths])
            # print(f"[⏱ NPZ  ] {shot_path}: {time.time()-t_npz:.4f}s")

            # === d) Compute similarity matrix (M, F)
            t_sim = time.time()
            sims = np.vstack([frame_embs.dot(q_emb) for q_emb in sub_embs])
            # print(f"[⏱ SIM  ] {shot_path}: {time.time()-t_sim:.4f}s")

            # === e) DP alignment
            t_dp = time.time()
            M, F = sims.shape
            dp = np.full((M, F), -np.inf, dtype=float)
            dp[0, :] = sims[0, :]
            for i in range(1, M):
                prefix_max = np.maximum.accumulate(dp[i - 1, :])
                for j in range(1, F):
                    dp[i, j] = sims[i, j] + prefix_max[j - 1]
            final_score = np.max(dp[-1, :]) / M
            # print(f"[⏱ DP   ] {shot_path}: {time.time()-t_dp:.4f}s")

            # === f) Get metadata and captions from core_cache
            idx = core_cache["shot_paths"].index(shot_path)
            meta = core_cache["shot_meta"]

            results.append({
                "shot_path": shot_path,
                "frame_paths": frame_paths,
                "score": float(final_score),
                "shot_id": int(meta["shot_id"][idx]),
                "fps": float(meta["fps"][idx]),
                "start_time": float(meta["start_time"][idx]),
                "source": meta["source"][idx],
                "end_time": float(meta["end_time"][idx]),
                "blip_caption": core_cache["shot_blip"][idx],
                "llm_caption": core_cache["shot_llm"][idx],
                "tags": core_cache["shot_meta"]["tags"][idx]
            })

        except Exception as e:
            print(f"[WARN] Failed to process shot: {shot_path} — {e}")

    print(f"[✅] Stage 2 total refinement time: {time.time() - start_total:.2f}s")
    return sorted(results, key=lambda x: x["score"], reverse=True)

