from __future__ import annotations
import time
from typing import List, Dict, Union
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

from Retrieval.cache_loader import (
    faiss_index_cache,
    ocr_cache,
    core_cache,
    shot_cache
)
from Retrieval.search_utils import (
    get_index_path,
    encode_query_for_search,
)
from Retrieval.config import FAISS_DIR


def stage1_retrieve_shots(query: str,
                           embedder,
                           search_type: str,
                           top_k: int,
                           type_search: str = "clip",
                           refine_stage2: bool = True
                          ) -> Union[List[str], List[Dict]]:
    assert search_type in ["frame", "shot"], "search_type must be 'frame' or 'shot'"

    t_total = time.time()

    # === 1. Encode query
    start_time = time.time()
    start_embed = time.time()
    query_vec = encode_query_for_search(query, type_search=type_search, embedder=embedder)
    embed_time = time.time() - start_embed
    print(f"[⏱️] Embedding time: {embed_time:.4f}s")

    # ---------- 2. Search (FAISS or OCR) ---------- #
    if type_search != 'ocr':
        index_path = get_index_path(type_search, search_type, FAISS_DIR)
        # === Load FAISS index (with cache)
        start_search = time.time()
        if index_path not in faiss_index_cache:
            raise RuntimeError(f" FAISS index not found in cache: {index_path}\nDid you forget to run preload_all_faiss_indexes()?")

        index = faiss_index_cache[index_path]
        print(f"[⚡] Using preloaded FAISS index: {index_path}")

        D, I = index.search(np.expand_dims(query_vec, axis=0), top_k)
        search_time = time.time() - start_search
        print(f"[⏱️] FAISS search time: {search_time:.4f}s")
    else:
        start_search = time.time()
        tfidf_matrix = ocr_cache["matrix"]
        rel_paths    = ocr_cache["paths"]

        sims = cosine_similarity(query_vec, tfidf_matrix, dense_output=False)[0]
        top_k_idx = sims.argsort()[-top_k:][::-1]
        D = sims[top_k_idx]
        I = np.array(top_k_idx)
        D = D[None, :]
        I = I[None, :]

        print(f"[⏱️] Sparse TF-IDF search time: {time.time() - start_search:.4f}s")

    # ---------- 3. Load metadata ---------- #
    start_load = time.time()
    if search_type == "frame":
        frame2shot = core_cache["frame_meta"]["source"]
        load_time = time.time() - start_load
        print(f"[⏱️] Loading paths from cache (frame) time: {load_time:.4f}s")

        if refine_stage2:
            shot_paths = [frame2shot[idx] for idx in I[0]]
            unique_shot_paths = list(dict.fromkeys(shot_paths))
            print(f"[✅] Stage 1 total time: {time.time() - start_time:.4f}s")
            return unique_shot_paths
        else:
            results = []
            for idx, dist in zip(I[0], D[0]):
                results.append({
                    "frame_path": core_cache["frame_paths"][idx],
                    "score": float(dist),
                    "frame_number": int(core_cache["frame_meta"]["frame_number"][idx]),
                    "shot_name": core_cache["frame_meta"]["shot"][idx],
                    "shot_idx": int(core_cache["frame_meta"]["shot_idx"][idx]),
                    "source": core_cache["frame_meta"]["source"][idx],
                    "timestamp": float(core_cache["frame_meta"]["timestamp"][idx]),
                    "fps": float(core_cache["frame_meta"]["fps"][idx]),
                    "blip_caption": core_cache["frame_blip"][idx],
                    "llm_caption": core_cache["frame_llm"][idx],
                    "tags": core_cache["frame_meta"]["tags"][idx]
                })
            print(f"[✅] Stage 1 total time: {time.time() - start_time:.4f}s")
            return results

    else:  # shot
        paths = core_cache["shot_paths"]
        load_time = time.time() - start_load
        print(f"[⏱️] Loading paths from cache (shot) time: {load_time:.4f}s")

        if refine_stage2:
            shot_paths = [paths[idx] for idx in I[0]]
            print(f"[✅] Stage 1 total time: {time.time() - start_time:.4f}s")
            return shot_paths
        else:
            results = []
            for idx in I[0]:
                shot_path = paths[idx]
                results.append({
                    "shot_path": paths[idx],
                    "score": float(D[0][list(I[0]).index(idx)]),
                    "shot_id": int(core_cache["shot_meta"]["shot_id"][idx]),
                    "fps": float(core_cache["shot_meta"]["fps"][idx]),
                    "start_time": float(core_cache["shot_meta"]["start_time"][idx]),
                    "end_time": float(core_cache["shot_meta"]["end_time"][idx]),
                    "blip_caption": core_cache["shot_blip"][idx],
                    "llm_caption": core_cache["shot_llm"][idx],
                    "source": core_cache["shot_meta"]["source"][idx],
                    "frame_paths": shot_cache[shot_path]["frame_paths"],
                    "tags": core_cache["shot_meta"]["tags"][idx]
                })
                
            print(f"[✅] Stage 1 total time: {time.time() - start_time:.4f}s")
            return results
