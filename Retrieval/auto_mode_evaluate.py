import argparse, json, csv
from Retrieval.cache_loader import preload_all_caches, shot_cache
from Retrieval.embedder import CLIPEmbedder
from Retrieval.stage1 import stage1_retrieve_shots
from Retrieval.stage2_dp import refine_shots_with_dp
from Retrieval.stage3_rerank import rerank_with_openai_parallel
from Retrieval.auto_mode import agentic

def _endswith_any(path, targets):
    return any(path.endswith(t) for t in targets)

def _rank(items, key_fn, gts):
    for idx, it in enumerate(items, 1):
        if key_fn(it, gts):
            return idx
    return -1

def _match_frame(item, gts):
    return _endswith_any(item["frame_path"], gts)

def _match_shot(item, gts):
    if "frame_paths" in item:
        return any(_endswith_any(fp, gts) for fp in item["frame_paths"])
    frames = shot_cache[item["shot_path"]]["frame_paths"]
    return any(_endswith_any(fp, gts) for fp in frames)

def evaluate(test_json, csv_out):
    preload_all_caches()
    embedder = CLIPEmbedder()
    with open(test_json, "r", encoding="utf-8") as f:
        questions = json.load(f)

    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "question_number",
            "full_description",
            "full_query",
            "rank_stage1",
            "rank_stage2",
            "rank_stage3"
        ])
        writer.writeheader()

        for q in questions:
            qnum = q.get("question_number")
            full_desc = q["query"]
            gts = q["ground_truth"]
            print(f"[⚡] Q{qnum}: {full_desc[:60]}…")
            try:
                params = agentic(full_desc)
                top_k1 = int(params.get("top_k1", 500))
                top_k3 = int(params.get("top_k3", 50))

                stage1_results = stage1_retrieve_shots(
                    query=params["query"],
                    embedder=embedder,
                    search_type="frame",
                    top_k=top_k1,
                    type_search="clip",
                    refine_stage2=False
                )
                rank1 = _rank(stage1_results, _match_frame, gts)

                shot_paths = stage1_retrieve_shots(
                    query=params["query"],
                    embedder=embedder,
                    search_type="frame",
                    top_k=top_k1,
                    type_search="clip",
                    refine_stage2=True
                )
                stage2_results = refine_shots_with_dp(shot_paths, params["full_query"], embedder)
                rank2 = _rank(stage2_results, _match_shot, gts)

                stage3_results = rerank_with_openai_parallel(
                    query=params["full_query"],
                    items=stage2_results,
                    top_k_rerank=top_k3,
                    max_workers=16,
                    search_type="frame"
                )
                rank3 = _rank(stage3_results, _match_shot, gts)

                writer.writerow({
                    "question_number": qnum,
                    "full_description": full_desc,
                    "full_query": params["full_query"],
                    "rank_stage1": rank1,
                    "rank_stage2": rank2,
                    "rank_stage3": rank3
                })
                print(f"    ✓ Saved row: Q{qnum} → ({rank1}, {rank2}, {rank3})")

            except Exception as e:
                print(f"    ✗ Error Q{qnum}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--testset", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    evaluate(args.testset, args.output)
