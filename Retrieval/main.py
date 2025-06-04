import argparse
from Retrieval import (
    preload_all_caches,
    CLIPEmbedder,
    stage1_retrieve_shots,
    refine_shots_with_dp,
    rerank_with_openai_parallel,
)
from Retrieval.config import MAX_WORKERS


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--query", required=True, type=str)
    p.add_argument("--type_search", default="clip", choices=["clip", "llm_caption", "blip_caption", "ocr"])
    p.add_argument("--search_type", default="frame", choices=["frame", "shot"])
    p.add_argument("--top_k", default=50, type=int)
    p.add_argument("--enable_dp", action="store_true")
    p.add_argument("--enable_rerank", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()

    print("🔧 Preloading caches ...")
    preload_all_caches()

    embedder = CLIPEmbedder()

    stage1 = stage1_retrieve_shots(
        query=args.query,
        embedder=embedder,
        search_type=args.search_type,
        top_k=args.top_k,
        type_search=args.type_search,
        refine_stage2=args.enable_dp,
    )

    if args.enable_dp:
        stage2 = refine_shots_with_dp(stage1, args.query, embedder)
    else:
        stage2 = stage1  # đã có metadata rồi

    if args.enable_rerank:
        final = rerank_with_openai_parallel(args.query, stage2)
    else:
        final = stage2

    print("\n=== Top-10 results ===")
    for i, item in enumerate(final[:10], 1):
        path = item.get("shot_path") or item.get("frame_path")
        print(f"{i:02d}. {path} | Stage1={item.get('score'):.4f} | BGE={item.get('bge_score', -1):.4f}")


if __name__ == "__main__":
    main()
