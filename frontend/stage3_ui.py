# frontend/stage3_ui.py

import streamlit as st
import time
from PIL import Image
from Retrieval.stage3_rerank import rerank_with_openai_parallel
from frontend.utils import (
    _load_images_batch,
    _print_timing,
    _load_image_from_path,
    _get_display_path,
)

def render_stage3_block(
    items: list[dict],
    query: str,
    top_k_rerank: int,
    max_workers: int,
    search_type: str,
):
    st.markdown("### 🔄 Stage 3 – GPT-4o-mini Re-rank")

    cache_key = f"stage3::{query}::{top_k_rerank}"
    time_key = f"{cache_key}::time"
    if cache_key not in st.session_state:
        t0 = time.time()
        with st.spinner("🚀 Re-ranking with GPT-4o-mini..."):
            reranked = rerank_with_openai_parallel(
                query=query,
                items=items,
                top_k_rerank=top_k_rerank,
                max_workers=max_workers,
                search_type = search_type
            )
        st.session_state[cache_key] = reranked
        st.session_state[time_key] = time.time() - t0
    else:
        reranked = st.session_state[cache_key]
        t0 = st.session_state.get(time_key)

    if t0 is not None:
        _print_timing("Stage 3 total time", t0)
    else:
        st.text("[✅] Using cached Stage 3 results")

    if not reranked:
        st.warning("No items returned from re-ranking.")
        return

    is_shot = "frame_paths" in reranked[0]

    if is_shot:
        st.info(f"Showing top-{len(reranked)} re-ranked shots")

        shot_frame_lists = [item["frame_paths"] for item in reranked]
        all_paths = [p for lst in shot_frame_lists for p in lst]

        t_load = time.time()
        images = _load_images_batch(all_paths, max_workers=len(all_paths))
        _print_timing("Batch load all shot frames", t_load)

        idx = 0
        for i, item in enumerate(reranked, start=1):
            st.markdown(f"#### 🎬 Re-ranked Shot #{i} — Score: {item['bge_score']:.2f}")
            frames = item["frame_paths"]
            n = len(frames)
            shot_images = images[idx: idx + n]
            idx += n

            cols = st.columns(8)
            for j, (path, img) in enumerate(zip(frames, shot_images)):
                with cols[j]:
                    st.image(img, use_container_width=True)
                    st.caption(path.split("/")[-1])

            st.markdown(
                f"<div style='text-align:center; margin:8px 0;'>"
                f"<b>Shot Path:</b> {item['shot_path']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # Dialog for full shot details
            dialog = st.dialog(f"Shot #{i} Details")

            @dialog
            def show_shot_info():
                st.markdown(f"## 🎞️ Chi tiết Shot #{i}")
                st.write("**Shot Path:**", item["shot_path"])
                st.write("**Shot ID:**",   item.get("shot_id", "–"))
                st.write("**FPS:**",       item.get("fps", "–"))
                st.write("**Start Time:**", f"{item.get('start_time', '–')} s")
                st.write("**End Time:**",   f"{item.get('end_time',   '–')} s")
                st.write("**Source:**",     item.get("source", "–"))
                st.write("**Stage-3 Score:**", item["bge_score"])
                st.markdown("---")
                st.write("**BLIP Caption:**"); st.info(item["blip_caption"])
                st.write("**LLM Caption:**");  st.success(item["llm_caption"])
                st.write("**Explanation:**");  st.write(item.get("bge_explanation", ""))

            if st.button("ℹ️ Detail Infor", key=f"info3_shot_{i}"):
                show_shot_info()

            preview_dialog = st.dialog(f"Preview Shot #{i}")

            @preview_dialog
            def show_preview():
                st.video(item["shot_path"], format="video/mp4", start_time=item.get("start_time", 0))
                print(item["shot_path"])
            if st.button("▶️ Preview Shot", key=f"preview_shot_stage3_{i}"):
                show_preview()

    else:
        st.info(f"Showing top-{len(reranked)} re-ranked frames")
        cols_per_row = 8

        # Render frame-level in grid
        for base in range(0, len(reranked), cols_per_row):
            batch = reranked[base: base + cols_per_row]
            cols = st.columns(cols_per_row)
            for j, item in enumerate(batch):
                idx_item = base + j + 1
                with cols[j]:
                    try:
                        img = _load_image_from_path(item["frame_path"])
                        st.image(img, use_container_width=True)
                    except:
                        st.write("Image load error")

                    short = _get_display_path(item["frame_path"])
                    st.caption(
                        f"#{idx_item} | {short}<br>"
                        f"Score: {item['bge_score']:.2f}<br>"
                        f"Time: {item.get('timestamp', 0):.2f}s",
                        unsafe_allow_html=True,
                    )

                    dialog = st.dialog(f"Frame #{idx_item} Details")

                    @dialog
                    def show_frame_info():
                        st.markdown(f"## 🖼️ Chi tiết Frame #{idx_item}")
                        st.write("**Frame Path:**",   item["frame_path"])
                        st.write("**Frame Number:**", item.get("frame_number", "–"))
                        st.write("**Shot Name:**",    item.get("shot_name", "–"))
                        st.write("**Shot Index:**",   item.get("shot_idx", "–"))
                        st.write("**Source:**",       item.get("source", "–"))
                        st.write("**Timestamp:**",    f"{item.get('timestamp',0)} s")
                        st.write("**FPS:**",          item.get("fps", "–"))
                        st.write("**Stage-1 Score:**", item.get("score", "–"))
                        st.write("**Stage-3 Score:**", item["bge_score"])
                        st.markdown("---")
                        st.write("**BLIP Caption:**"); st.info(item["blip_caption"])
                        st.write("**LLM Caption:**");  st.success(item["llm_caption"])
                        st.write("**Explanation:**");  st.write(item.get("bge_explanation", ""))

                    if st.button("ℹ️ Detail Infor", key=f"info3_frame_{idx_item}"):
                        show_frame_info()

                    # — Preview Shot dialog —
                    preview_dialog = st.dialog(f"Preview Shot for Frame #{idx_item+1}")

                    @preview_dialog
                    def show_preview():
                        # reconstruct shot_path from frame's shot_idx
                        shot_path = item["source"]
                        print(shot_path)
                        st.video(shot_path, format="video/mp4", start_time=item["timestamp"])

                    if st.button("▶️ Preview Shot", key=f"preview_stage3_{idx_item}"):
                        show_preview()
