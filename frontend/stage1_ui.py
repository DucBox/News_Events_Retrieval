import streamlit as st
import os
import time
from PIL import Image
from Retrieval.stage1 import stage1_retrieve_shots
from Retrieval.cache_loader import core_cache, shot_cache
from concurrent.futures import ThreadPoolExecutor
from frontend.utils import _load_images_batch, _print_timing, _load_image_from_path, _get_display_path

def render_stage1_block(query: str, top_k: int, search_type: str, type_search: str, embedder, refine_stage2: bool, results):
    st.markdown("### 🔍 Stage 1 Retrieval Results")

    t0 = time.time()
    if not results:
        st.warning("No results returned.")
        return

    if search_type == "frame":
        st.info(f"Showing top-{len(results)} frame-level results")
        frame_paths = [res["frame_path"] for res in results]

        t_batch = time.time()
        images = _load_images_batch(frame_paths, max_workers=32)
        _print_timing("Batch image load", t_batch)

        cols = st.columns(8)
        for i, (result, img) in enumerate(zip(results, images)):
            with cols[i % 8]:
                st.image(img, use_container_width=True)

                short_path = _get_display_path(result["frame_path"])
                st.caption(
                    f"#{i+1} | {short_path}<br>"
                    f"Score: {result['score']:.4f}<br>"
                    f"Time: {result['timestamp']:.2f}s",
                    unsafe_allow_html=True
                )

                # — Info dialog —
                dialog = st.dialog(f"Frame #{i+1} Details")

                @dialog
                def show_info_dialog():
                    st.markdown(f"## 🖼️ Chi tiết Frame #{i+1}")
                    st.write("**Frame Path:**", result["frame_path"])
                    st.write("**Frame Number:**", result["frame_number"])
                    st.write("**Shot Name:**", result["shot_name"])
                    st.write("**Shot Index:**", result["shot_idx"])
                    st.write("**Tags:**",      result["tags"])
                    st.write("**Source:**", result["source"])
                    st.write("**Timestamp:**", f"{result['timestamp']} s")
                    st.write("**FPS:**", result["fps"])
                    st.write("**Score:**", result["score"])
                    st.markdown("---")
                    st.write("**BLIP Caption:**"); st.info(result["blip_caption"])
                    st.write("**LLM Caption:**");  st.success(result["llm_caption"])

                if st.button("ℹ️ Detail Infor", key=f"info_{i}"):
                    show_info_dialog()

                # — Preview Shot dialog —
                preview_dialog = st.dialog(f"Preview Shot for Frame #{i+1}")

                @preview_dialog
                def show_preview():
                    # reconstruct shot_path from frame's shot_idx
                    shot_path = result["source"]
                    print(shot_path)
                    st.video(shot_path, format="video/mp4", start_time=result["timestamp"])

                if st.button("▶️ Preview Shot", key=f"preview_{i}"):
                    show_preview()

    elif search_type == "shot":
        st.info(f"Showing top-{len(results)} shot-level results")

        shot_frame_lists = [res["frame_paths"] for res in results]
        all_paths = [p for frame_list in shot_frame_lists for p in frame_list]

        t_batch = time.time()
        images = _load_images_batch(all_paths, max_workers=len(all_paths))
        _print_timing("Batch load all shot frames", t_batch)

        idx = 0
        for i, result in enumerate(results, start=1):
            st.markdown(f"### 🎬 Shot #{i} — Score: {result['score']:.4f}")

            paths = result["frame_paths"]
            n = len(paths)
            shot_images = images[idx : idx + n]
            idx += n

            cols = st.columns(8)
            for j, (frame_path, img) in enumerate(zip(paths, shot_images)):
                with cols[j]:
                    st.image(img, use_container_width=True)
                    st.caption(frame_path.split("/")[-1])

            st.markdown(
                f"<div style='text-align:center; margin:8px 0;'>"
                f"<b>Shot Path:</b> {result['shot_path']}"
                f"</div>",
                unsafe_allow_html=True,
            )

            # dialog for full-shot details
            dialog = st.dialog(f"Shot #{i} Details")

            @dialog
            def show_shot_info():
                st.markdown(f"## 🎞️ Chi tiết Shot #{i}")
                st.write("**Shot Path:**", result["shot_path"])
                st.write("**Shot ID:**",   result.get("shot_id", "–"))
                st.write("**FPS:**",       result.get("fps", "–"))
                st.write("**Tags:**",      result.get("tags", "-"))
                st.write("**Start Time:**", f"{result.get('start_time','–')} s")
                st.write("**End Time:**",   f"{result.get('end_time','–')} s")
                st.write("**Source:**",     result.get("source", "–"))
                st.write("**Score:**",      result["score"])
                st.markdown("---")
                st.write("**BLIP Caption:**"); st.info(result["blip_caption"])
                st.write("**LLM Caption:**");  st.success(result["llm_caption"])

            if st.button("ℹ️ Detail Infor", key=f"info_shot_{i}"):
                show_shot_info()

            preview_dialog = st.dialog(f"Preview Shot #{i}")

            @preview_dialog
            def show_preview():
                st.video(result["shot_path"], format="video/mp4", start_time=result.get("start_time", 0))
                print(result["shot_path"])
            if st.button("▶️ Preview Shot", key=f"preview_shot_{i}"):
                show_preview()

    _print_timing("Stage 1 TOTAL", t0)