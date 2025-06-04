# frontend/stage2_ui.py

import streamlit as st
import time
from Retrieval.stage2_dp import refine_shots_with_dp
from frontend.utils import (
    _load_images_batch,
    _print_timing,
    _load_image_from_path,
    _get_display_path,
)

def render_stage2_block(
    results: list[dict],
    query: str,
    embedder,
    top_m: int = 20,
):
    num_shots = len(results)
    st.markdown("### 🧠 Stage 2 – DP Refinement Results")
    st.write(f"Refined {num_shots} shots") 
    results = results[:top_m]

    shot_frame_lists = [res["frame_paths"] for res in results]
    all_paths = [p for frame_list in shot_frame_lists for p in frame_list]

    t1 = time.time()
    images = _load_images_batch(all_paths, max_workers= len(all_paths))
    _print_timing("Batch load all frames", t1)

    idx = 0
    for i, result in enumerate(results, start=1):
        st.markdown(f"#### 🎬 Refined Shot #{i} — Score: {result['score']:.4f}")
        paths = result["frame_paths"]
        n = len(paths)
        shot_images = images[idx : idx + n]
        idx += n

        cols = st.columns(8)
        for j, (path, img) in enumerate(zip(paths, shot_images)):
            with cols[j]:
                st.image(img, use_container_width=True)
                st.caption(path.split("/")[-1])

        st.markdown(
            f"<div style='text-align:center; margin:8px 0;'>"
            f"<b>Score:</b> {result['score']:.4f} | "
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
            st.write("**Start:**",     f"{result.get('start_time','–')} s")
            st.write("**End:**",       f"{result.get('end_time','–')} s")
            st.write("**Source:**",    result.get("source", "–"))
            st.write("**Frames**",     result.get("frame_paths", "-"))
            st.write("**Score:**",     result["score"])
            st.write("**Tags:**",      result.get("tags", "-"))
            st.markdown("---")
            st.write("**BLIP Caption:**"); st.info(result["blip_caption"])
            st.write("**LLM Caption:**");  st.success(result["llm_caption"])

        if st.button("ℹ️ Detail Infor", key=f"info_stage2_shot_{i}"):
            show_shot_info()

        preview_dialog = st.dialog(f"Preview Shot #{i}")

        @preview_dialog
        def show_preview():
            st.video(result["shot_path"], format="video/mp4", start_time=result.get("start_time", 0))
            print(result["shot_path"])
        if st.button("▶️ Preview Shot", key=f"preview_shot_{i}"):
            show_preview()
