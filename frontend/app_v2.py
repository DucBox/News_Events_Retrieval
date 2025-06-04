import streamlit as st
from Retrieval.cache_loader import preload_all_caches
from Retrieval.embedder import CLIPEmbedder
from Retrieval.auto_mode import agentic 
from Retrieval.stage1 import stage1_retrieve_shots
from Retrieval.stage2_dp import refine_shots_with_dp
from frontend.stage1_ui import render_stage1_block
from frontend.stage2_ui import render_stage2_block
from frontend.stage3_ui import render_stage3_block

# automatic‑mode agent
auto_agent_enabled = True  
if auto_agent_enabled:
    try:
    except Exception as e:
        auto_agent_enabled = False
        st.warning(
            "Không tìm thấy module Retrieval.agentic – Automatic mode sẽ bị ẩn.\n" + str(e)
        )

@st.cache_resource
def init_resources():
    """Load FAISS, TF‑IDF, HDF5 and return a shared CLIP embedder."""
    preload_all_caches()
    return CLIPEmbedder()


# Helper – run the three‑stage pipeline 

def _execute_search(
    *,
    query_description: str,
    query: str,
    full_query: str,
    top_k1: int,
    search_type: str,
    search_method: str,
    enable_s2: bool,
    top_m2: int,
    enable_s3: bool,
    top_k3: int,
    embedder,
):

    if search_type == "shot" and search_method == "ocr":
        st.error(" OCR chưa được hỗ trợ ở cấp độ shot. Vui lòng chọn 'frame' hoặc phương pháp tìm kiếm khác.")
        return

    if enable_s2:
        if search_type == "frame":
            sub_queries = [s.strip() for s in query.split(".") if s.strip()]
            full_shot_paths = []
            for q in sub_queries:
                result = stage1_retrieve_shots(
                    query=q,
                    embedder=embedder,
                    search_type=search_type,
                    top_k=top_k1,
                    type_search=search_method,
                    refine_stage2=True,
                )
                full_shot_paths.extend(result)
            shot_paths = list(dict.fromkeys(full_shot_paths))  # order‑preserving unique
            st.header("Stage 2 – DP Refinement")
            stage2_results = refine_shots_with_dp(shot_paths, full_query, embedder)
            render_stage2_block(
                results=stage2_results,
                query=full_query,
                embedder=embedder,
                top_m=top_m2,
            )
            items_for_stage3 = stage2_results
        else:
            st.error("Stage‑2 hiện chỉ hỗ trợ khi search_type='frame'.")
            return
    else:
        # ------------------------ Stage‑1 UI (frame OR shot) -----------------
        st.header("Stage 1 – Retrieval")
        if search_type == "frame":
            sub_queries = [s.strip() for s in query.split(".") if s.strip()]
            results = []
            for q in sub_queries:
                result = stage1_retrieve_shots(
                    query=q,
                    embedder=embedder,
                    search_type=search_type,
                    top_k=top_k1,
                    type_search=search_method,
                    refine_stage2=False,
                )
                results.extend(result)

            frame_dict = {}
            for item in results:
                fp = item["frame_path"]
                if fp not in frame_dict or item["score"] > frame_dict[fp]["score"]:
                    frame_dict[fp] = item
            stage1_results = sorted(frame_dict.values(), key=lambda x: x["score"], reverse=True)
            render_stage1_block(
                query=query,
                top_k=top_k1,
                search_type=search_type,
                type_search=search_method,
                embedder=embedder,
                refine_stage2=False,
                results=stage1_results,
            )
            items_for_stage3 = stage1_results
        else:  
            stage1_results = stage1_retrieve_shots(
                query=query,
                embedder=embedder,
                search_type=search_type,
                top_k=top_k1,
                type_search=search_method,
                refine_stage2=False,
            )
            render_stage1_block(
                query=query,
                top_k=top_k1,
                search_type=search_type,
                type_search=search_method,
                embedder=embedder,
                refine_stage2=False,
                results=stage1_results,
            )
            items_for_stage3 = stage1_results

    # ------------------------------ Stage‑3 ---------------------------------
    if enable_s3:
        st.header("Stage 3 – LLM Re‑rank")
        render_stage3_block(
            items=items_for_stage3,
            query=query_description,
            top_k_rerank=top_k3,
            max_workers=16,
            search_type=search_type,
        )

# interaction mode
def _render_interactive(embedder):
    """Original sidebar form for manual parameter selection."""

    with st.sidebar.form("search_form"):
        st.header("Resources")
        if st.form_submit_button("Load Caches & Models", use_container_width=True):
            with st.spinner("Loading..."):
                _ = init_resources()
            st.success("Loaded!")

        st.header("Search Parameters")
        query_description = st.text_area("Query Description", height=80)
        full_query = st.text_area("Full Query", height=80)
        query = st.text_input("Query")
        top_k1 = st.number_input("Top‑k (Stage 1)", min_value=1, value=50, step=1)
        search_type = st.selectbox("Search Level", ["frame", "shot"])
        search_method = st.selectbox("Search Method", ["clip", "llm_caption", "blip_caption", "ocr"])

        st.header("Stage Controls")
        enable_s2 = st.checkbox("Enable Stage 2 (DP Refinement)", value=False)
        top_m2 = st.number_input("Top‑m (Stage 2)", min_value=1, value=20, step=1)
        enable_s3 = st.checkbox("Enable Stage 3 (LLM Re‑rank)", value=False)
        top_k3 = st.number_input("Top‑k (Stage 3)", min_value=1, value=20, step=1)

        run_interactive = st.form_submit_button("Run Search")
        if run_interactive:
            st.session_state.run_interactive = True

        if search_type == "shot" and search_method == "ocr":
            st.error(" OCR chưa được hỗ trợ ở cấp độ shot. Vui lòng chọn 'frame' hoặc phương pháp tìm kiếm khác.")
            return False  # abort early

    if not st.session_state.get("run_interactive"):
        st.info("🚀 Nhấn **Run Search** để bắt đầu.")
        return False

    _execute_search(
        query_description = query_description,
        query=query,
        full_query=full_query,
        top_k1=top_k1,
        search_type=search_type,
        search_method=search_method,
        enable_s2=enable_s2,
        top_m2=top_m2,
        enable_s3=enable_s3,
        top_k3=top_k3,
        embedder=embedder,
    )
    return True


# automatic mode
def _render_automatic(embedder):
    if not auto_agent_enabled:
        st.warning("Automatic mode is disabled – missing agentic() implementation.")
        return False

    with st.sidebar.form("auto_form"):
        st.header("Automatic Mode")
        full_description = st.text_area("Full Description", height=120)
        run_auto = st.form_submit_button("Automate Search")
        if run_auto:
            st.session_state.auto_run = True
            try:
                params = agentic(full_description)
            except Exception as e:
                st.error(" agentic() failed: " + str(e))
                st.session_state.auto_run = False
                return False
            st.session_state.auto_params = params

    if not st.session_state.get("auto_run"):
        st.info("Nhập mô tả và nhấn **Automate Search** để kích hoạt tìm kiếm tự động.")
        return False

    params = st.session_state.auto_params

    # Display parsed parameters for transparency
    with st.expander("🔍 Parameters detected by agentic()", expanded=False):
        st.json(params, expanded=False)

    # Map expected keys with sensible fallbacks
    _execute_search(
        query=params.get("query", ""),
        full_query=params.get("full_query", ""),
        top_k1=int(params.get("top_k1", 50)),
        search_type=params.get("search_type", "frame"),
        search_method=params.get("search_method", "clip"),
        enable_s2=bool(params.get("enable_s2", False)),
        top_m2=int(params.get("top_m2", 50)),
        enable_s3=bool(params.get("enable_s3", False)),
        top_k3=int(params.get("top_k3", 50)),
        embedder=embedder,
    )
    return True


def render():
    st.title("News-Events Retrieval")

    embedder = init_resources()

    mode_options = ["Interactive"] + (["Automatic"] if auto_agent_enabled else [])
    mode = st.sidebar.radio("Mode", mode_options, index=0)

    if st.sidebar.button("🧹 Clear UI & Reset State"):
        st.session_state.clear()
        st.rerun()

    if mode == "Interactive":
        _render_interactive(embedder)
    else:  # "Automatic"
        _render_automatic(embedder)


if __name__ == "__main__":
    render()
