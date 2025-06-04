import glob, json, os, time, faiss, h5py, numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from scipy.sparse import load_npz
import joblib

from Retrieval.config import (
    JSON_ROOT, H5_DIR, CORE_H5, FAISS_DIR,
    TFIDF_MATRIX, OCR_PATHS_JSON, OCR_VECTORIZER, EMB_ROOT, MAX_WORKERS
)

# === GLOBAL CACHES ===
shot_cache       = {}
core_cache       = {}
faiss_index_cache= {}
ocr_cache        = {}
embedding_cache  = {}


# ---------- 1. Shot-level JSON ---------- #
def preload_shot_json():
    json_files = glob.glob(os.path.join(JSON_ROOT, "L*", "V*.json"))
    print(f"[📥] Preload JSON shots ({len(json_files)} files)")

    def _load(path):
        local = {}
        with open(path) as f:
            data = json.load(f)
        for shot in data:
            local[shot["path"]] = {
                "shot_id": shot["shot"],
                "frame_paths": shot["source_frame"],
                "start_time": shot["time"][0],
                "end_time": shot["time"][1],
                "fps": shot["fps"],
                "frames": shot["frames"],
                "source": shot["source"],
            }
        return local

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = [ex.submit(_load, p) for p in json_files]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            shot_cache.update(fut.result())
    print(f"[✅] Shot cache: {len(shot_cache)} entries")


# ---------- 2. FAISS ---------- #
def preload_faiss():
    print("[📥] Preload FAISS indexes")
    for fname in os.listdir(FAISS_DIR):
        if fname.endswith(".index"):
            index_path = os.path.join(FAISS_DIR, fname)
            faiss_index_cache[index_path] = faiss.read_index(index_path)
            print(f"   • {fname} ({faiss_index_cache[index_path].ntotal} vec)")


# ---------- 3. core.h5 ---------- #
def _decode(a):
    return [x.decode() if isinstance(x, bytes) else x for x in a]

def preload_core_h5():
    print("[📥] Preload core.h5")
    with h5py.File(CORE_H5, "r") as f:
        core_cache["frame_paths"] = _decode(f["frame/paths"][:])
        core_cache["frame_blip"]  = _decode(f["frame/blip_caption"][:])
        core_cache["frame_llm"]   = _decode(f["frame/llm_caption"][:])
        core_cache["frame_meta"]  = {
            k: _decode(f["frame/metadata"][k][:]) if f["frame/metadata"][k].dtype.kind == "S"
            else f["frame/metadata"][k][:]
            for k in f["frame/metadata"]
        }

        core_cache["shot_paths"] = _decode(f["shot/paths"][:])
        core_cache["shot_blip"]  = _decode(f["shot/blip_caption"][:])
        core_cache["shot_llm"]   = _decode(f["shot/llm_caption"][:])
        core_cache["shot_meta"]  = {
            k: _decode(f["shot/metadata"][k][:]) if f["shot/metadata"][k].dtype.kind == "S"
            else f["shot/metadata"][k][:]
            for k in f["shot/metadata"]
        }
    print(f"[✅] core.h5: {len(core_cache['frame_paths'])} frames | {len(core_cache['shot_paths'])} shots")

# ---------- 4. OCR TF-IDF ---------- #
def preload_ocr():
    """
    Load TF-IDF sparse matrix + tokenizer-aware vectorizer.
    Giờ vectorizer.pkl đã được dump từ Retrieval.search_utils.hybrid_tokenizer,
    nên ta không cần monkey-patch nữa.
    """
    import json
    import joblib
    from scipy.sparse import load_npz
    from Retrieval.config import TFIDF_MATRIX, OCR_PATHS_JSON, OCR_VECTORIZER
    from Retrieval.cache_loader import ocr_cache

    print("[📥] Preload OCR TF-IDF")

    # Load sparse matrix
    ocr_cache["matrix"] = load_npz(TFIDF_MATRIX)

    # Load vectorizer (đã tham chiếu đúng Retrieval.search_utils.hybrid_tokenizer)
    ocr_cache["vectorizer"] = joblib.load(OCR_VECTORIZER)

    # Load relative paths
    with open(OCR_PATHS_JSON, "r", encoding="utf-8") as f:
        ocr_cache["paths"] = json.load(f)

    print(f"[✅] OCR matrix shape: {ocr_cache['matrix'].shape}")

# ---------- 5. NPZ Embeddings ---------- #
def preload_embeddings_npz():
    npz_files = glob.glob(os.path.join(EMB_ROOT, "L*", "V*.npz"))
    print(f"[📥] Preload NPZ embeddings ({len(npz_files)})")

    def _load(npz_path):
        Lxx = os.path.basename(os.path.dirname(npz_path))
        Vyyy = os.path.splitext(os.path.basename(npz_path))[0]
        key = f"{Lxx}/{Vyyy}"
        data = np.load(npz_path, allow_pickle=True)
        return key, {"paths": list(data["paths"]), "vectors": data["embeddings"]}

    with ThreadPoolExecutor(MAX_WORKERS) as ex:
        futs = [ex.submit(_load, p) for p in npz_files]
        for fut in tqdm(as_completed(futs), total=len(futs)):
            k, v = fut.result()
            embedding_cache[k] = v
    print(f"[✅] NPZ cache: {len(embedding_cache)} videos")


# ---------- Helper để preload tất cả ---------- #
def preload_all_caches():
    preload_shot_json()
    preload_faiss()
    preload_core_h5()
    preload_ocr()
    preload_embeddings_npz()
