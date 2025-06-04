import re, numpy as np, os
from underthesea import word_tokenize
from sklearn.metrics.pairwise import cosine_similarity

from Retrieval.config import OCR_VECTORIZER
from Retrieval.cache_loader import (
    faiss_index_cache, ocr_cache, core_cache
)

# ---------- 1. Tokenizer ---------- #
def hybrid_tokenizer(text: str):
    vi_tokens = word_tokenize(text, format="text").split()
    en_tokens = re.findall(r"[a-zA-Z_]{2,}", text)
    return [t.lower() for t in (vi_tokens + en_tokens)]


# ---------- 2. get_index_path ---------- #
def get_index_path(type_search: str, search_type: str, index_dir: str):
    supported = {
        "clip": {"frame": "clip_frame.index", "shot": "clip_shot.index"},
        "llm_caption": {"frame": "llm_frame.index", "shot": "llm_shot.index"},
        "blip_caption": {"frame": "blip_frame.index", "shot": "blip_shot.index"},
    }
    if type_search not in supported:
        raise ValueError(f"type_search '{type_search}' not supported")
    if search_type not in supported[type_search]:
        raise ValueError(f"No index for {type_search}/{search_type}")
    return os.path.join(index_dir, supported[type_search][search_type])


# ---------- 3. encode_query ---------- #
def encode_query_for_search(query: str, type_search: str, embedder=None):
    from openai import OpenAI
    from Retrieval.config import OPENAI_API_KEY
    openai_client = OpenAI(api_key=OPENAI_API_KEY)

    if type_search == "clip":
        if embedder is None:
            raise ValueError("embedder (CLIP) bắt buộc cho 'clip'")
        return embedder.encode_text(query).cpu().numpy().astype(np.float32)

    if type_search in ("llm_caption", "blip_caption"):
        res = openai_client.embeddings.create(
            input=query,
            model="text-embedding-3-small",
        )
        return np.array(res.data[0].embedding, dtype=np.float32)

    if type_search == "ocr":
        vectorizer = ocr_cache["vectorizer"]
        return vectorizer.transform([query]).toarray().astype(np.float32)

    raise ValueError(f"Unsupported type_search: {type_search}")
