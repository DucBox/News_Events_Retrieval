#!/usr/bin/env python3

import os
import re
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from underthesea import word_tokenize
from scipy.sparse import vstack, save_npz
import joblib
import time

# === IMPORT TOKENIZER TỪ MODULE CỐ ĐỊNH ===
from Retrieval.search_utils import hybrid_tokenizer

# === CONFIG ===
OCR_ROOT           = "/content/drive/MyDrive/HCMC_AI/data/OCR"
SAVE_DIR           = "/content/drive/MyDrive/News-Events-Retrieval/Data/ocr_index_hybrid_v2"
VECTORIZER_PATH    = os.path.join(SAVE_DIR, "vectorizer.pkl")
TFIDF_MATRIX_PATH  = os.path.join(SAVE_DIR, "tfidf_matrix.npz")
REL_PATHS_PATH     = os.path.join(SAVE_DIR, "rel_paths.json")

os.makedirs(SAVE_DIR, exist_ok=True)

# === Utils ===
def get_all_txt_paths(root: str) -> list[str]:
    """
    Parallel search all .txt under subdirectories.
    """
    subdirs = [p for p in Path(root).iterdir() if p.is_dir()]
    print(f"[📂] Found {len(subdirs)} subdirectories under {root}")
    def _rglob_txt(d: Path) -> list[str]:
        return [str(p) for p in d.rglob("*.txt")]

    all_lists: list[list[str]] = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for lst in tqdm(ex.map(_rglob_txt, subdirs), total=len(subdirs), desc="🔍 Searching TXT files"):
            all_lists.append(lst)
    # flatten and sort
    all_paths = sorted(path for sublist in all_lists for path in sublist)
    return all_paths

def read_txt(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    except:
        return ""

def txt_to_jpg_path(txt_path: str) -> str:
    return txt_path.replace("/OCR/", "/Mid_Frames/").replace(".txt", ".jpg")

# === Step 1: Load all OCR paths ===
print("[Step 1] Gathering OCR .txt paths...")
all_txt_paths = get_all_txt_paths(OCR_ROOT)
print(f"📂 Total OCR files found: {len(all_txt_paths)}")

# === Step 2: Read all texts in parallel ===
print("[Step 2] Reading OCR texts...")
with ThreadPoolExecutor(max_workers=16) as ex:
    all_texts = list(tqdm(ex.map(read_txt, all_txt_paths), total=len(all_txt_paths), desc="📥 Reading texts"))

# === Step 3: Fit vectorizer (no max_features) ===
print("[Step 3] Fitting TF-IDF vectorizer (no limit)...")
vectorizer = TfidfVectorizer(tokenizer=hybrid_tokenizer)
t0 = time.time() 
vectorizer.fit(all_texts)
print(f"⚙️ Vectorizer fitted in {time.time() - t0:.1f}s")

# === Save vectorizer ===
joblib.dump(vectorizer, VECTORIZER_PATH)
print(f"✅ Saved vectorizer to {VECTORIZER_PATH}")

# === Step 4: Transform all texts to sparse TF-IDF ===
print("[Step 4] Transforming texts to TF-IDF sparse vectors...")
def _transform_single(text: str):
    return vectorizer.transform([text])

tfidf_blocks = []
with ProcessPoolExecutor(max_workers=16) as ex:
    for block in tqdm(ex.map(_transform_single, all_texts), total=len(all_texts), desc="🔁 Transforming TF-IDF"):
        tfidf_blocks.append(block)

tfidf_matrix = vstack(tfidf_blocks)
print(f"✅ Final TF-IDF matrix shape: {tfidf_matrix.shape}")

# === Step 5: Save sparse matrix + rel_paths.json ===
print("[Step 5] Saving matrix and paths...")
save_npz(TFIDF_MATRIX_PATH, tfidf_matrix)
print(f"💾 Saved TF-IDF matrix to {TFIDF_MATRIX_PATH}")

# parallel rel_paths
print("[Step 5] Generating relative image paths...")
with ThreadPoolExecutor(max_workers=16) as ex:
    rel_paths = list(tqdm(ex.map(txt_to_jpg_path, all_txt_paths),
                          total=len(all_txt_paths), desc="🔁 Mapping to JPG"))

with open(REL_PATHS_PATH, "w", encoding="utf-8") as f:
    json.dump(rel_paths, f, ensure_ascii=False, indent=2)
print(f"💾 Saved relative paths to {REL_PATHS_PATH}")

print("🎉 All done!")
