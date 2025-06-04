import streamlit as st
import os
import time
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

def _load_images_batch(paths, max_workers=16):
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(Image.open, paths))

def _print_timing(label: str, t0: float) -> None:
    dt = time.time() - t0
    st.text(f"[⏱️] {label:<25}: {dt:7.4f}s")

def _load_image_from_path(path: str):
    return Image.open(path)

def _get_display_path(p: str) -> str:
    parts = p.split("/")
    if "Mid_Frames" in parts:
        idx = parts.index("Mid_Frames")
        return "/".join(parts[idx + 1:idx + 3])
    elif "Shot" in p:
        idx = parts.index("Short_Video")
        return "/".join(parts[idx + 1:idx + 3])
    return "?"