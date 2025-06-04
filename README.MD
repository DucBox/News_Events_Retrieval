# 🗞️ News-Events Retrieval System

This project builds a **video retrieval system** specialized for **news and event footage**. It allows users to search for relevant video segments using **text descriptions**, ranging from simple concrete queries to complex multi-scene abstract queries.

## 🎯 Goal

To enable **semantic retrieval** of news video segments from large archives using multimodal features (visual + textual), and re-rank the results for higher accuracy using advanced LLMs.

---

## ✅ Key Features

* **Multimodal Search Support:**

  * Text → Image (CLIP-based)
  * Text → Caption (BLIP or LLM caption vectors)
  * Text → OCR (TF-IDF/keyword match – optional)

* **Shot-level Indexing:**

  * Each video is split into shots, each with 8 representative frames
  * Shot embeddings and captions are pooled from those frames

* **Hierarchical Retrieval Pipeline:**

  * Stage 1: Retrieve top-K candidates using FAISS index
  * Stage 2: Dynamic Programming alignment for multi-shot queries (optional)
  * Stage 3: Re-ranking using LLM (e.g., GPT-4) based on caption relevance

* **Flexible Querying:**

  * Supports both **single-scene** and **multi-scene** queries
  * Enables **textual abstraction** over multiple shots

---

## 🧱 Dataset Structure (not included)

> **Note:** All `.h5`, `.index`, and `.json` data files are **not included** in this repository. Below is the expected structure to recreate or understand the system.

### 📁 Folder Layout

```bash
data_layout_h5/
├── core.h5
├── clip.h5
├── blip.h5
├── llm.h5
├── clip_frame.index
├── clip_shot.index
├── blip_frame.index
├── blip_shot.index
├── llm_frame.index
├── llm_shot.index

Short_Video_JSON_Size8/
└── Lxx/
    └── Vyyy.json
```

---

## 🧩 HDF5 File Structures

### `core.h5`

Stores all metadata (frame-level and shot-level), caption annotations, and mappings.

```bash
core/
├── frame/
│   ├── paths, blip_caption, llm_caption
│   └── metadata: frame_number, source, fps, timestamp, shot_idx
├── shot/
│   ├── paths, blip_caption, llm_caption
│   └── metadata: shot_id, start_frame, end_frame, start_time, end_time, fps, source
```

### `clip.h5`, `blip.h5`, `llm.h5`

Store 768-dim embedding vectors for both frames and shots.

```bash
frame/
├── paths, vectors [F, 768]
shot/
├── paths, vectors [S, 768]
```

---

## 📄 JSON Shot Descriptions

Located in `Short_Video_JSON_Size8/`, each file contains:

```json
[
  {
    "shot": 1,
    "frames": [start_frame, end_frame],
    "time": [start_sec, end_sec],
    "source": "L01_V001",
    "fps": 25.0,
    "source_frame": [list of 8 frame image paths],
    "embedding": [768 floats],
    "path": ".../Shot_0001.mp4"
  }
]
```

---

## 🔍 Supported Retrieval Modes

| Method           | Description                                         |
| ---------------- | --------------------------------------------------- |
| CLIP             | Vector similarity from query to frame/shot          |
| Caption (BLIP)   | Caption embedding from pretrained BLIP model        |
| Caption (LLM)    | Refined captions embedded via LLM (GPT-4)           |
| Re-ranking (LLM) | Use LLM to re-rank top-K based on caption alignment |
| Shot alignment   | Optional DP alignment for multi-shot queries        |

---

## 💡 Applications

* Fast retrieval of specific scenes from long news videos
* Support for both keyword-based and abstract, multi-event queries
* Groundwork for caption-based video understanding or summarization

---

## 🏆 Achievements
✅ Accepted at SOIC 2024
* The initial version of this system was officially accepted at the SOIC 2024 under the theme "Multimodal Information Retrieval".

## 🎖 Finalist – HCMC AI Challenge 2024
* Selected as a national finalist in the HCMC AI Challenge 2024, recognizing outstanding AI projects across Vietnam

---

## 📌 Authors

* Author: Ngô Quang Đức
* gmail: quangducngo0811@gmail.com
