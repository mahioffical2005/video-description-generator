---
title: Video Description Generator
emoji: 🔥
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.46.0
app_file: app.py
pinned: false
---

# 🎬 Multimodal Video Description & Scene Understanding Generator

[![Live Demo](https://img.shields.io/badge/🤗%20Hugging%20Face-Live%20Demo-yellow.svg)](https://huggingface.co/spaces/mahi-project/video-description-generator)
[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/mahioffical2005/video-description-generator)
[![Python](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg?logo=pytorch)](https://pytorch.org/)
[![Hugging Face Transformers](https://img.shields.io/badge/Transformers-BLIP-FFD21E.svg)](https://huggingface.co/Salesforce/blip-image-captioning-base)
[![YOLOv8](https://img.shields.io/badge/Ultralytics-YOLOv8-00FFFF.svg)](https://docs.ultralytics.com/)
[![Gradio](https://img.shields.io/badge/UI-Gradio-orange.svg)](https://gradio.app/)

An end-to-end, deep learning-powered video summarization pipeline that combines **Object Detection (YOLOv8)** and **Vision-Language Understanding (Salesforce BLIP)** with a proprietary **3-layer deduplication engine** to generate coherent, non-repetitive video narratives.

---

## 🔗 Live Application & Repository Links

- **🌐 Live Demo (Hugging Face Spaces):** [https://huggingface.co/spaces/mahi-project/video-description-generator](https://huggingface.co/spaces/mahi-project/video-description-generator)
- **💻 GitHub Repository:** [https://github.com/mahioffical2005/video-description-generator](https://github.com/mahioffical2005/video-description-generator)

---

## 🌟 Key Highlights & Engineering Features

### 1. Multimodal AI Pipeline
- **Object Detection (Ultralytics YOLOv8n):** Real-time spatial object detection to identify key elements (people, vehicles, animals, objects) and determine their persistence across video frames.
- **Vision-Language Captioning (Salesforce BLIP-base):** Generates contextual, natural language descriptions for visual scenes.

### 2. Intelligent 3-Layer Deduplication Engine
Standard frame-by-frame captioning suffers from severe repetition (e.g., repeating similar captions for adjacent frames). This project solves that using a specialized 3-stage filter:
- **Layer 1: Visual Keyframe Filtering (Frame Level):** Calculates normalized pixel-difference metrics between consecutive frames to skip visually static / freeze frames before running inference.
- **Layer 2: Semantic & Lexical Filtering (Text Level):** Strips syntactic stop words and runs multi-dimensional comparisons:
  - **Jaccard Content-Word Overlap:** Catches sentences sharing $\ge 45-50\%$ of core keywords.
  - **Subset Containment:** Detects when one sentence is an uninformative subset of another ($\ge 70\%$ containment).
  - **Fuzzy Sequence Matching:** Catches paraphrased or reordered expressions.
- **Layer 3: Cohesive Narrative Synthesis (Output Level):** Weaves distinct chronological events into a unified, flowing summary rather than an unpunctuated stream of disconnected sentences.

### 3. Aspect-Ratio-Preserving Frame Extraction
- Avoids stretching or squashing non-square videos (e.g., 16:9 1080p footage).
- Scales videos proportionally with efficient seeking (`cv2.CAP_PROP_POS_FRAMES`) for fast decoding.

### 4. Hardware Optimization & Batching
- Auto-detects available hardware (`CUDA` GPU vs. `CPU`).
- Uses batched tensor generation (`blip_processor(images=frames, ...)`), speeding up captioning by **5x–10x** compared to sequential processing.

---

## 🏗️ Architecture & Pipeline Flow

```
[ Uploaded MP4 Video ]
         │
         ▼
[ 1. Aspect-Ratio-Preserving Frame Extraction (OpenCV) ]
         │
         ▼
[ 2. Visual Similarity Pruning ] ──► (Discards Static / Redundant Frames)
         │
    ┌────┴──────────────────────────┐
    ▼                               ▼
[ YOLOv8 Object Detection ]    [ Batched BLIP Captioning ]
(Identifies frequent objects)   (Generates scene captions)
    └────┬──────────────────────────┘
         │
         ▼
[ 3. Multi-Level Semantic Deduplication ]
(Jaccard Keyword Overlap + Containment + Fuzzy String Matching)
         │
         ▼
[ 4. Narrative Synthesis Engine ]
         │
         ▼
[ Structured Markdown Output: Overview + Objects + Timeline ]
```

---

## 💻 Tech Stack

- **Deep Learning Framework:** PyTorch
- **Vision-Language Model:** Salesforce BLIP (`Salesforce/blip-image-captioning-base`) via Hugging Face Transformers
- **Object Detection:** Ultralytics YOLOv8 Nano (`yolov8n.pt`)
- **Computer Vision:** OpenCV (`opencv-python-headless`), Pillow
- **Web Interface:** Gradio 5 / 6 (`gr.Blocks`, `gr.Video`)
- **Cloud Deployment:** Hugging Face Spaces, Render

---

## 🚀 Local Quickstart Guide

### 1. Clone the Repository
```bash
git clone https://github.com/mahioffical2005/video-description-generator.git
cd video-description-generator
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

*(Optional: For CPU-only environments, install lightweight PyTorch via:)*
```bash
pip install torch torchvision --extra-index-url https://download.pytorch.org/whl/cpu
```

### 3. Run the Application
```bash
python app.py
```

Once running, navigate to **`http://127.0.0.1:7860`** in your browser.

---

## 📊 Sample Output Structure

When a video is processed, the system outputs:

```markdown
### 📋 Cohesive Video Summary
A young person is walking along a sidewalk near a busy urban street. A red car drives past in the background.

---

### 🔍 Detected Objects & Elements
Key elements detected throughout the video: **person**, **car**, and **traffic light**.

---

### ⏱️ Distinct Visual Events (2 unique moments identified)
- **Moment 1:** A young person is walking along a sidewalk near a busy urban street.
- **Moment 2:** A red car drives past in the background.
```

---

## 👤 Author & Portfolio
- **Developer:** Mahitha Geddavalasa
- **GitHub:** [@mahioffical2005](https://github.com/mahioffical2005)
- **Hugging Face:** [@mahi-project](https://huggingface.co/mahi-project)
