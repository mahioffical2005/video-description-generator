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

## 🎯 Interview Showcase: Problem Statement & Engineering Challenges

> This section summarizes the technical decisions, algorithmic design, and problem-solving methodology behind the project for technical interviews and code reviews.

### 1. The Core Problem
Most off-the-shelf video captioning tutorials take a naive approach: sample arbitrary frames, run image captioning on each frame, and concatenate the text. This leads to **three critical failure modes**:
1. **Severe Text Repetition & Looping:** Consecutive video frames contain nearly identical content, generating repetitive statements (e.g., *"a person walking... a young person walking... someone walking down the road"*).
2. **Aspect Ratio Distortion:** Forcibly resizing frames to arbitrary square dimensions (e.g., $480 \times 480$) distorts shapes and degrades both bounding-box localization and caption quality.
3. **Sequential Latency Bottlenecks:** Running sequential, unbatched inference on 15+ frames causes high latency (30s+ on CPU), making interactive deployment impractical.

---

### 2. Engineering Challenges & Solutions

| Challenge | Root Cause | Solution Implemented |
| :--- | :--- | :--- |
| **Temporal Frame Redundancy** | Slow camera movement produces visually identical frames. | **Visual Keyframe Pruning (`is_visually_similar`):** Downsamples consecutive frames to $64 \times 64$ grayscale and calculates mean absolute pixel difference. Static/freeze frames are discarded before inference. |
| **Semantic Text Duplication** | BLIP generates slightly varied phrasing for the same scene event. | **Multi-Level Semantic Deduplication:** Strips stop-words, calculates **Jaccard Content-Word Overlap** ($\ge 45\%$) and **Subset Containment** ($\ge 70\%$) to prune redundant sentences. |
| **Inference Latency** | Sequential loop calling `blip_model.generate()` one frame at a time. | **Batched Tensor Generation:** Batches keyframes into a single padded tensor pass (`torch.no_grad()`), accelerating captioning by **5x–10x**. |
| **Variable Aspect Ratios** | Non-square video resolutions (e.g., 16:9 1080p). | **Proportional Downsampling:** Scales maximum dimension to 640px while preserving native aspect ratio, using fast OpenCV seeking (`CAP_PROP_POS_FRAMES`). |
| **Disjointed Output** | Raw sentence concatenation lacks narrative structure. | **Narrative Synthesis:** Weaves distinct chronological events into an executive overview, object frequency summary, and timeline breakdown. |

---

### 3. Technical Interview Q&A (Key Talking Points)

<details>
<summary><b>Q1: Why combine YOLOv8 and BLIP instead of using an end-to-end Video-LLM (like Video-LLaVA or Video-ChatGPT)?</b></summary>
<br>
<b>Answer:</b> Resource footprint, deployment feasibility, and cost. Modern Video-LLMs require 16GB–40GB of VRAM (expensive A100/H100 GPUs) and have inference latencies of several seconds per query. By pairing a lightweight object detector (<b>YOLOv8n</b>, 3.2M params) with a specialized vision-language model (<b>BLIP-base</b>, 223M params), this pipeline runs comfortably on commodity CPUs and free cloud tiers (Hugging Face Spaces) with sub-second per-stage latency, while still providing both fine-grained object tracking and natural narrative synthesis.
</details>

<details>
<summary><b>Q2: How does your deduplication algorithm differ from standard embedding cosine similarity?</b></summary>
<br>
<b>Answer:</b> Cosine similarity over embeddings requires running an additional embedding model (such as Sentence-BERT), which adds memory overhead and latency. Furthermore, embedding models often struggle with subset containment (e.g., scoring "a man walking" and "a man in a black jacket walking down a street" with medium similarity). Our algorithm uses a lightweight, zero-overhead combination of <b>Jaccard content-word overlap</b> and <b>lexical containment</b>, executing in $O(N)$ time with zero additional GPU memory.
</details>

<details>
<summary><b>Q3: How do you handle edge cases like corrupted video metadata or variable frame rates?</b></summary>
<br>
<b>Answer:</b> In <code>extract_frames()</code>, we check if <code>CAP_PROP_FRAME_COUNT</code> returns $\le 0$ (a known issue with streamed MP4s or variable-framerate files). If frame count is valid, we calculate evenly spaced indices and use fast seeking (<code>CAP_PROP_POS_FRAMES</code>). If metadata is corrupted, we fall back to a dynamic stepping loop with a hard cap on <code>max_frames</code>, preventing unbounded memory allocation.
</details>

---

## 🌟 Key Features

### 1. Multimodal AI Pipeline
- **Object Detection (Ultralytics YOLOv8n):** Real-time spatial object detection to identify key elements (people, vehicles, animals, objects) and determine their persistence across video frames.
- **Vision-Language Captioning (Salesforce BLIP-base):** Generates contextual, natural language descriptions for visual scenes.

### 2. Intelligent 3-Layer Deduplication Engine
- **Layer 1: Visual Keyframe Filtering (Frame Level):** Pixel-difference metrics skip visually static frames.
- **Layer 2: Semantic & Lexical Filtering (Text Level):** Jaccard keyword overlap + containment checks.
- **Layer 3: Cohesive Narrative Synthesis (Output Level):** Weaves distinct chronological events into a unified, flowing summary.

### 3. User-Configurable Gradio Interface
- **Deduplication Mode Selector:** Toggle between **Strict (No Repetition)**, **Balanced**, and **Relaxed** filters.
- **Keyframe Slider:** Adjust the number of frames to sample (6 to 18).
- **Native Video Player:** Integrated video player with scrub and preview controls.

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
