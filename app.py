import os
import cv2
import torch
import traceback
from PIL import Image
from difflib import SequenceMatcher
from transformers import BlipProcessor, BlipForConditionalGeneration
from ultralytics import YOLO
import gradio as gr

# --- Device Configuration ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"🚀 Initializing models on device: {DEVICE}")

# --- Load models once globally ---
yolo_model = YOLO("yolov8n.pt")

blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_model.to(DEVICE)
blip_model.eval()


# --- Aspect-Ratio-Preserving Frame Extraction ---
def extract_frames(video_path, max_frames=12, max_dim=640):
    """
    Extracts evenly spaced frames from a video while preserving aspect ratio.
    Limits the maximum dimension to max_dim for memory and processing efficiency.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []

    def process_frame(raw_bgr):
        h, w = raw_bgr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / float(max(h, w))
            new_w, new_h = int(w * scale), int(h * scale)
            raw_bgr = cv2.resize(raw_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        rgb_frame = cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb_frame)

    if total_frames <= 0:
        # Fallback for streams or videos with unreadable frame counts
        count = 0
        step = 10
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                frames.append(process_frame(frame))
            count += 1
    else:
        # Select evenly spaced indices across the entire duration
        sample_count = min(max_frames, max(1, total_frames))
        step = max(1, total_frames // sample_count)
        indices = [min(i * step, total_frames - 1) for i in range(sample_count)]
        indices = sorted(list(set(indices)))

        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            frames.append(process_frame(frame))

    cap.release()
    return frames


# --- Caption Merging & Formatting ---
def merge_and_format_captions(captions, similarity_threshold=0.75):
    """
    Cleans, deduplicates, and formats captions into coherent sentences.
    """
    merged = []
    for cap in captions:
        cap = cap.strip()
        if not cap:
            continue
        # Ensure sentence capitalization
        formatted = cap[0].upper() + cap[1:]
        if not formatted.endswith((".", "!", "?")):
            formatted += "."

        # Deduplicate using fuzzy similarity
        if all(SequenceMatcher(None, cap.lower(), m.lower()).ratio() < similarity_threshold for m in merged):
            merged.append(formatted)

    return merged


# --- Human-Friendly Object Summarization ---
def summarize_objects(object_counts, total_frames):
    """
    Produces a natural-language summary of frequently detected objects across frames.
    """
    # Filter objects that appear in at least 2 frames or 15% of frames
    threshold = max(2, int(total_frames * 0.15))
    common_objects = [
        obj for obj, count in sorted(object_counts.items(), key=lambda x: x[1], reverse=True)
        if count >= threshold
    ]

    if not common_objects:
        return "No persistent key objects were identified with high confidence."

    if len(common_objects) == 1:
        obj_text = f"**{common_objects[0]}**"
    elif len(common_objects) == 2:
        obj_text = f"**{common_objects[0]}** and **{common_objects[1]}**"
    else:
        obj_text = f"{', '.join(f'**{o}**' for o in common_objects[:-1])}, and **{common_objects[-1]}**"

    return f"Frequently detected elements across scenes: {obj_text}."


# --- Main Description Generator (Streaming) ---
def generate_motion_description(video_input):
    try:
        if video_input is None:
            yield "⚠️ Please upload or record a video first."
            return

        video_path = video_input if isinstance(video_input, str) else getattr(video_input, "name", str(video_input))

        yield "⏳ **Step 1/3:** Extracting video frames..."
        frames = extract_frames(video_path, max_frames=12, max_dim=640)

        if not frames:
            yield "❌ **Error:** Unable to extract frames from the uploaded video. Please verify the format."
            return

        yield f"⏳ **Step 2/3:** Running YOLOv8 object detection across {len(frames)} frames..."
        object_counts = {}
        # Batch inference with YOLO
        yolo_results = yolo_model(frames, verbose=False, conf=0.45)
        for res in yolo_results:
            for cls, conf in zip(res.boxes.cls, res.boxes.conf):
                name = yolo_model.names[int(cls)]
                object_counts[name] = object_counts.get(name, 0) + 1

        yield f"⏳ **Step 3/3:** Generating visual descriptions with BLIP ({DEVICE.upper()})..."
        # Batched BLIP inference
        inputs = blip_processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_new_tokens=40)

        raw_captions = [
            blip_processor.decode(out_id, skip_special_tokens=True).strip()
            for out_id in generated_ids
        ]

        deduped_captions = merge_and_format_captions(raw_captions)
        object_summary = summarize_objects(object_counts, len(frames))

        # Build clean, structured output
        summary_paragraphs = " ".join(deduped_captions) if deduped_captions else "No distinct scene actions identified."

        output_markdown = f"""### 📋 Video Overview

{summary_paragraphs}

---

### 🔍 Key Detected Objects
{object_summary}

---

### ⏱️ Scene Breakdown
"""
        for i, cap in enumerate(deduped_captions, start=1):
            output_markdown += f"- **Scene moment {i}:** {cap}\n"

        yield output_markdown

    except Exception as e:
        yield f"❌ **An unexpected error occurred:**\n\n```python\n{traceback.format_exc()}\n```"


# --- Gradio UI Layout ---
custom_css = """
.container { max-width: 900px; margin: auto; padding-top: 1rem; }
.output-box { border-radius: 8px; }
"""

with gr.Blocks(title="Video Description Generator", css=custom_css) as demo:
    gr.Markdown(
        """
        # 🎬 Video Description Generator
        Upload an MP4 video to extract keyframes, detect prominent objects, and generate a cohesive narrative.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload or Record Video", sources=["upload"])
            generate_btn = gr.Button("✨ Generate Description", variant="primary", size="lg")
            device_badge = gr.Markdown(f"⚡ *Hardware acceleration:* `{DEVICE.upper()}`")

        with gr.Column(scale=1):
            output_display = gr.Markdown(label="Generated Description", elem_classes=["output-box"])

    generate_btn.click(
        fn=generate_motion_description,
        inputs=[video_input],
        outputs=[output_display],
    )

if __name__ == "__main__":
    demo.launch()
