import os
import cv2
import torch
import numpy as np
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

# Common English stop words for semantic token comparison
STOP_WORDS = {
    "a", "an", "the", "in", "on", "at", "of", "and", "with", "is", "are", "to",
    "for", "from", "by", "that", "this", "it", "as", "into", "there", "some",
    "be", "or", "over", "near", "next", "up", "down", "around", "their", "its"
}


# --- Visual Similarity Check ---
def is_visually_similar(img1, img2, diff_threshold=0.06):
    """
    Checks if two PIL images are nearly identical (e.g. static camera / freeze frame).
    Uses downscaled grayscale difference for fast comparison.
    """
    try:
        g1 = cv2.resize(cv2.cvtColor(np.array(img1), cv2.COLOR_RGB2GRAY), (64, 64))
        g2 = cv2.resize(cv2.cvtColor(np.array(img2), cv2.COLOR_RGB2GRAY), (64, 64))
        mean_diff = np.mean(np.abs(g1.astype(float) - g2.astype(float))) / 255.0
        return mean_diff < diff_threshold
    except Exception:
        return False


# --- Aspect-Ratio-Preserving & Visually Distinct Frame Extraction ---
def extract_frames(video_path, max_frames=12, max_dim=640, filter_visual_duplicates=True):
    """
    Extracts evenly spaced, visually distinct frames while preserving aspect ratio.
    Filters out static/duplicate frames so identical shots don't generate duplicate captions.
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
        # Fallback for streams or unknown frame count
        count = 0
        step = 8
        last_frame = None
        while len(frames) < max_frames:
            ret, frame = cap.read()
            if not ret:
                break
            if count % step == 0:
                img = process_frame(frame)
                if not filter_visual_duplicates or last_frame is None or not is_visually_similar(img, last_frame):
                    frames.append(img)
                    last_frame = img
            count += 1
    else:
        # Sample slightly more candidate frames to allow filtering out visual duplicates
        num_candidates = min(max_frames * 2, max(1, total_frames))
        step = max(1, total_frames // num_candidates)
        candidate_indices = [min(i * step, total_frames - 1) for i in range(num_candidates)]
        candidate_indices = sorted(list(set(candidate_indices)))

        last_frame = None
        for idx in candidate_indices:
            if len(frames) >= max_frames:
                break
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret or frame is None:
                continue
            img = process_frame(frame)

            # Skip visually redundant frames
            if filter_visual_duplicates and last_frame is not None and is_visually_similar(img, last_frame):
                continue

            frames.append(img)
            last_frame = img

        # Fallback: if video was very static and too few frames were kept, ensure at least 2 frames
        if len(frames) < 2 and total_frames > 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = cap.read()
            if ret and frame is not None:
                frames = [process_frame(frame)]

    cap.release()
    return frames


# --- Content Word Extraction for Semantic Comparison ---
def get_content_words(text):
    """
    Extracts informative keywords by removing punctuation and common stop words.
    """
    words = [w.strip(".,!?:;\"'()[]{}") for w in text.lower().split()]
    return set(w for w in words if w and w not in STOP_WORDS)


# --- Multi-Level Duplicate Detection ---
def is_duplicate_caption(candidate, existing_captions, char_thresh=0.58, word_thresh=0.50):
    """
    Determines if candidate caption is redundant compared to existing captions.
    Combines:
    1. Fuzzy character sequence similarity (SequenceMatcher)
    2. Content word overlap (Jaccard similarity)
    3. Subset containment (one sentence is just a slight variation of another)
    """
    cand_lower = candidate.lower().strip()
    cand_words = get_content_words(cand_lower)

    for existing in existing_captions:
        exist_lower = existing.lower().strip()
        exist_words = get_content_words(exist_lower)

        # 1. Fuzzy string sequence similarity
        char_sim = SequenceMatcher(None, cand_lower, exist_lower).ratio()
        if char_sim >= char_thresh:
            return True

        # 2. Content word overlap & subset containment
        if cand_words and exist_words:
            intersection = cand_words.intersection(exist_words)
            union = cand_words.union(exist_words)
            jaccard = len(intersection) / float(len(union))
            containment = len(intersection) / float(min(len(cand_words), len(exist_words)))

            # Redundant if 50%+ keyword Jaccard overlap OR 70%+ containment
            if jaccard >= word_thresh or containment >= 0.70:
                return True

    return False


# --- Fine-Tuned Caption Deduplication & Formatting ---
def merge_and_format_captions(captions, strictness="Strict (No Repetition)"):
    """
    Aggressively deduplicates captions across semantic, lexical, and structural levels.
    """
    threshold_map = {
        "Strict (No Repetition)": (0.55, 0.45),
        "Balanced": (0.62, 0.52),
        "Relaxed": (0.75, 0.65)
    }
    char_thresh, word_thresh = threshold_map.get(strictness, (0.55, 0.45))

    accepted = []
    for raw in captions:
        raw = raw.strip()
        if not raw:
            continue
        # Ensure proper sentence capitalization and punctuation
        formatted = raw[0].upper() + raw[1:]
        if not formatted.endswith((".", "!", "?")):
            formatted += "."

        # Check for duplication against all previously accepted captions
        if not is_duplicate_caption(formatted, accepted, char_thresh=char_thresh, word_thresh=word_thresh):
            accepted.append(formatted)

    return accepted


# --- Narrative Synthesis ---
def synthesize_narrative(captions):
    """
    Combines distinct key moments into a flowing, readable narrative
    rather than a disjointed list of repetitive statements.
    """
    if not captions:
        return "No distinct visual actions identified."
    if len(captions) == 1:
        return captions[0]

    # Combine deduplicated unique moments into a coherent summary paragraph
    return " ".join(captions)


# --- Human-Friendly Object Summarization ---
def summarize_objects(object_counts, total_frames):
    """
    Produces a clean natural-language summary of frequently detected objects across frames.
    """
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

    return f"Key elements detected throughout the video: {obj_text}."


# --- Main Description Generator (Streaming) ---
def generate_motion_description(video_input, dedup_mode="Strict (No Repetition)", max_keyframes=12):
    try:
        if video_input is None:
            yield "⚠️ Please upload or record a video first."
            return

        video_path = video_input if isinstance(video_input, str) else getattr(video_input, "name", str(video_input))

        yield "⏳ **Step 1/3:** Extracting visually distinct keyframes..."
        frames = extract_frames(video_path, max_frames=int(max_keyframes), max_dim=640, filter_visual_duplicates=True)

        if not frames:
            yield "❌ **Error:** Unable to extract frames from the uploaded video. Please verify the format."
            return

        yield f"⏳ **Step 2/3:** Running YOLOv8 object detection across {len(frames)} unique keyframes..."
        object_counts = {}
        yolo_results = yolo_model(frames, verbose=False, conf=0.45)
        for res in yolo_results:
            for cls, conf in zip(res.boxes.cls, res.boxes.conf):
                name = yolo_model.names[int(cls)]
                object_counts[name] = object_counts.get(name, 0) + 1

        yield f"⏳ **Step 3/3:** Generating visual descriptions with BLIP ({DEVICE.upper()})..."
        inputs = blip_processor(images=frames, return_tensors="pt", padding=True).to(DEVICE)
        with torch.no_grad():
            generated_ids = blip_model.generate(**inputs, max_new_tokens=40)

        raw_captions = [
            blip_processor.decode(out_id, skip_special_tokens=True).strip()
            for out_id in generated_ids
        ]

        # Apply multi-level fine-tuned deduplication
        deduped_captions = merge_and_format_captions(raw_captions, strictness=dedup_mode)
        object_summary = summarize_objects(object_counts, len(frames))
        narrative_summary = synthesize_narrative(deduped_captions)

        # Build clean, structured output
        output_markdown = f"""### 📋 Cohesive Video Summary

{narrative_summary}

---

### 🔍 Detected Objects & Elements
{object_summary}

---

### ⏱️ Distinct Visual Events ({len(deduped_captions)} unique moments identified)
"""
        for i, cap in enumerate(deduped_captions, start=1):
            output_markdown += f"- **Moment {i}:** {cap}\n"

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
        Upload an MP4 video to extract keyframes, eliminate duplicate repetitive captions, and generate a cohesive narrative.
        """
    )

    with gr.Row():
        with gr.Column(scale=1):
            video_input = gr.Video(label="Upload or Record Video", sources=["upload"])
            with gr.Accordion("⚙️ Deduplication & Analysis Settings", open=False):
                dedup_selector = gr.Radio(
                    choices=["Strict (No Repetition)", "Balanced", "Relaxed"],
                    value="Strict (No Repetition)",
                    label="Deduplication Filter Mode",
                    info="Strict mode removes semantic and keyword duplicates for a clean narrative."
                )
                frames_slider = gr.Slider(
                    minimum=6,
                    maximum=18,
                    value=12,
                    step=2,
                    label="Max Keyframes to Sample"
                )

            generate_btn = gr.Button("✨ Generate Description", variant="primary", size="lg")
            device_badge = gr.Markdown(f"⚡ *Hardware acceleration:* `{DEVICE.upper()}`")

        with gr.Column(scale=1):
            output_display = gr.Markdown(label="Generated Description", elem_classes=["output-box"])

    generate_btn.click(
        fn=generate_motion_description,
        inputs=[video_input, dedup_selector, frames_slider],
        outputs=[output_display],
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(server_name="0.0.0.0", server_port=port)


