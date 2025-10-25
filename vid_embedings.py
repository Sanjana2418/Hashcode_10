# ==========================================
# 📦 Install dependencies
# ==========================================
!pip install transformers torch pillow tqdm ffmpeg-python

# ==========================================
# 🎥 Import libraries
# ==========================================
import os
import subprocess
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import cv2
from transformers import CLIPProcessor, CLIPModel
from google.colab import files

# ==========================================
# ⚙️ Configuration
# ==========================================
FRAME_DIR = "frames"
FPS = 1  # how many frames per second to extract

# ==========================================
# 📁 Upload and extract video frames
# ==========================================
def get_video_path():
    print("📤 Please upload your video file...")
    uploaded = files.upload()
    video_path = list(uploaded.keys())[0]
    print(f"✅ Uploaded video: {video_path}")
    return video_path

def extract_frames(video_path, frame_dir, fps=1):
    os.makedirs(frame_dir, exist_ok=True)
    subprocess.run([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={fps}",
        os.path.join(frame_dir, "frame_%04d.jpg"),
        "-hide_banner", "-loglevel", "error"
    ])
    print(f"🎞️ Extracted frames saved to: {frame_dir}")

# ==========================================
# 🧠 Generate embeddings
# ==========================================
def embed_frames(model, processor, frame_dir, device):
    frame_files = sorted(os.listdir(frame_dir))
    embeddings = []

    for frame_file in tqdm(frame_files, desc="🔍 Embedding frames"):
        img_path = os.path.join(frame_dir, frame_file)
        img = Image.open(img_path).convert("RGB")
        inputs = processor(images=img, return_tensors="pt").to(device)

        with torch.no_grad():
            emb = model.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings)
    return embeddings, frame_files

def embed_query(model, processor, text, device):
    inputs = processor(text=[text], return_tensors="pt", padding=True).to(device)
    with torch.no_grad():
        emb = model.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

# ==========================================
# 🧩 Match query to best frame
# ==========================================
def find_best_frame(frame_embeds, text_embed, frame_files):
    sims = np.dot(text_embed, frame_embeds.T)[0]
    best_idx = int(np.argmax(sims))
    return frame_files[best_idx], sims[best_idx]

# ==========================================
# 🚀 Main pipeline
# ==========================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🧮 Using device: {device}")

    video_path = get_video_path()
    query = input("🧠 Enter your text query (e.g. 'person starts speaking'): ")

    extract_frames(video_path, FRAME_DIR, FPS)

    print("🚀 Loading OpenCLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name).to(device)
    processor = CLIPProcessor.from_pretrained(model_name)

    frame_embeds, frame_files = embed_frames(model, processor, FRAME_DIR, device)
    text_embed = embed_query(model, processor, query, device)

    best_frame, score = find_best_frame(frame_embeds, text_embed, frame_files)
    print(f"\n🏆 Best matching frame: {best_frame} (similarity: {score:.3f})")

    # Display the best matching frame
    from IPython.display import Image as IPImage, display
    display(IPImage(filename=os.path.join(FRAME_DIR, best_frame)))

if __name__ == "__main__":
    main()
