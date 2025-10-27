# ==============================================================================
# 🚀 FULLY INCREMENTAL VIDEO RAG & VISUAL SEARCH
# ==============================================================================

# ----------------------------
# 📦 Install Dependencies (Combined List)
# ----------------------------

# !pip install ffmpeg-python vosk sentence-transformers faiss-cpu transformers torch numpy Pillow tqdm requests pydub

# ----------------------------
# 🎥 Core Imports
# ----------------------------
import os, json, wave, ffmpeg, threading, time, numpy as np, requests, re
import subprocess
import torch
from PIL import Image
from tqdm import tqdm
from google.colab import files
import faiss
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer
from transformers import CLIPProcessor, CLIPModel
# Note: PyPDF2/docx/pydub/etc. are excluded here for brevity, assuming focus is on Video Mode 1

# ----------------------------
# 🔑 Configuration and Global Variables
# ----------------------------
NEBIUS_API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDBjeDc5Z25lZDdndDlicGYSIXNlcnZpY2VhY2NvdW50LWUwMGpiZ25nZ2tmbTBmYmVocTIMCPKx88cGEMbliZMCOgsI7rSLkwcQwLD6SEACWgNlMDA.AAAAAAAAAAErukSVVLitCcaFMxoS3Lkkixc17mvLhW6V6yCO1xJ9k8bjTsplCdFcPBHFpBDT3TitA5lBrSAD603QBeGbwiMP"

FRAME_DIR = "frames"
CLIP_FPS = 1 # Frames per second to process
AUDIO_CHUNK_SIZE = 400
VISUAL_CHUNK_SIZE = 30 # Number of frames to process/embed at a time

# --- Audio/Text RAG Globals ---
transcript_chunks = []
all_words = []
embedder = SentenceTransformer("all-MiniLM-L6-v2")
AUDIO_EMBEDDING_DIM = 384
audio_index = faiss.IndexFlatL2(AUDIO_EMBEDDING_DIM)
audio_lock = threading.Lock() # Lock for audio/text RAG components

# --- Visual Search Globals ---
frame_files_indexed = [] # Stores file names of indexed frames (for lookup)
VISUAL_EMBEDDING_DIM = 512 # Dimension for CLIP-ViT-B/32
visual_index = faiss.IndexFlatL2(VISUAL_EMBEDDING_DIM)
visual_lock = threading.Lock() # Lock for visual components
CLIP_MODEL = None
CLIP_PROCESSOR = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
file_path = None
file_ext = None


# ----------------------------------------------------
# 🔊 RAG (Text/Audio) Utility Functions
# ----------------------------------------------------

def add_audio_chunk_to_index(text_list):
    """Embed new text chunks and add to audio FAISS index."""
    embeddings = np.array(embedder.encode(text_list))
    with audio_lock:
        audio_index.add(embeddings)

def retrieve_text(query, top_k=3):
    """Performs semantic search on the audio FAISS index."""
    q_emb = embedder.encode([query])
    with audio_lock:
        if audio_index.ntotal == 0:
            return "⏳ No transcribed chunks ready yet..."

        k = min(top_k, len(transcript_chunks))
        _, idxs = audio_index.search(np.array(q_emb), k)

        results = []
        for i in idxs[0]:
            c = transcript_chunks[i]
            results.append(f"[{c['start_time']:.1f}s - {c['end_time']:.1f}s] {c['text']}")
    return "\n".join(results)


# ----------------------------------------------------
# 👁️ VISUAL (Image) Utility Functions
# ----------------------------------------------------

def init_clip_model():
    """Initializes the CLIP model and processor."""
    global CLIP_MODEL, CLIP_PROCESSOR
    print(f"🧮 Using device: {DEVICE}")
    print("🚀 Loading OpenCLIP model...")
    model_name = "openai/clip-vit-base-patch32"
    CLIP_MODEL = CLIPModel.from_pretrained(model_name).to(DEVICE)
    CLIP_PROCESSOR = CLIPProcessor.from_pretrained(model_name)

def embed_query_clip(text):
    """Generates CLIP embedding for the text query."""
    inputs = CLIP_PROCESSOR(text=[text], return_tensors="pt", padding=True).to(DEVICE)
    with torch.no_grad():
        emb = CLIP_MODEL.get_text_features(**inputs)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy()

def retrieve_visual(query):
    """Performs visual search on the image FAISS index."""
    if CLIP_MODEL is None:
        return "⏳ CLIP model is still initializing."

    q_emb = embed_query_clip(query)

    with visual_lock:
        if visual_index.ntotal == 0:
            return "⏳ No frames indexed yet..."

        # Search for the single best match
        _, idxs = visual_index.search(np.array(q_emb), 1)
        best_idx = idxs[0][0]

        # Get filename and timestamp
        best_frame_file = frame_files_indexed[best_idx]
        frame_number = int(best_frame_file.split('_')[-1].split('.')[0])
        timestamp = (frame_number - 1) / CLIP_FPS

    return best_frame_file, timestamp

def process_and_add_visual_chunk(frame_files_batch):
    """Embeds a batch of frames and adds them to the visual FAISS index."""
    embeddings = []

    for frame_file in frame_files_batch:
        img_path = os.path.join(FRAME_DIR, frame_file)
        if not os.path.exists(img_path): continue

        img = Image.open(img_path).convert("RGB")
        inputs = CLIP_PROCESSOR(images=img, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            emb = CLIP_MODEL.get_image_features(**inputs)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            embeddings.append(emb.cpu().numpy())

    if embeddings:
        embeddings = np.vstack(embeddings)
        with visual_lock:
            visual_index.add(embeddings)
            frame_files_indexed.extend(frame_files_batch)
        print(f"🖼️ Indexed {len(frame_files_batch)} new frames. Total: {visual_index.ntotal}")

# ----------------------------------------------------
# 🧠 Unified QA Function
# ----------------------------------------------------

def ask_multimodal_question(question):
    """Determines if the query is for text RAG or visual search."""

    # 1. Word Timestamp Query (Text/Audio)
    if file_ext in ["mp4", "mov", "avi", "mkv"] and re.search(r"'(.*?)'", question):
        word = re.findall(r"'(.*?)'", question)[0]
        word_lower = word.lower()
        matches = [w for w in all_words if w["word"].lower() == word_lower]
        if matches:
            times = [f"{w['start']:.2f}s" for w in matches]
            print(f"\n⏱ Exact timestamps for '{word}': {', '.join(times)}")
        else:
            print(f"\n⏱ '{word}' not found in transcribed text so far.")
        return

    # 2. Visual Search Query (Image)
    # Heuristic: If the question contains visual terms (e.g., 'frame', 'see', 'look', 'image'), treat it as visual search.
    if re.search(r'\b(frame|see|look|image|visual|show|when)\b', question, re.IGNORECASE):
        result = retrieve_visual(question)
        if isinstance(result, str):
            print(f"\n🤖 Visual Answer: {result}")
            return

        best_frame_file, timestamp = result

        print(f"\n=============================================")
        print("👀 Visual Search Result")
        print(f"🏆 Best matching frame: {best_frame_file}")
        print(f"⏰ Approximate time in video: {timestamp:.1f} seconds")
        print("=============================================")

        # Display the best matching frame
        try:
            from IPython.display import Image as IPImage, display
            display(IPImage(filename=os.path.join(FRAME_DIR, best_frame_file)))
        except ImportError:
            print("Note: Run this in a Jupyter/Colab environment to display the image.")
        return

    # 3. Semantic RAG Query (Text/Audio)
    context = retrieve_text(question)

    if "No transcribed chunks ready yet..." in context:
        print("\n🤖 Answer: The video is still transcribing. Try again in a few moments.")
        return

    url = "https://api.studio.nebius.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    data = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for answering based on the provided text context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 150
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        print("\n🤖 Answer:\n", answer)
    else:
        print(f"Error: {response.status_code} {response.text}")


# ----------------------------------------------------
# 🧵 Incremental Thread Functions
# ----------------------------------------------------

def transcribe_incremental_thread(audio_path):
    """Handles real-time audio transcription and RAG indexing."""
    global all_words

    # Initialization (Vosk model loading excluded for brevity here, assume it's set up)
    if not os.path.exists("vosk-model-small-en-us-0.15"):
        os.system("wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip")
        os.system("unzip -q vosk-model-small-en-us-0.15.zip")

    model = Model("vosk-model-small-en-us-0.15")
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    buffer_text = ""
    buffer_start_time = 0
    print("🎯 Starting incremental transcription...")

    while True:
        data = wf.readframes(4000)
        if len(data) == 0: break

        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            words = res.get("result", [])
            text = res.get("text", "")

            with audio_lock:
                for w in words: all_words.append({"word": w["word"], "start": w["start"], "end": w["end"]})

            if text:
                if not buffer_text and words: buffer_start_time = words[0]["start"]
                buffer_text += text + " "

                if len(buffer_text) >= AUDIO_CHUNK_SIZE:
                    buffer_end_time = words[-1]["end"] if words else buffer_start_time
                    chunk = {"text": buffer_text.strip(), "start_time": buffer_start_time, "end_time": buffer_end_time}

                    with audio_lock: transcript_chunks.append(chunk)
                    add_audio_chunk_to_index([buffer_text.strip()])
                    print(f"📝 New audio chunk ready: [{buffer_start_time:.2f}s - {buffer_end_time:.2f}s]")
                    buffer_text = ""

    # Final remaining text
    if buffer_text:
        final_end_time = all_words[-1]["end"] if all_words else buffer_start_time
        chunk = {"text": buffer_text.strip(), "start_time": buffer_start_time, "end_time": final_end_time}
        with audio_lock: transcript_chunks.append(chunk)
        add_audio_chunk_to_index([buffer_text.strip()])
        print(f"📝 Final audio chunk ready: [{buffer_start_time:.2f}s - {final_end_time:.2f}s]")

    print("\n✅ Audio transcription finished.")


def visual_processing_thread(video_path):
    """Handles real-time frame extraction and CLIP indexing."""

    init_clip_model()
    os.makedirs(FRAME_DIR, exist_ok=True)

    # This uses a generator/subprocess trick to process frames incrementally
    print("🎯 Starting incremental visual processing...")

    # FFMPEG command to extract frames and pipe them out, or save them incrementally.
    # For simplicity and robust file-based indexing, we'll stream/save them to disk
    # and process them in batches as they appear.

    frame_index = 1
    while True:
        # Check if the next batch of frames has been extracted/saved
        start_frame_num = frame_index
        end_frame_num = frame_index + VISUAL_CHUNK_SIZE - 1

        # We need an external tool (like a separate FFmpeg run or a specialized library)
        # to continuously extract frames to disk.
        # For simplicity in this script, we'll simulate continuous extraction by checking for the files:

        # 1. Run FFmpeg to extract the next VISUAL_CHUNK_SIZE frames (This is highly simplified)
        # In a real app, a continuous stream reader would be better.
        # Here we rely on the main loop or a wrapper to ensure frames are extracted.

        current_frame_file = f"frame_{start_frame_num:04d}.jpg"
        if not os.path.exists(os.path.join(FRAME_DIR, current_frame_file)):

            # --- The true 'incremental' extraction step ---
            # This is the most complex part to do purely in Python/FFmpeg commands.
            # We'll rely on a continuous extraction command run beforehand and just read the files.

            # For this example, we'll assume a background process or prior call extracted them.
            # If all expected frames have appeared, the process is done.
            print("\n✅ Visual indexing is likely complete (or waiting for extraction to catch up).")
            break

        # 2. Collect the batch of filenames
        batch_filenames = [f"frame_{i:04d}.jpg" for i in range(start_frame_num, end_frame_num + 1)]

        # 3. Process and Index the batch
        process_and_add_visual_chunk(batch_filenames)

        frame_index = end_frame_num + 1
        time.sleep(1) # Small pause to simulate real-time work


# ----------------------------------------------------
# 🚀 Main Execution Setup
# ----------------------------------------------------

def get_uploaded_file():
    """Handles file upload and returns the path."""
    print("📤 Please upload your video file (mp4, mov, avi, mkv)...")
    uploaded = files.upload()
    if not uploaded:
        print("❌ No file uploaded.")
        return None, None
    f_path = list(uploaded.keys())[0]
    f_ext = f_path.split(".")[-1].lower()
    print(f"✅ Uploaded file: {f_path}")
    return f_path, f_ext

def run_multimodal_incremental_mode(video_path):
    """Sets up both audio and visual incremental processing, with cleanup."""

    # -----------------------------
    # 🧹 STEP 0: CLEANUP AND RESET
    # -----------------------------
    global frame_files_indexed, visual_index, transcript_chunks, all_words, audio_index

    print("\n🧹 Cleaning previous session data...")

    # Reset Visual Search Globals
    frame_files_indexed = []
    visual_index.reset()

    # Reset Audio RAG Globals
    transcript_chunks = []
    all_words = []
    audio_index.reset()

    # Clear the Frames directory on disk
    import shutil # We need to ensure shutil is imported at the top of the script
    if os.path.exists(FRAME_DIR):
        # Recursively remove the directory and all its contents
        shutil.rmtree(FRAME_DIR)
    os.makedirs(FRAME_DIR) # Recreate the empty directory

    # 1. Setup Audio Extraction
    audio_path = "temp_audio.wav"
    print("1/4: Extracting audio...") # Adjusted count due to new cleanup step
    try:
        ffmpeg.input(video_path).output(audio_path, format='wav', ac=1, ar='16000').run(overwrite_output=True)
        print("✅ Audio extracted")
    except Exception as e:
        print(f"❌ Error extracting audio: {e}")
        return

    # 2. Setup Frame Extraction (Must be done first to start creating files)
    print("2/4: Starting background frame extraction...")
    # This command runs in the background and saves frames to FRAME_DIR as the video plays.
    subprocess.Popen([
        "ffmpeg", "-i", video_path,
        "-vf", f"fps={CLIP_FPS}",
        os.path.join(FRAME_DIR, "frame_%04d.jpg"),
        "-y",
        "-hide_banner", "-loglevel", "error"
    ])

    # 3. Start Threads
    print("3/4: Starting audio and visual processing threads...")
    audio_thread = threading.Thread(target=transcribe_incremental_thread, args=(audio_path,), daemon=True)
    visual_thread = threading.Thread(target=visual_processing_thread, args=(video_path,), daemon=True)

    audio_thread.start()
    visual_thread.start()

    # 4. Start Interactive Loop
    print("\n=============================================")
    print("⚡️ MULTIMODAL Q&A STARTED (INCREMENTAL)")
    print("=============================================")
    print("🎯 Querying both text and images is enabled as the video processes.")
    print("Example Text Query: What is the speaker talking about?")
    print("Example Visual Query: Show me the frame when the person starts speaking")
    print("Type 'exit' to quit.")

    while True:
        q = input("\n❓ Your question: ")
        if q.lower() in ["exit", "quit"]:
            print("👋 Exiting...")
            break
        ask_multimodal_question(q)

if __name__ == "__main__":

    if not os.path.exists(FRAME_DIR): os.makedirs(FRAME_DIR)

    print("\n=============================================")
    print("🌐 Multimodal Incremental Processing System")
    print("=============================================")

    # Skip the menu and go straight to the primary goal mode
    file_path, file_ext = get_uploaded_file()

    if file_path and file_ext in ["mp4", "mov", "avi", "mkv"]:
        run_multimodal_incremental_mode(file_path)
    else:
        print("❌ Invalid file. Please upload a video file (mp4, mov, avi, mkv).")