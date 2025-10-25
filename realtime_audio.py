
# ----------------------------
# Install Dependencies
# ----------------------------
!pip install ffmpeg-python vosk sentence-transformers faiss-cpu transformers accelerate requests pydub PyPDF2 python-docx

# ----------------------------
# Imports
# ----------------------------
import os, json, wave, ffmpeg, threading, time, numpy as np, requests, re
from vosk import Model, KaldiRecognizer
from sentence_transformers import SentenceTransformer
import faiss
from google.colab import files

# ----------------------------
# Step 0: Nebius API Key
# ----------------------------
NEBIUS_API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDBnMHk2eGhleTJ6NjhjeWMSIXNlcnZpY2VhY2NvdW50LWUwMHNkZ3JxYW5mbmVqZjhhczILCNfg88cGELDfxB86DAjW44uTBxDAoZDqAUACWgNlMDA.AAAAAAAAAAHE15wn0qyl8gDoY3AxNb1GStEEw5NXtoLtoPbYhJo7CWpNAqtkVq0cjNyeI-yhZLkhKpcwkD2zAna9swGjorsH"

# ----------------------------
# Step 1: Upload video
# ----------------------------
uploaded = files.upload()
video_path = list(uploaded.keys())[0]
print(f"🎥 Video detected: {video_path}")

# ----------------------------
# Step 2: Extract audio
# ----------------------------
audio_path = "audio.wav"
ffmpeg.input(video_path).output(audio_path, format='wav', ac=1, ar='16000').run(overwrite_output=True)
print("✅ Audio extracted")

# ----------------------------
# Step 3: Initialize Vosk model
# ----------------------------
if not os.path.exists("vosk-model-small-en-us-0.15"):
    !wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
    !unzip -q vosk-model-small-en-us-0.15.zip

model = Model("vosk-model-small-en-us-0.15")
wf = wave.open(audio_path, "rb")
rec = KaldiRecognizer(model, wf.getframerate())
rec.SetWords(True)

# ----------------------------
# Step 4: Prepare global variables
# ----------------------------
transcript_chunks = []  # list of dicts: {text, start_time, end_time}
all_words = []          # list of dicts: {word, start, end}
chunk_size = 400
buffer_text = ""
buffer_start_time = 0
lock = threading.Lock()  # To protect transcript_chunks during updates




# ----------------------------
# Step 5: Embedder & FAISS index
# ----------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
embedding_dim = 384  # all-MiniLM-L6-v2 dimension
index = faiss.IndexFlatL2(embedding_dim)

def add_to_index(new_texts):
    """Embed new chunks and add to FAISS index."""
    embeddings = np.array(embedder.encode(new_texts))
    with lock:
        index.add(embeddings)

# ----------------------------
# Step 6: Incremental transcription in a separate thread
# ----------------------------
def transcribe_incremental():
    global buffer_text, buffer_start_time
    print("🎯 Starting incremental transcription...")
    while True:
        data = wf.readframes(4000)
        if len(data) == 0:
            break
        if rec.AcceptWaveform(data):
            res = json.loads(rec.Result())
            words = res.get("result", [])
            text = res.get("text", "")
            for w in words:
                all_words.append({"word": w["word"], "start": w["start"], "end": w["end"]})
            if text:
                if not buffer_text and words:
                    buffer_start_time = words[0]["start"]
                buffer_text += text + " "
                if len(buffer_text) >= chunk_size:
                    buffer_end_time = words[-1]["end"] if words else buffer_start_time
                    chunk = {"text": buffer_text.strip(), "start_time": buffer_start_time, "end_time": buffer_end_time}
                    with lock:
                        transcript_chunks.append(chunk)
                    add_to_index([buffer_text.strip()])
                    print(f"📝 New chunk ready: [{buffer_start_time:.2f}s - {buffer_end_time:.2f}s]")
                    buffer_text = ""
    # Add any remaining text
    if buffer_text:
        buffer_end_time = all_words[-1]["end"] if all_words else buffer_start_time
        chunk = {"text": buffer_text.strip(), "start_time": buffer_start_time, "end_time": buffer_end_time}
        with lock:
            transcript_chunks.append(chunk)
        add_to_index([buffer_text.strip()])
        print(f"📝 Final chunk ready: [{buffer_start_time:.2f}s - {buffer_end_time:.2f}s]")
    print("🎯 Incremental transcription finished. You can now ask questions.")

# Start transcription in a thread
transcription_thread = threading.Thread(target=transcribe_incremental, daemon=True)
transcription_thread.start()

# ----------------------------
# Step 7: Retrieval & QA functions
# ----------------------------
def retrieve(query, top_k=3):
    q_emb = embedder.encode([query])
    with lock:
        if len(transcript_chunks) == 0:
            return "⏳ No chunks ready yet..."
        _, idxs = index.search(np.array(q_emb), top_k)
        results = []
        for i in idxs[0]:
            if i < len(transcript_chunks):
                c = transcript_chunks[i]
                results.append(f"[{c['start_time']:.1f}s - {c['end_time']:.1f}s] {c['text']}")
    return "\n".join(results)

def get_word_timestamp(word):
    word_lower = word.lower()
    matches = [w for w in all_words if w["word"].lower() == word_lower]
    if matches:
        return [f"{w['start']:.2f}s" for w in matches]
    else:
        return ["Word not found"]

def ask_question(question):
    match = re.findall(r"'(.*?)'", question)
    if match:
        word = match[0]
        times = get_word_timestamp(word)
        print(f"\n⏱ Exact timestamps for '{word}': {', '.join(times)}")
        return
    context = retrieve(question)
    url = "https://api.studio.nebius.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    data = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "max_tokens": 150
    }
    response = requests.post(url, headers=headers, json=data)
    if response.status_code == 200:
        answer = response.json()["choices"][0]["message"]["content"]
        print("\n🤖 Answer:\n", answer)
    else:
        print("Error:", response.status_code, response.text)

# ----------------------------
# Step 8: Interactive loop
# ----------------------------
print("🎯 You can ask questions while transcription is still running. Type 'exit' to quit.\n")
while True:
    q = input("❓ Your question: ")
    if q.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break
    ask_question(q)
