import os
import sounddevice as sd
import wavio
import requests
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from tqdm import tqdm
import time



# -------------------------------------------------------------
# 1. Initialize embedding model
# -------------------------------------------------------------
print("Loading embedding model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded successfully.")

# Storage for transcript chunks
transcript_chunks = []

# -------------------------------------------------------------
# 2. Input menu
# -------------------------------------------------------------
print("\n=== PIL Data Input Mode ===")
print("1️⃣  Load from existing text file")
print("2️⃣  Load from video/audio file (MP4/MP3/WAV) [Not implemented]")
print("3️⃣  Record from microphone 🎙")
choice = input("\nChoose mode (1/2/3): ").strip()

# -------------------------------------------------------------
# 3. Handle input modes
# -------------------------------------------------------------
if choice == "1":
    # --- Text file mode ---
    filepath = input("📄 Enter path to text file: ").strip()
    with open(filepath, "r", encoding="utf-8") as f:
        full_text = f.read()
    words = full_text.split()
    for i in range(0, len(words), 400):
        chunk = " ".join(words[i:i+400])
        transcript_chunks.append({"text": chunk, "start_time": None, "end_time": None})
    print(f"✅ Loaded {len(transcript_chunks)} text chunks from {filepath}.")

elif choice == "3":
    # --- Microphone mode ---
    print("🎙 Recording from microphone...")
    samplerate = 16000
    duration = int(input("⏱ Enter duration in seconds: "))
    recording = sd.rec(int(samplerate * duration), samplerate=samplerate, channels=1, dtype='int16')
    sd.wait()
    wavio.write("mic_record.wav", recording, samplerate, sampwidth=2)
    audio_path = "mic_record.wav"
    print("✅ Audio saved as mic_record.wav")

    # --- Upload to AssemblyAI ---
    ASSEMBLYAI_API_KEY = "4611c26c986948aabeb5d5e13be256e1"  # 🔑 your key
    headers = {"authorization": ASSEMBLYAI_API_KEY}

    print("🎧 Uploading audio to AssemblyAI...")
    upload_url = "https://api.assemblyai.com/v2/upload"
    with open(audio_path, "rb") as f:
        response = requests.post(upload_url, headers=headers, data=f)
    audio_url = response.json()["upload_url"]

    # Request transcription
    json_data = {"audio_url": audio_url, "language_code": "en"}
    transcript_url = "https://api.assemblyai.com/v2/transcript"
    transcribe_req = requests.post(transcript_url, json=json_data, headers=headers)
    transcribe_id = transcribe_req.json()["id"]

    # Poll until transcription is ready
    print("⏳ Waiting for transcription...")
    while True:
        poll = requests.get(f"{transcript_url}/{transcribe_id}", headers=headers)
        status = poll.json()
        if status["status"] == "completed":
            full_text = status["text"]
            print("✅ Transcription complete!")
            break
        elif status["status"] == "error":
            print("❌ Transcription failed:", status["error"])
            exit()
        else:
            time.sleep(5)

    # Split transcript into chunks
    words = full_text.split()
    for i in range(0, len(words), 400):
        chunk = " ".join(words[i:i+400])
        transcript_chunks.append({"text": chunk, "start_time": None, "end_time": None})

    print(f"📝 Extracted {len(transcript_chunks)} chunks from mic input.")

else:
    print("⚠ Invalid choice. Exiting.")
    exit()

# -------------------------------------------------------------
# -------------------------------------------------------------
# 4. Embedding + FAISS Index
# -------------------------------------------------------------
if len(transcript_chunks) == 0:
    print("⚠ No transcript chunks found. Cannot run queries.")
    exit()

texts = [chunk["text"] for chunk in transcript_chunks]
print(f"📚 Total text chunks: {len(texts)}")
print("🧠 Generating embeddings...")

embeddings_list = []
for i, t in enumerate(texts):
    if not t.strip():
        print(f"⚠ Skipping empty chunk #{i}")
        continue
    print(f"🔹 Encoding chunk #{i+1}/{len(texts)} (length: {len(t.split())} words)")
    emb = model.encode(t)
    embeddings_list.append(emb)

if len(embeddings_list) == 0:
    print("❌ No valid text to embed. Exiting.")
    exit()

embeddings = np.array(embeddings_list, dtype='float32')

print("⚙ Building FAISS index...")
dim = embeddings.shape[1]
index = faiss.IndexFlatL2(dim)
index.add(embeddings)
print(f"✅ FAISS index built with {len(embeddings)} entries.")

# -------------------------------------------------------------
# 5. Query loop
# -------------------------------------------------------------
while True:
    query = input("\n🔍 Ask a question (or type 'exit' to quit): ").strip()
    if query.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break

    query_embedding = np.array([model.encode(query)], dtype='float32')
    D, I = index.search(query_embedding, k=3)

    print("\nTop matching context:")
    for idx in I[0]:
        print("•", texts[idx][:300], "...\n")