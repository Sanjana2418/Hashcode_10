!pip install ffmpeg-python vosk sentence-transformers faiss-cpu transformers accelerate requests pydub PyPDF2 python-docx

import os, json, torch, ffmpeg, numpy as np, wave, requests, re
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
from google.colab import files
from sentence_transformers import SentenceTransformer
import faiss
from PyPDF2 import PdfReader
import docx

# --- Step 0: Nebius API Key ---
NEBIUS_API_KEY = "v1.CmMKHHN0YXRpY2tleS1lMDBjeDc5Z25lZDdndDlicGYSIXNlcnZpY2VhY2NvdW50LWUwMGpiZ25nZ2tmbTBmYmVocTIMCPKx88cGEMbliZMCOgsI7rSLkwcQwLD6SEACWgNlMDA.AAAAAAAAAAErukSVVLitCcaFMxoS3Lkkixc17mvLhW6V6yCO1xJ9k8bjTsplCdFcPBHFpBDT3TitA5lBrSAD603QBeGbwiMP"

# --- Step 1: Upload video or file ---
uploaded = files.upload()
file_path = list(uploaded.keys())[0]
print(f"Uploaded: {file_path}")

ext = file_path.split(".")[-1].lower()

# --- Step 2: Handle both cases separately ---
transcript_chunks = []
all_words = []

if ext in ["mp4", "mov", "avi", "mkv"]:  
    print("🎥 Detected video file")

    # --- Extract audio ---
    audio_path = "audio.wav"
    ffmpeg.input(file_path).output(audio_path, format='wav', ac=1, ar='16000').run(overwrite_output=True)
    print("✅ Audio extracted")

    # --- Transcribe using Vosk ---
    if not os.path.exists("vosk-model-small-en-us-0.15"):
        !wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
        !unzip -q vosk-model-small-en-us-0.15.zip

    model = Model("vosk-model-small-en-us-0.15")
    wf = wave.open(audio_path, "rb")
    rec = KaldiRecognizer(model, wf.getframerate())
    rec.SetWords(True)

    chunk_size = 400
    buffer_text = ""
    buffer_start_time = 0

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
                    transcript_chunks.append({
                        "text": buffer_text.strip(),
                        "start_time": buffer_start_time,
                        "end_time": buffer_end_time
                    })
                    buffer_text = ""
    wf.close()

    if buffer_text:
        buffer_end_time = all_words[-1]["end"] if all_words else buffer_start_time
        transcript_chunks.append({
            "text": buffer_text.strip(),
            "start_time": buffer_start_time,
            "end_time": buffer_end_time
        })

    print(f"📝 Total chunks: {len(transcript_chunks)}")
    print("Preview of first chunk:\n", transcript_chunks[0])

else:
    print("📄 Detected document file")

    # --- Extract text from supported formats ---
    def extract_text_from_doc(path):
        ext = path.split(".")[-1].lower()
        text = ""
        if ext == "pdf":
            reader = PdfReader(path)
            for page in reader.pages:
                t = page.extract_text()
                if t:
                    text += t + "\n"
        elif ext == "docx":
            doc = docx.Document(path)
            text = "\n".join([p.text for p in doc.paragraphs])
        elif ext == "txt":
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
        return text

    raw_text = extract_text_from_doc(file_path)
    words = raw_text.split()
    chunk_size = 400
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        transcript_chunks.append({"text": chunk, "start_time": None, "end_time": None})

    print(f"📝 Extracted {len(transcript_chunks)} text chunks from file.")
    print("Preview of first chunk:\n", transcript_chunks[0])

# --- Step 4: Embed chunks using Sentence-BERT ---
embedder = SentenceTransformer("all-MiniLM-L6-v2")
texts = [c["text"] for c in transcript_chunks]
embeddings = np.array(embedder.encode(texts, show_progress_bar=True))
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)
print("✅ FAISS index built")

# --- Step 5: Retrieval function with timestamps (same as before) ---
def retrieve(query, top_k=2):
    q_emb = embedder.encode([query])
    _, idxs = index.search(np.array(q_emb), top_k)
    results = []
    for i in idxs[0]:
        chunk = transcript_chunks[i]
        if ext in ["mp4", "mov", "avi", "mkv"]:
            start, end = chunk["start_time"], chunk["end_time"]
            results.append(f"[{start:.1f}s - {end:.1f}s] {chunk['text']}")
        else:
            results.append(chunk["text"])
    return "\n".join(results)

# --- Step 6: Get accurate word-level timestamps (video only) ---
def get_word_timestamp(word):
    if not all_words:
        return ["⛔ Word-level timestamps only available for videos."]
    word_lower = word.lower()
    matches = [w for w in all_words if w["word"].lower() == word_lower]
    if matches:
        return [f"{w['start']:.2f}s" for w in matches]
    else:
        return ["Word not found"]

# --- Step 7: Ask questions using Nebius Chat API ---
def ask_video(question):
    # If 'word' timestamp query and video file
    match = re.findall(r"'(.*?)'", question)
    if match and ext in ["mp4", "mov", "avi", "mkv"]:
        word = match[0]
        times = get_word_timestamp(word)
        print(f"\n⏱ Exact timestamps for '{word}': {', '.join(times)}")
        return

    # Normal semantic QA
    context = retrieve(question)
    url = "https://api.studio.nebius.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {NEBIUS_API_KEY}"}
    data = {
        "model": "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant for answering based on uploaded video or document context."},
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

# --- Step 8: Interactive loop (unchanged) ---
print("🎯 You can now ask multiple questions. Type 'exit' to quit.\n")
while True:
    q = input("❓ Your question: ")
    if q.lower() in ["exit", "quit"]:
        print("👋 Exiting...")
        break
    ask_video(q)