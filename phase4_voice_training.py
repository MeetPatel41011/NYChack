import os
import requests
from google.cloud import storage
from dotenv import load_dotenv

load_dotenv()

# --- Config ---
PROJECT_ID = "tese-491515"
BUCKET_NAME = f"{PROJECT_ID}-digital-twin-datalake"
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

# Update this to match the exact name of an extracted WAV file in your GCS bucket
TARGET_AUDIO_FILE = "02_audio/RPReplay_Final1774720401_extracted.wav" 
LOCAL_TEMP = "training_audio.wav"

def train_custom_voice():
    print(f"📥 Pulling training data from gs://{BUCKET_NAME}/{TARGET_AUDIO_FILE}...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(TARGET_AUDIO_FILE)
    blob.download_to_filename(LOCAL_TEMP)

    print("🧬 Sending to ElevenLabs for Instant Cloning...")
    url = "https://api.elevenlabs.io/v1/voices/add"
    headers = {"xi-api-key": ELEVENLABS_API_KEY}
    data = {
        "name": "Meet Digital Twin",
        "description": "Cloned directly from GCP Data Lake"
    }
    
    with open(LOCAL_TEMP, "rb") as f:
        files = [("files", (LOCAL_TEMP, f, "audio/wav"))]
        response = requests.post(url, headers=headers, data=data, files=files)
        
    if response.status_code == 200:
        voice_id = response.json().get("voice_id")
        print(f"\n✅ SUCCESS! YOUR VOICE ID: {voice_id}")
    else:
        print(f"❌ Error: {response.text}")

    if os.path.exists(LOCAL_TEMP):
        os.remove(LOCAL_TEMP)

if __name__ == "__main__":
    train_custom_voice()