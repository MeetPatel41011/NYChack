import os
from google.cloud import storage
from moviepy import VideoFileClip

# --- Configuration ---
PROJECT_ID = "tese-491515" # Replace with your project ID
BUCKET_NAME = f"{PROJECT_ID}-digital-twin-datalake"
RAW_DATA_FOLDER = "./raw_persona_data" # Put all your raw files in this local folder

# Define our strict enterprise storage schema
DIRS = {
    "docs": "01_documents/",
    "audio": "02_audio/",
    "visuals": "03_visuals/",
    "index": "04_system_index/" # We will use this in Phase 3
}

def setup_gcs_bucket():
    """Ensures the bucket exists."""
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    if not bucket.exists():
        print(f"Creating bucket: {BUCKET_NAME} in us-central1...")
        bucket = storage_client.create_bucket(bucket, location="us-central1")
    return bucket

def upload_blob(bucket, local_file_path, destination_blob_name):
    """Uploads a file to the bucket."""
    blob = bucket.blob(destination_blob_name)
    blob.upload_from_filename(local_file_path)
    print(f"  -> Uploaded to: gs://{BUCKET_NAME}/{destination_blob_name}")

def extract_audio(video_path, audio_output_path):
    """Strips the audio track from an MP4/MOV for Custom Voice training."""
    try:
        print(f"  -> Extracting audio track from {os.path.basename(video_path)}...")
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_output_path, logger=None)
        video.close()
        return True
    except Exception as e:
        print(f"  -> Audio extraction failed: {e}")
        return False

def smart_route_data():
    """Iterates through raw files, categorizes them, extracts audio, and uploads to GCS."""
    bucket = setup_gcs_bucket()
    
    if not os.path.exists(RAW_DATA_FOLDER):
        os.makedirs(RAW_DATA_FOLDER)
        print(f"Created '{RAW_DATA_FOLDER}'. Please place your raw files there and re-run.")
        return

    print("Initiating Phase 1: Smart Routing & Ingestion...\n")
    
    for filename in os.listdir(RAW_DATA_FOLDER):
        file_path = os.path.join(RAW_DATA_FOLDER, filename)
        if os.path.isdir(file_path):
            continue
            
        ext = filename.split('.')[-1].lower()
        
        # 1. Route Documents & Text (Including PDFs - Gemini will handle internal images later)
        if ext in ['txt', 'jsonl', 'json', 'pdf', 'docx', 'csv']:
            print(f"Processing Document: {filename}")
            upload_blob(bucket, file_path, f"{DIRS['docs']}{filename}")
            
        # 2. Route Raw Audio (Voice notes)
        elif ext in ['ogg', 'opus', 'wav', 'mp3', 'm4a']:
            print(f"Processing Audio Note: {filename}")
            upload_blob(bucket, file_path, f"{DIRS['audio']}{filename}")
            
        # 3. Route Videos (and extract audio for the voice model!)
        elif ext in ['mp4', 'mov', 'avi']:
            print(f"Processing Video: {filename}")
            
            # Upload the visual video file
            upload_blob(bucket, file_path, f"{DIRS['visuals']}{filename}")
            
            # Extract and upload the audio for TTS training
            audio_filename = f"{filename.split('.')[0]}_extracted.wav"
            audio_path = os.path.join(RAW_DATA_FOLDER, audio_filename)
            
            if extract_audio(file_path, audio_path):
                upload_blob(bucket, audio_path, f"{DIRS['audio']}{audio_filename}")
                os.remove(audio_path) # Clean up local audio file after upload
                
        # 4. Route Standard Images
        elif ext in ['jpg', 'jpeg', 'png', 'gif']:
            print(f"Processing Image: {filename}")
            upload_blob(bucket, file_path, f"{DIRS['visuals']}{filename}")
            
        else:
            print(f"Skipping unknown file type: {filename}")

    print("\n✅ Phase 1 Complete! Data lake is perfectly structured and primed for processing.")

if __name__ == "__main__":
    smart_route_data()