import os
import json
from google.cloud import storage
import vertexai
from vertexai.generative_models import GenerativeModel, Part
from vertexai.vision_models import MultiModalEmbeddingModel, Video, Image as VertexImage

# --- Configuration ---
PROJECT_ID = "tese-491515" 
LOCATION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-digital-twin-datalake"

# Output files for Phase 3
DOCS_OUTPUT = "processed_docs.jsonl"
AUDIO_OUTPUT = "processed_audio.jsonl"
VISUALS_OUTPUT = "processed_visuals.jsonl"

vertexai.init(project=PROJECT_ID, location=LOCATION)
storage_client = storage.Client(project=PROJECT_ID)

def get_blobs_from_folder(folder_name):
    """Helper to get all files from a specific GCS folder."""
    bucket = storage_client.bucket(BUCKET_NAME)
    return [blob for blob in bucket.list_blobs(prefix=folder_name) if not blob.name.endswith('/')]

def clean_json_string(raw_string):
    """Removes markdown formatting to ensure JSON is parseable."""
    return raw_string.strip().lstrip('```json').rstrip('```').strip()

# ---------------------------------------------------------
# PIPELINE A: The Document & PDF Pipeline (Graph Extraction)
# ---------------------------------------------------------
def process_documents():
    print("\n--- Starting Document Pipeline ---")
    model = GenerativeModel("gemini-2.5-pro")
    blobs = get_blobs_from_folder("01_documents/")
    
    extraction_prompt = """
    Analyze this document. Extract key behavioral traits, facts, and preferences 
    of the person as a JSON list of triplets: 
    [{"subject": "Persona", "predicate": "...", "object": "..."}]
    Return ONLY valid JSON.
    """
    
    with open(DOCS_OUTPUT, 'w') as f:
        for blob in blobs:
            print(f"Extracting Graph Knowledge from: {blob.name}")
            gcs_uri = f"gs://{BUCKET_NAME}/{blob.name}"
            document_part = Part.from_uri(uri=gcs_uri, mime_type=blob.content_type)
            
            try:
                response = model.generate_content([document_part, extraction_prompt])
                # CRITICAL FIX: Clean the response before parsing
                json_str = clean_json_string(response.text)
                triplets = json.loads(json_str)
                
                for triplet in triplets:
                    graph_sentence = f"{triplet['subject']} {triplet['predicate']} {triplet['object']}"
                    record = {
                        "id": f"graph_{blob.name.replace('/', '_')}",
                        "text": graph_sentence,
                        "type": "graph_knowledge",
                        "source_uri": gcs_uri
                    }
                    f.write(json.dumps(record) + '\n')
            except Exception as e:
                print(f"Skipping {blob.name} due to error: {e}")

# ---------------------------------------------------------
# PIPELINE B: The Audio Pipeline (Speech-to-Text via Gemini)
# ---------------------------------------------------------
def process_audio():
    print("\n--- Starting Audio Pipeline ---")
    model = GenerativeModel("gemini-2.5-pro")
    blobs = get_blobs_from_folder("02_audio/")
    
    transcription_prompt = "Provide a highly accurate, word-for-word transcription of this audio note. Do not add any summary or extra text."
    
    with open(AUDIO_OUTPUT, 'w') as f:
        for blob in blobs:
            if "extracted.wav" in blob.name:
                continue
                
            print(f"Transcribing Audio: {blob.name}")
            gcs_uri = f"gs://{BUCKET_NAME}/{blob.name}"
            # Ensure we are passing the correct mime_type for MP3/WAV
            audio_part = Part.from_uri(uri=gcs_uri, mime_type=blob.content_type)
            
            try:
                response = model.generate_content([audio_part, transcription_prompt])
                record = {
                    "id": f"audio_{blob.name.replace('/', '_')}",
                    "text": response.text.strip(),
                    "type": "raw_memory",
                    "source_uri": gcs_uri
                }
                f.write(json.dumps(record) + '\n')
            except Exception as e:
                print(f"Failed to transcribe {blob.name}: {e}")

# ---------------------------------------------------------
# PIPELINE C: The Visual Pipeline (Multimodal Embeddings)
# ---------------------------------------------------------
def process_visuals():
    print("\n--- Starting Visual Pipeline ---")
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    blobs = get_blobs_from_folder("03_visuals/")
    
    if not blobs:
        print("No visual files found in GCS. Skipping visual pipeline.")
        return

    with open(VISUALS_OUTPUT, 'w') as f:
        for blob in blobs:
            print(f"Generating Vector Embedding for: {blob.name}")
            local_filename = f"/tmp/{os.path.basename(blob.name)}"
            blob.download_to_filename(local_filename)
            gcs_uri = f"gs://{BUCKET_NAME}/{blob.name}"
            
            try:
                embedding_vector = None
                if local_filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image = VertexImage.load_from_file(local_filename)
                    embedding_vector = model.get_embeddings(image=image).image_embedding
                elif local_filename.lower().endswith(('.mp4', '.avi', '.mov')):
                    video = Video.load_from_file(local_filename)
                    embedding_vector = model.get_embeddings(video=video).video_embeddings[0].embedding
                
                if embedding_vector:
                    record = {
                        "id": f"visual_{blob.name.replace('/', '_')}",
                        "embedding": embedding_vector,
                        "type": "visual_memory",
                        "source_uri": gcs_uri
                    }
                    f.write(json.dumps(record) + '\n')
            except Exception as e:
                print(f"Failed to embed {blob.name}: {e}")
            finally:
                if os.path.exists(local_filename):
                    os.remove(local_filename)

if __name__ == "__main__":
    print("Initiating Phase 2 Data Pipelines...")
    process_documents()
    process_audio()
    process_visuals()
    print("\n✅ Phase 2 Complete! All files processed and .jsonl files generated.")