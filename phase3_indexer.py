import json
import os
import vertexai
from google.cloud import storage
from vertexai.vision_models import MultiModalEmbeddingModel

# --- Configuration ---
PROJECT_ID = "tese-491515"  # Replace with your project ID
LOCATION = "us-central1"
BUCKET_NAME = f"{PROJECT_ID}-digital-twin-datalake"
INDEX_OUTPUT_FILE = "hybrid_index.json"

vertexai.init(project=PROJECT_ID, location=LOCATION)

def generate_hybrid_index():
    print("Initiating Phase 3: Building the Unified Memory Bank...\n")
    
    # We use the Multimodal model for EVERYTHING to maintain a strict 1408-dimension size
    print("Loading Multimodal Embedding Model...")
    model = MultiModalEmbeddingModel.from_pretrained("multimodalembedding@001")
    
    formatted_records = []

    # 1. Embed the Graph Knowledge (From Docs/PDFs)
    if os.path.exists("processed_docs.jsonl"):
        print("Embedding Graph Knowledge...")
        with open("processed_docs.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                # Pass the text through the multimodal model
               
                vector = model.get_embeddings(contextual_text=data['text']).text_embedding
                
                formatted_records.append({
                    "id": data['id'],
                    "embedding": vector,
                    # Metadata filtering tag!
                    "restricts": [{"namespace": "memory_type", "allow": ["graph_knowledge"]}], 
                    "source_uri": data['source_uri'] # Optional: Keep track of original file
                })

    # 2. Embed the Audio Transcripts
    if os.path.exists("processed_audio.jsonl"):
        print("Embedding Audio Transcripts...")
        with open("processed_audio.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                
                vector = model.get_embeddings(contextual_text=data['text']).text_embedding
                
                formatted_records.append({
                    "id": data['id'],
                    "embedding": vector,
                    "restricts": [{"namespace": "memory_type", "allow": ["audio_transcript"]}],
                    "source_uri": data['source_uri']
                })

    # 3. Format the Visual Embeddings (Already vectorized in Phase 2)
    if os.path.exists("processed_visuals.jsonl"):
        print("Formatting Visual Memory...")
        with open("processed_visuals.jsonl", 'r') as f:
            for line in f:
                data = json.loads(line)
                
                formatted_records.append({
                    "id": data['id'],
                    "embedding": data['embedding'], # Directly porting the 1408-dim vector
                    "restricts": [{"namespace": "memory_type", "allow": ["visual_memory"]}],
                    "source_uri": data['source_uri']
                })

    # 4. Save the Master JSON file
    print(f"\nWriting {len(formatted_records)} total memories to {INDEX_OUTPUT_FILE}...")
    with open(INDEX_OUTPUT_FILE, 'w') as out:
        for record in formatted_records:
            # Vertex AI Vector Search strictly requires newline-delimited JSON
            out.write(json.dumps(record) + '\n')
            
    # 5. Upload to the correct Data Lake folder
    print("Uploading Master Index to Google Cloud Storage...")
    storage_client = storage.Client(project=PROJECT_ID)
    bucket = storage_client.bucket(BUCKET_NAME)
    blob = bucket.blob(f"04_system_index/{INDEX_OUTPUT_FILE}")
    blob.upload_from_filename(INDEX_OUTPUT_FILE)
    
    print(f"✅ Phase 3 Complete! Index uploaded to gs://{BUCKET_NAME}/04_system_index/")

if __name__ == "__main__":
    generate_hybrid_index()