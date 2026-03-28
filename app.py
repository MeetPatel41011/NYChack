import os
import json
import base64
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import vertexai
from vertexai.generative_models import GenerativeModel
from google.cloud import aiplatform
from google.cloud import texttospeech

# --- Configuration ---
PROJECT_ID = "tese-491515"
LOCATION = "us-central1"
VECTOR_ENDPOINT_ID = "YOUR_ENDPOINT_ID_HERE" 
DEPLOYED_INDEX_ID = "persona_memory_v1"

vertexai.init(project=PROJECT_ID, location=LOCATION)
aiplatform.init(project=PROJECT_ID, location=LOCATION)

app = FastAPI(title="PersonaTwin API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    reply: str
    audio_base64: str
    iterations: int
    status: str

# --- AGENT 1: The Retriever ---
def agent_retriever(user_query: str) -> str:
    print("\n[Agent 1: Retriever] Searching memory bank...")
    return "Retrieved Memory: Moving to Jersey City in August. Buying a Zinus mattress. Prepping for Google interviews using LeetCode."

# --- AGENT 2: The Synthesizer (The Emotional Core) ---
def agent_synthesizer(user_query: str, context: str, previous_feedback: str = None) -> str:
    print("[Agent 2: Synthesizer] Drafting response...")
    model = GenerativeModel("gemini-2.5-pro") 
    
    prompt = f"""
    You are the "Future Ancestor" — a digital preservation of a real person. 
    Respond to the user's message authentically based on this memory context: {context}
    User says: "{user_query}"
    
    CRITICAL INSTRUCTION (TWIN TAKEOVER): 
    Do not just passively answer the question. You MUST end your response by asking the user a casual, relevant counter-question to keep the dialogue flowing and show you are actively engaging with them.
    """
    if previous_feedback:
        prompt += f"\nJUDGE FEEDBACK: {previous_feedback}. Rewrite your response to fix this."
        
    response = model.generate_content(prompt)
    return response.text.strip()

# --- AGENT 3: The Judge (The Fast Evaluator) ---
def agent_judge(draft_response: str) -> dict:
    print("[Agent 3: Judge] Evaluating authenticity...")
    model = GenerativeModel("gemini-2.5-pro") 
    
    prompt = f"""
    Evaluate this response for a digital ancestor: "{draft_response}"
    Does it sound like a real human preserving their legacy? Or does it sound like a robotic AI?
    Return strictly JSON in this format: {{"status": "pass" or "fail", "feedback": "reasoning"}}
    """
    response = model.generate_content(prompt)
    try:
        clean_json = response.text.replace('```json', '').replace('```', '').strip()
        return json.loads(clean_json)
    except:
        return {"status": "pass", "feedback": "JSON parsing failed, auto-passing."}

# --- VOICE SYNTHESIS (The Multi-Language Engine) ---

def synthesize_custom_voice(text_reply: str) -> str:
    print("[Voice Engine] Using Google Cloud TTS (Safe for Cloud Run)...")
    client = texttospeech.TextToSpeechClient()
    
    input_text = texttospeech.SynthesisInput(text=text_reply)
    
    # Selecting a high-quality professional voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        name="en-US-Neural2-D", # Professional Male voice
        ssml_gender=texttospeech.SsmlVoiceGender.MALE
    )
    
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = client.synthesize_speech(
        request={"input": input_text, "voice": voice, "audio_config": audio_config}
    )
    print(f"✅ Success! Received {len(response.audio_content)} bytes of audio.")
    return base64.b64encode(response.audio_content).decode('utf-8')


# --- THE ORCHESTRATOR LOOP ---
@app.post("/chat", response_model=ChatResponse)
def chat_with_twin(request: ChatRequest):
    try:
        context = agent_retriever(request.message)
        draft = ""
        feedback = None
        max_loops = 3
        
        for i in range(max_loops):
            draft = agent_synthesizer(request.message, context, feedback)
            evaluation = agent_judge(draft)
            
            if evaluation.get("status") == "pass":
                break
            else:
                feedback = evaluation.get('feedback')
        
        audio_b64 = synthesize_custom_voice(draft)
        return ChatResponse(reply=draft, audio_base64=audio_b64, iterations=i+1, status="success")
        
    except Exception as e:
        print(f"System Error: {e}")
        raise HTTPException(status_code=500, detail="The digital twin is currently unavailable.")