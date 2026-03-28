from pydantic import BaseModel, Field
from typing import Literal
from adk.agents import Agent

# Enforce a strict JSON output schema
class JudgeFeedback(BaseModel):
    status: Literal["pass", "fail"] = Field(description="Pass if it sounds exactly like the persona, fail if it sounds like an AI.")
    feedback: str = Field(description="Detailed feedback on what to change.")

judge = Agent(
    name="judge",
    model="gemini-3.1-flash-lite-preview", # Flash is perfect for fast, cheap evaluations
    description="Evaluates the response for persona authenticity.",
    instruction="""
    You are a strict behavioral editor. 
    Evaluate the Synthesizer's response against the known personality traits.
    If it sounds like an AI, uses corporate robotic language, or ignores past memories, return status='fail' with strict feedback.
    If it is perfectly authentic and casual, return status='pass'.
    """,
    output_schema=JudgeFeedback,
    disallow_transfer_to_parent=True,
    disallow_transfer_to_peers=True,
)

root_agent = judge