import os
from adk.agents import RemoteA2aAgent, BaseAgent, LoopAgent, SequentialAgent, EventActions
from adk.context import InvocationContext
from adk.callbacks import create_save_output_callback

# 1. Connect to the sub-agents (URLs will be provided by Cloud Run in Phase 5)
retriever = RemoteA2aAgent(
    name="retriever", 
    agent_card=os.environ.get("RETRIEVER_URL", "http://localhost:8001/a2a/agent/.well-known/agent-card.json"),
    after_agent_callback=create_save_output_callback("retrieved_context")
)
synthesizer = RemoteA2aAgent(
    name="synthesizer", 
    agent_card=os.environ.get("SYNTHESIZER_URL", "http://localhost:8002/a2a/agent/.well-known/agent-card.json")
)
judge = RemoteA2aAgent(
    name="judge", 
    agent_card=os.environ.get("JUDGE_URL", "http://localhost:8003/a2a/agent/.well-known/agent-card.json"),
    after_agent_callback=create_save_output_callback("judge_feedback")
)

# 2. Control Flow: Break the loop if the Judge approves
class EscalationChecker(BaseAgent):
    """Breaks the synthesis loop if the Judge passes the response."""
    async def _run_async_impl(self, ctx: InvocationContext):
        feedback = ctx.session.state.get("judge_feedback")
        if isinstance(feedback, dict) and feedback.get("status") == "pass":
            yield EventActions(escalate=True)

# 3. The Self-Correcting Loop
synthesis_loop = LoopAgent(
    name="synthesis_loop",
    agents=[synthesizer, judge, EscalationChecker()],
    max_iterations=3 # Prevent infinite loops to save API costs
)

# 4. The Master Pipeline
master_pipeline = SequentialAgent(
    name="digital_twin_pipeline",
    agents=[retriever, synthesis_loop]
)

root_agent = master_pipeline