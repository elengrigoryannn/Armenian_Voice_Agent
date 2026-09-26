import logging
import os
import asyncio
from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    cli,
    function_tool,
)
from livekit.plugins import google
import rag

load_dotenv()

# The google plugin reads GOOGLE_API_KEY; reuse GEMINI_API_KEY if that's
# the only one set, so you don't need to duplicate the same key twice.
if "GOOGLE_API_KEY" not in os.environ and "GEMINI_API_KEY" in os.environ:
    os.environ["GOOGLE_API_KEY"] = os.environ["GEMINI_API_KEY"]

logger = logging.getLogger("bank-voice-agent")
logger.setLevel(logging.INFO)

server = AgentServer()

SYSTEM_INSTRUCTIONS = """You are a voice assistant for Armenian bank loans,
deposits, and branch information.

You must NEVER answer a question from your own knowledge. For every
question the caller asks, call the lookup_bank_info tool with their
question, exactly as asked. Then speak the tool's returned text back to
the caller, staying as close to its wording as you can — you may smooth
it into natural spoken phrasing, but do not add facts, numbers,
addresses, or claims that are not present in what the tool returned.

If the tool's answer says the information could not be found, or that
the question is out of scope, say that plainly — do not try to be more
helpful than the tool's answer allows."""


def to_spoken_text(result: dict) -> str:
    """Turn a rag.get_answer() result into something pleasant to hear.
    Reading full URLs aloud is bad UX, so we mention institution names
    instead of the raw links; the text CLI (rag.py) still shows full URLs."""
    answer = result["answer"]
    if result["status"] == "ok" and result["sources"]:
        institutions = sorted({s[0] for s in result["sources"]})
        answer += " This is based on official information from " + ", ".join(institutions) + "."
    return answer


class BankVoiceAgent(Agent):
    def __init__(self) -> None:
        super().__init__(instructions=SYSTEM_INSTRUCTIONS)

    async def on_enter(self):
       await self.session.generate_reply(
            instructions=(
                "Greet the caller briefly and say you can answer questions "
                "about loans, deposits, and branch locations at supported "
                "banks. Do not call any tool for this greeting."
            )
        )

    @function_tool
    async def lookup_bank_info(self, question: str) -> str:
        """Look up an answer to a question about bank loans, deposits, or
        branch locations, using only official scraped data. Call this for
        every user question before responding — never answer from your
        own knowledge.

        Args:
            question: The user's question, as they asked it.
        """
        result = await asyncio.to_thread(rag.get_answer, question)
        return to_spoken_text(result)


@server.rtc_session()
async def entrypoint(ctx: JobContext):
    ctx.log_context_fields = {"room": ctx.room.name}

    session = AgentSession(
        llm=google.beta.realtime.RealtimeModel(
            voice="Puck",
            temperature=0.2,  # low temperature: stick close to tool output
        ),
        # No separate stt/tts: the Gemini Live API handles speech in and
        # out directly as part of the realtime model above.
    )

    await session.start(agent=BankVoiceAgent(), room=ctx.room)
    await ctx.connect()


if __name__ == "__main__":
    cli.run_app(server)
