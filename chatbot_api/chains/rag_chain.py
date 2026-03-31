"""Entry point: multi-tool agent orchestrating reviews + structured data."""

import logging
from functools import lru_cache

from chatbot_api.agents.hospital_agent import build_agent
from chatbot_api.agents.retriever import VectLiteRetriever
from chatbot_api.models.schemas import ChatResponse, SourceDocument

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _get_retriever():
    return VectLiteRetriever()


def _extract_tools_used(messages: list) -> str:
    """Extract the names of tools used from agent messages."""
    tools = []
    for msg in messages:
        if hasattr(msg, 'type') and msg.type == 'tool':
            tools.append(msg.name)
    return ", ".join(dict.fromkeys(tools)) if tools else "agent"


async def ask(question: str) -> ChatResponse:
    """Run the multi-tool agent and return the response."""
    agent = build_agent()

    result = agent.invoke({"messages": [("human", question)]})
    messages = result.get("messages", [])

    # Last message = agent response (content can be str or list of blocks)
    raw_content = messages[-1].content if messages else ""
    if isinstance(raw_content, list):
        answer = "\n".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in raw_content
        )
    else:
        answer = str(raw_content)
    tool_used = _extract_tools_used(messages)
    logger.info("Tools used: %s", tool_used)

    # Retrieve sources if the reviews tool was used
    sources = []
    if "reviews" in tool_used:
        retriever = _get_retriever()
        docs = retriever.invoke(question)
        sources = [
            SourceDocument(
                hospital_name=d.metadata.get("hospital_name", ""),
                score=round(d.metadata.get("score", 0), 4),
                visit_id=d.metadata.get("visit_id", 0),
                date_of_admission=d.metadata.get("date_of_admission", ""),
                review_excerpt=d.metadata.get("review_raw", "")[:200],
            )
            for d in docs
        ]

    return ChatResponse(answer=answer, tool_used=tool_used, sources=sources)
