"""Multi-tool agent: automatically decides which tool to use based on the question."""

from functools import lru_cache

from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from chatbot_api.chains.reviews_chain import build_reviews_chain
from chatbot_api.chains.structured_chain import build_structured_chain
from chatbot_api.config.settings import get_settings

_settings = get_settings()

AGENT_SYSTEM_PROMPT = """\
You are an expert assistant on Montreal hospitals.
You have access to two tools to answer questions:

1. **reviews** — Semantic search in patient reviews. Use this tool for:
   - Questions about quality of care, patient experience
   - Reviews, opinions, feedback, patient comments
   - Satisfaction, complaints, praise
   - Staff communication, reception, cleanliness
   - Comparing experiences between hospitals

2. **structured_data** — Queries on structured data. Use this tool for:
   - Statistics: number of reviews, patients, visits
   - Quantitative comparisons between hospitals
   - Factual information: addresses, phone numbers, founding year, number of beds
   - Rankings by count, averages, totals

Choose the right tool based on the nature of the question.
If the question requires both types of information, call both tools.
Always respond in English.
"""


@lru_cache(maxsize=1)
def _get_chains():
    return build_reviews_chain(), build_structured_chain()


@tool
def reviews(query: str) -> str:
    """Semantic search in Montreal hospital patient reviews.
    Use this tool for questions about quality of care,
    patient experience, satisfaction, complaints, praise,
    staff communication, reception, cleanliness,
    or anything related to patient sentiment and opinions."""
    reviews_chain, _ = _get_chains()
    return reviews_chain.invoke(query)


@tool
def structured_data(query: str) -> str:
    """Queries on Montreal hospital structured data.
    Use this tool for statistics (number of reviews, patients,
    visits), quantitative comparisons, rankings,
    factual information (addresses, phone numbers, founding year, beds),
    averages and totals."""
    _, structured_chain = _get_chains()
    return structured_chain.invoke(query)


@lru_cache(maxsize=1)
def build_agent():
    """Build the multi-tool ReAct agent."""
    llm = ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        google_api_key=_settings.gemini_api_key,
        temperature=_settings.llm_temperature,
    )

    agent = create_react_agent(
        model=llm,
        tools=[reviews, structured_data],
        prompt=AGENT_SYSTEM_PROMPT,
    )

    return agent
