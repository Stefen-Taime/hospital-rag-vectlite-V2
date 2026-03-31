"""RAG chain for qualitative analysis of patient reviews."""

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

from chatbot_api.agents.retriever import VectLiteRetriever
from chatbot_api.config.settings import get_settings

_settings = get_settings()

REVIEWS_PROMPT = """\
You are an expert assistant for analyzing hospital reviews in Montreal.
Always respond in English, in a clear, structured and factual manner.
Base your answers **only** on the reviews provided in the context below.
If the requested information is not present in the context, state it clearly.
Mention the relevant hospitals and summarize trends when appropriate.

Context:
{context}
"""


def _format_docs(docs: list[Document]) -> str:
    parts = []
    for d in docs:
        header = (
            f"[Hospital: {d.metadata.get('hospital_name', 'N/A')} | "
            f"Score: {d.metadata.get('score', 0):.3f}]"
        )
        parts.append(f"{header}\n{d.page_content}")
    return "\n\n---\n\n".join(parts)


def build_reviews_chain():
    """Build the RAG chain for patient reviews."""
    retriever = VectLiteRetriever()

    llm = ChatGoogleGenerativeAI(
        model=_settings.llm_model,
        google_api_key=_settings.gemini_api_key,
        temperature=_settings.llm_temperature,
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", REVIEWS_PROMPT),
        ("human", "{question}"),
    ])

    chain = (
        {"context": retriever | _format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain
