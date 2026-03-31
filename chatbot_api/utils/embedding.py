"""Embedding service via OpenAI."""

from openai import OpenAI

from chatbot_api.config.settings import get_settings

_settings = get_settings()
_client = OpenAI(api_key=_settings.openai_api_key)


def embed_query(text: str) -> list[float]:
    """Generate the embedding for a user query."""
    response = _client.embeddings.create(
        model=_settings.embedding_model,
        input=text,
    )
    return response.data[0].embedding
