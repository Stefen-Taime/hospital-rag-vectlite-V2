"""HTTP client for communicating with chatbot_api."""

import httpx

from chatbot_front.config.settings import settings

_TIMEOUT = httpx.Timeout(120.0, connect=10.0)


def chat(question: str) -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.post(
            f"{settings.api_base_url}/chat",
            json={"question": question},
        )
        resp.raise_for_status()
        return resp.json()


def get_stats() -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{settings.api_base_url}/stats")
        resp.raise_for_status()
        return resp.json()


def health_check() -> dict:
    with httpx.Client(timeout=_TIMEOUT) as client:
        resp = client.get(f"{settings.api_base_url}/health")
        resp.raise_for_status()
        return resp.json()
