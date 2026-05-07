import os
import requests
from openai import OpenAI
from collections.abc import Iterator

from config import (
    LLM_BACKEND,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    OPENAI_COMPATIBLE_API_KEY_ENV,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_MODEL,
    TEMPERATURE,
    TOP_P,
    MAX_OUTPUT_TOKENS,
)

def stream_generate_answer(prompt: str) -> Iterator[str]:
    """Stream an answer using the configured LLM backend.

    Currently streaming is implemented for the OpenAI-compatible backend.
    Other backends fall back to one-shot generation.
    """

    if LLM_BACKEND == "openai_compatible":
        yield from stream_with_openai_compatible(prompt)
        return

    # Fallback for ollama / retrieval_only / unsupported backends.
    answer = generate_answer(prompt)
    if answer:
        yield answer


def stream_with_openai_compatible(prompt: str) -> Iterator[str]:
    """Stream an answer using an OpenAI-compatible chat completions API."""

    api_key = os.getenv(OPENAI_COMPATIBLE_API_KEY_ENV)

    if not api_key:
        raise ValueError(
            f"{OPENAI_COMPATIBLE_API_KEY_ENV} is missing. "
            "Please set it in your .env file or deployment secrets."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_COMPATIBLE_BASE_URL,
    )

    stream = client.chat.completions.create(
        model=OPENAI_COMPATIBLE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a source-grounded product knowledge assistant. "
                    "Answer only using the retrieved context provided in the user prompt."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
        stream=True,
    )

    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta

            
def generate_answer(prompt: str) -> str:
    """Generate an answer using the configured LLM backend."""

    if LLM_BACKEND == "openai_compatible":
        return generate_with_openai_compatible(prompt)

    if LLM_BACKEND == "ollama":
        return generate_with_ollama(prompt)

    if LLM_BACKEND == "retrieval_only":
        return ""

    raise ValueError(f"Unsupported LLM_BACKEND: {LLM_BACKEND}")


def generate_with_openai_compatible(prompt: str) -> str:
    """Generate an answer using an OpenAI-compatible chat completions API."""

    api_key = os.getenv(OPENAI_COMPATIBLE_API_KEY_ENV)

    if not api_key:
        raise ValueError(
            f"{OPENAI_COMPATIBLE_API_KEY_ENV} is missing. "
            "Please set it in your .env file or deployment secrets."
        )

    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_COMPATIBLE_BASE_URL,
    )

    response = client.chat.completions.create(
        model=OPENAI_COMPATIBLE_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a source-grounded product knowledge assistant. "
                    "Answer only using the retrieved context provided in the user prompt."
                ),
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
        max_tokens=MAX_OUTPUT_TOKENS,
    )

    content = response.choices[0].message.content

    if not content:
        raise ValueError("The OpenAI-compatible API returned an empty response.")

    return content.strip()


def generate_with_ollama(prompt: str) -> str:
    """Generate an answer using a local Ollama server."""

    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/generate",
        json={
            "model": OLLAMA_MODEL,
            "prompt": prompt,
            "stream": False,
        },
        timeout=120,
    )

    response.raise_for_status()

    content = response.json().get("response", "")

    if not content:
        raise ValueError("Ollama returned an empty response.")

    return content.strip()


def warm_up_ollama() -> None:
    """Warm up the local Ollama model if Ollama is enabled."""

    if LLM_BACKEND != "ollama":
        return

    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate",
            json={
                "model": OLLAMA_MODEL,
                "prompt": "Reply with exactly one word: ready",
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except Exception as e:
        print(f"Ollama warm-up failed: {e}")