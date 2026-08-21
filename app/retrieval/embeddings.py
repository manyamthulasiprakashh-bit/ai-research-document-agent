import os

from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

MODEL_NAME = "gemini-embedding-001"


def _create_embedding(text: str, task_type: str) -> list[float]:
    """Create an embedding for text using the requested task type."""
    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        raise ValueError(
            "GEMINI_API_KEY is missing. Add it to your .env file before creating embeddings."
        )

    client = genai.Client(api_key=api_key)
    response = client.models.embed_content(
        model=MODEL_NAME,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )

    if not response.embeddings or not response.embeddings[0].values:
        raise RuntimeError("The embedding API returned no embedding values.")

    return list(response.embeddings[0].values)


def create_embedding(text: str) -> list[float]:
    """Create a document embedding for a text chunk."""
    return _create_embedding(text, "RETRIEVAL_DOCUMENT")


def create_query_embedding(text: str) -> list[float]:
    """Create a query embedding for a user search query."""
    return _create_embedding(text, "RETRIEVAL_QUERY")
