import os

from dotenv import load_dotenv
from google import genai

load_dotenv()

MODEL_NAME = "gemini-3.5-flash"


def generate_answer(
	question: str,
	retrieved_chunks: list[dict],
) -> str:
	"""Generate a concise answer grounded only in retrieved document chunks."""
	api_key = os.getenv("GEMINI_API_KEY")

	if not api_key:
		raise ValueError(
			"GEMINI_API_KEY is missing. Add it to your .env file before generating answers."
		)

	context = "\n\n".join(
		f"Source: {chunk.get('source', 'Unknown')}\n"
		f"Chunk ID: {chunk.get('chunk_id', 'Unknown')}\n"
		f"Text: {chunk.get('text', '')}"
		for chunk in retrieved_chunks
	)

	prompt = f"""Answer the question using only the supplied context.
Do not invent facts or use outside knowledge.
If the context does not contain enough information, explicitly say:
\"The available documents do not provide enough information to answer this question.\"
Give a concise and clear answer.
The context labels distinguish LOCAL DOCUMENT sources from WEB SOURCE sources.
Clearly distinguish information from local documents and web sources in your answer.
For every WEB SOURCE used, include its supplied URL as a citation.

Context:
---
{context}
---

Question: {question}
"""

	client = genai.Client(api_key=api_key)
	response = client.models.generate_content(
		model=MODEL_NAME,
		contents=prompt,
	)

	if not response.text:
		raise RuntimeError("The Gemini API returned an empty answer.")

	return response.text.strip()
