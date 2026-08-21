import os
from urllib.parse import urlparse

import requests
from dotenv import load_dotenv

load_dotenv()

TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class WebSearchError(RuntimeError):
    """Raised when a web search cannot be completed."""


class WebSearcher:
    """Collect structured web sources from the Tavily search API."""

    def __init__(
        self,
        api_key: str | None = None,
        endpoint: str = TAVILY_SEARCH_URL,
    ) -> None:
        """Initialize the searcher using a key from the environment by default."""
        self._api_key = api_key or os.getenv("TAVILY_API_KEY")
        self._endpoint = endpoint

    def search(
        self,
        query: str,
        max_results: int = 5,
    ) -> list[dict[str, str]]:
        """Return structured web sources for a query."""
        if not query.strip():
            raise ValueError("query cannot be empty")

        if max_results <= 0:
            raise ValueError("max_results must be greater than 0")

        if not self._api_key:
            raise WebSearchError(
                "TAVILY_API_KEY is missing. Add it to your .env file before searching."
            )

        payload = {
            "api_key": self._api_key,
            "query": query,
            "max_results": max_results,
            "search_depth": "basic",
            "include_answer": False,
        }

        try:
            response = requests.post(
                self._endpoint,
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
        except requests.RequestException as error:
            raise WebSearchError("The web search provider request failed.") from error
        except ValueError as error:
            raise WebSearchError("The web search provider returned invalid JSON.") from error

        provider_results = data.get("results", [])
        if not isinstance(provider_results, list):
            raise WebSearchError("The web search provider returned an invalid result list.")

        structured_results: list[dict[str, str]] = []
        for result in provider_results[:max_results]:
            if not isinstance(result, dict):
                continue

            url = str(result.get("url", "")).strip()
            domain = urlparse(url).netloc.removeprefix("www.")
            structured_results.append(
                {
                    "title": str(result.get("title", "")).strip(),
                    "url": url,
                    "snippet": str(
                        result.get("snippet", result.get("content", ""))
                    ).strip(),
                    "domain": domain,
                }
            )

        return structured_results


def main() -> None:
    """Run a small web-source collection demonstration."""
    query = "What is Retrieval-Augmented Generation?"
    searcher = WebSearcher()

    try:
        results = searcher.search(query)
    except WebSearchError as error:
        print(f"Web search error: {error}")
        return

    print("----- WEB RESEARCH -----")
    print(f"\nQuery:\n{query}")

    for index, result in enumerate(results, start=1):
        print(f"\n----- SOURCE {index} -----")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Domain: {result['domain']}")
        print(f"Snippet: {result['snippet']}")


if __name__ == "__main__":
    main()
