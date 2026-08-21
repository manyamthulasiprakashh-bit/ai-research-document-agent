from app.ingestion.document_manager import process_documents
from app.research.research_router import ResearchRouter
from app.retrieval.vector_store import VectorStore


def main():
    vector_store = VectorStore()
    processed_documents = process_documents(
        "data/documents",
        vector_store,
        chunk_size=30,
        overlap=5,
    )
    print(f"Processed documents: {', '.join(processed_documents)}")

    router = ResearchRouter(vector_store)
    questions = [
        "What is Retrieval-Augmented Generation?",
        "What are the latest developments in AI agents?",
        "What does my document say about RAG?",
    ]

    for question in questions:
        result = router.research_question(question)

        print("\n----- QUESTION -----")
        print(result.question)
        print("\n----- ROUTING DECISION -----")
        print(result.routing_decision)
        print("\n----- RETRIEVED LOCAL SOURCES -----")
        if result.local_results:
            for local_result in result.local_results:
                print(
                    f"{local_result['source']} | "
                    f"chunk={local_result['chunk_id']} | "
                    f"score={float(local_result['score']):.6f}"
                )
        else:
            print("None")
        print(f"\n----- WEB SEARCH USED: {result.web_search_used} -----")
        print("----- RETRIEVED WEB SOURCES -----")
        if result.web_results:
            for web_result in result.web_results:
                print(
                    f"{web_result['domain']} | "
                    f"{web_result['url']} | "
                    f"{web_result['title']}"
                )
        else:
            print("None")
        print("\n----- GENERATED ANSWER -----")
        print(result.answer)
        if result.errors:
            print("\n----- ROUTING ERRORS -----")
            for error in result.errors:
                print(error)
        print("\n----- SOURCE CITATIONS -----")
        citation_index = 1
        for local_result in result.local_results:
            print(
                f"{citation_index}. LOCAL DOCUMENT: {local_result['source']} | "
                f"chunk={local_result['chunk_id']} | "
                f"score={float(local_result['score']):.6f}"
            )
            citation_index += 1
        for web_result in result.web_results:
            print(
                f"{citation_index}. WEB SOURCE: {web_result['domain']} | "
                f"{web_result['url']}"
            )
            citation_index += 1


if __name__ == "__main__":
    main()