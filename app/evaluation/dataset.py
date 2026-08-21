EVALUATION_DATASET: list[dict[str, object]] = [
    {
        "question": "What is Retrieval-Augmented Generation?",
        "expected_source": "research_notes.txt",
        "expected_keywords": ["retrieval", "generation"],
    },
    {
        "question": "How does a research assistant use retrieved passages?",
        "expected_source": "research_notes.txt",
        "expected_keywords": ["research", "passages"],
    },
    {
        "question": "Why are source citations useful?",
        "expected_source": "research_notes.txt",
        "expected_keywords": ["source", "citations"],
    },
    {
        "question": "What happens during document ingestion?",
        "expected_source": "document_workflows.txt",
        "expected_keywords": ["ingestion", "loading", "text"],
    },
    {
        "question": "Why is text cleaned before chunking?",
        "expected_source": "document_workflows.txt",
        "expected_keywords": ["cleaned", "chunks"],
    },
    {
        "question": "Why are document chunks represented by embeddings?",
        "expected_source": "document_workflows.txt",
        "expected_keywords": ["chunks", "embedding"],
    },
    {
        "question": "Why should the original filename be kept with a chunk?",
        "expected_source": "document_workflows.txt",
        "expected_keywords": ["filename", "traceable"],
    },
    {
        "question": "What can an AI research assistant analyze?",
        "expected_source": "sample.txt",
        "expected_keywords": ["research", "documents"],
    },
    {
        "question": "What does RAG allow an AI system to do before generation?",
        "expected_source": "sample.txt",
        "expected_keywords": ["retrieve", "information", "generating"],
    },
    {
        "question": "How does RAG support answers based on trusted sources?",
        "expected_source": "sample.txt",
        "expected_keywords": ["retrieve", "trusted", "sources"],
    },
]
