
from src.retrieval import KnowledgeRetriever


def main() -> None:
    retriever = KnowledgeRetriever()

    print(f"Indexed chunks: {len(retriever.chunks)}")

    test_queries = [
        "How fast are refunds to the ZentroMart Wallet?",
        "What is the return policy?",
        "How do I claim an assured warranty?",
        "Can I schedule a delivery at midnight?",
    ]

    for query in test_queries:
        print("\n" + "=" * 80)
        print(f"QUERY: {query}")

        results = retriever.retrieve(query, top_k=4)

        if not results:
            print("No matching authoritative chunks found.")
            continue

        for rank, result in enumerate(results, start=1):
            chunk = result.chunk
            print(
                f"\n{rank}. {chunk.section_id} "
                f"| {chunk.source_file} "
                f"| score={result.similarity:.4f}"
            )
            print(
                f"Status: {chunk.status}; "
                f"effective_from: {chunk.effective_from}"
            )
            print(chunk.text[:900])


if __name__ == "__main__":
    main()
