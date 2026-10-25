
from src.rag import answer_with_rag
from src.retrieval import KnowledgeRetriever


TEST_QUESTIONS = [
    "How fast are refunds to the ZentroMart Wallet?",
    "What is the return policy?",
    "How do I claim an assured warranty?",
    "Can I schedule a delivery at midnight?",
]


def main():
    retriever = KnowledgeRetriever()

    print(f"Indexed chunks: {len(retriever.chunks)}")

    for question in TEST_QUESTIONS:
        print("\n" + "=" * 80)
        print(f"QUESTION: {question}")

        result = answer_with_rag(question, retriever=retriever)

        print(f"\nGROUNDED: {result['grounded']}")
        print(f"\nANSWER:\n{result['answer']}")

        print("\nRETRIEVED EVIDENCE:")
        if not result["evidence"]:
            print("No evidence retrieved.")
        else:
            for item in result["evidence"]:
                print(
                    f"- {item['section_id'] or item['title']} "
                    f"| {item['source_file']} "
                    f"| score={item['similarity']}"
                )


if __name__ == "__main__":
    main()
