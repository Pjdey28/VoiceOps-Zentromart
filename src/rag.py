import re
from src.llm import generate_response
from src.retrieval import KnowledgeRetriever

def answer_with_rag(
    question: str,
    retriever: KnowledgeRetriever | None = None,
    top_k: int = 5,
) -> dict:
    if not question or not question.strip():
        return {
            "answer": "Please tell me what you need help with.",
            "evidence": [],
            "grounded": False,
        }

    if retriever is None:
        retriever = KnowledgeRetriever()

    question = question.strip()
    query_lower = question.lower()

    broad_policy_query = (
        "policy" in query_lower
        and any(term in query_lower for term in (
            "return", "refund", "delivery", "cancellation",
            "cancel", "warranty", "payment",
        ))
    )

    retrieved = retriever.retrieve(
        question,
        top_k=max(top_k, 10) if broad_policy_query else top_k,
        authoritative_only=True,
    )

    if broad_policy_query and retrieved:
        policy_results = [
            item for item in retrieved
            if item.chunk.source_file.lower().startswith("pol_")
        ]

        if policy_results:
            best_policy_file = policy_results[0].chunk.source_file
            existing_ids = {item.chunk.chunk_id for item in retrieved}

            for chunk in retriever.chunks:
                if (
                    chunk.source_file == best_policy_file
                    and chunk.authoritative
                    and chunk.chunk_id not in existing_ids
                ):
                    from src.retrieval import RetrievedChunk
                    retrieved.append(
                        RetrievedChunk(chunk=chunk, similarity=0.0)
                    )
                    existing_ids.add(chunk.chunk_id)

    evidence = [
        {
            "section_id": item.chunk.section_id,
            "source_file": item.chunk.source_file,
            "title": item.chunk.title,
            "similarity": round(item.similarity, 4),
            "text": item.chunk.text,
        }
        for item in retrieved
    ]

    context = "\n\n".join(
        (
            f"[Evidence: {item.chunk.section_id or item.chunk.title}; "
            f"Source: {item.chunk.source_file}]\n"
            f"{item.chunk.text}"
        )
        for item in retrieved
    )

    system_prompt = """
You are the customer-support assistant for ZentroMart.

Today's date is 20 October 2026.
Business days run Mon-Sat, excluding holidays.csv.

Use only the supplied policy and catalog evidence to answer the customer.

Rules:
- Never invent policy details, timelines, amounts, eligibility, or capabilities.
- If the supplied evidence is completely insufficient to answer the question, do not invent an answer. Instead, politely explain that you don't have enough information in the ZentroMart documents and offer to connect them to a human support agent.
- For broad policy questions, summarize all relevant rules present in the
  supplied evidence, including category-specific exceptions and date conditions.
- Do not claim that a section states a rule it does not actually contain.
- Current authoritative policy documents take precedence over conflicting FAQs.
- Use catalog evidence for product-specific facts such as SKU, category, seller,
  Assured status, and brand warranty. Combine it with the relevant policy
  section before answering return or warranty questions.
- Do not assume marketplace seller items are ZentroMart-sold items.
- Keep the response concise, clear, and polite.
- Do not reveal order-specific information without verification.
- Do not claim that you performed an action or created a ticket.
- Cite claims with section IDs in a final Sources line.
"""
    user_prompt = f"""
Customer question:
{question}

Retrieved policy and catalog evidence:
{context}

Answer the customer using the evidence above.
"""

    answer = generate_response(
        [
            {"role": "system", "content": system_prompt.strip()},
            {"role": "user", "content": user_prompt.strip()},
        ],
        temperature=0.0,
    )

    return {
        "answer": answer.strip(),
        "evidence": evidence,
        "grounded": bool(retrieved), # Grounded status relies on whether chunks exist
    }