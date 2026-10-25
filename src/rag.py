
import re

from src.llm import generate_response
from src.retrieval import KnowledgeRetriever


INSUFFICIENT_EVIDENCE = (
    "Sorry, I couldn't find enough information in the available ZentroMart "
    "documents to answer that confidently. I can connect you to a support "
    "colleague who can help."
)

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been",
    "being", "do", "does", "did", "how", "what", "when", "where",
    "which", "who", "whom", "why", "can", "could", "would", "should",
    "i", "me", "my", "we", "our", "you", "your", "it", "its",
    "to", "of", "for", "from", "on", "in", "at", "by", "with",
    "and", "or", "but", "about", "please", "tell", "explain",
    "want", "know", "get", "have", "has", "had", "there",
    "policy", "zentromart",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    }


def _has_sufficient_lexical_evidence(query: str, retrieved) -> bool:
    """
    Conservative first-pass gate:
    require at least 40% of meaningful query terms to occur in the
    retrieved evidence. This is not a semantic relevance guarantee.
    """
    query_terms = _tokens(query)

    if not query_terms:
        return False

    evidence_text = " ".join(
        f"{item.chunk.title} {item.chunk.section_id} {item.chunk.text}"
        for item in retrieved
    )
    evidence_terms = _tokens(evidence_text)

    overlap = query_terms.intersection(evidence_terms)
    coverage = len(overlap) / len(query_terms)

    return coverage >= 0.40



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

    # Broad policy requests need coverage of the relevant policy,
    # rather than only the highest-scoring individual sections.
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
        # Find the strongest current policy document in the results.
        policy_results = [
            item for item in retrieved
            if item.chunk.source_file.lower().startswith("pol_")
        ]

        if policy_results:
            best_policy_file = policy_results[0].chunk.source_file

            # Add all authoritative sections from that policy so the
            # model can explain category-specific rules completely.
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

    if not retrieved or not _has_sufficient_lexical_evidence(question, retrieved):
        return {
            "answer": INSUFFICIENT_EVIDENCE,
            "evidence": evidence,
            "grounded": False,
        }

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

Use only the supplied policy and catalog evidence to answer the customer.

Rules:
- Never invent policy details, timelines, amounts, eligibility, or capabilities.
- If evidence does not answer the question, say so and offer support escalation.
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
- If the evidence is insufficient, do not pretend the answer is supported.
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
        "grounded": True,
    }
