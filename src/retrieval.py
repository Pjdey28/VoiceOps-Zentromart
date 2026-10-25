
from dataclasses import dataclass
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.knowledge_base import KnowledgeChunk, load_knowledge_base


@dataclass
class RetrievedChunk:
    chunk: KnowledgeChunk
    similarity: float


class KnowledgeRetriever:
    def __init__(self):
        self.chunks = load_knowledge_base()

        # Prefer section-level evidence over document front matter.
        sectioned_files = {
            chunk.source_file
            for chunk in self.chunks
            if chunk.section_id
        }

        self.chunks = [
            chunk
            for chunk in self.chunks
            if chunk.section_id or chunk.source_file not in sectioned_files
        ]

        if not self.chunks:
            raise ValueError("The knowledge base contains no retrievable chunks.")

        # Include headings and IDs so broad queries can match section titles.
        self.documents = [
            f"{chunk.title}\n{chunk.section_id}\n{chunk.text}"
            for chunk in self.chunks
        ]

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
            strip_accents="unicode",
            stop_words="english",
        )

        self.matrix = self.vectorizer.fit_transform(self.documents)

    @staticmethod
    def _catalog_match_score(query: str, chunk: KnowledgeChunk) -> float:
        """Boost exact catalog identifiers and product-name matches."""
        if chunk.category != "catalog":
            return 0.0

        query_lower = query.lower()
        title_lower = chunk.title.lower()
        sku_lower = chunk.section_id.lower().removeprefix("products.csv:")

        if sku_lower in query_lower:
            return 1.0
        if title_lower in query_lower:
            return 0.95

        title_terms = {
            term for term in re.findall(r"[a-z0-9]+", title_lower)
            if len(term) > 2
        }
        query_terms = set(re.findall(r"[a-z0-9]+", query_lower))
        if not title_terms:
            return 0.0

        coverage = len(title_terms & query_terms) / len(title_terms)
        return 0.8 * coverage if coverage >= 0.5 else 0.0

    def retrieve(
        self,
        query: str,
        top_k: int = 4,
        authoritative_only: bool = True,
    ) -> list[RetrievedChunk]:
        if not query or not query.strip() or top_k <= 0:
            return []

        catalog_candidates = [
            (chunk, self._catalog_match_score(query, chunk))
            for chunk in self.chunks
            if chunk.category == "catalog"
        ]
        catalog_candidates = [
            item for item in catalog_candidates if item[1] >= 0.4
        ]
        sku_matches = {
            match.lower()
            for match in re.findall(r"\b[A-Z]{3}-\d{4}\b", query, re.IGNORECASE)
        }
        if sku_matches:
            catalog_candidates = [
                item
                for item in catalog_candidates
                if item[0].section_id.rsplit(":", 1)[-1].lower() in sku_matches
            ]
        else:
            catalog_candidates.sort(key=lambda item: -item[1])
            catalog_candidates = catalog_candidates[:2]

        expanded_query = query.strip()
        if catalog_candidates:
            expanded_query += " " + " ".join(
                chunk.text for chunk, _ in catalog_candidates
            )

        query_vector = self.vectorizer.transform([expanded_query])
        scores = cosine_similarity(query_vector, self.matrix).ravel()

        results = []

        for index, score in enumerate(scores):
            chunk = self.chunks[index]

            if authoritative_only and not chunk.authoritative:
                continue

            score = max(float(score), self._catalog_match_score(query, chunk))
            if catalog_candidates and chunk.category != "catalog":
                categories = {
                    str(candidate[0].text).split("Return category:", 1)[-1]
                    .splitlines()[0]
                    .strip()
                    .lower()
                    for candidate in catalog_candidates
                }
                query_lower = query.lower()
                policy_terms = (
                    "return", "refund", "warranty", "assured",
                    "claim", "send back",
                )
                if (
                    any(category and category in chunk.text.lower()
                        for category in categories)
                    and any(term in query_lower for term in policy_terms)
                ):
                    score += 0.25

            if score <= 0:
                continue

            results.append(
                RetrievedChunk(
                    chunk=chunk,
                    similarity=float(score),
                )
            )

        results.sort(
            key=lambda result: (
                -result.similarity,
                result.chunk.priority,
                result.chunk.section_id,
            )
        )

        catalog_results = [
            result
            for result in results
            if (
                result.chunk.category == "catalog"
                and self._catalog_match_score(query, result.chunk) >= 0.4
            )
        ]
        if sku_matches:
            catalog_results = [
                result
                for result in catalog_results
                if result.chunk.section_id.rsplit(":", 1)[-1].lower()
                in sku_matches
            ]
        if catalog_results:
            # Product questions need the catalog row and the policy that
            # explains how that row should be interpreted.
            selected_catalog = catalog_results[:2]
            selected_ids = {
                result.chunk.chunk_id for result in selected_catalog
            }
            policy_results = [
                result
                for result in results
                if result.chunk.chunk_id not in selected_ids
                and result.chunk.category != "catalog"
            ]
            return (selected_catalog + policy_results)[:top_k]

        return results[:top_k]
