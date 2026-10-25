
from dataclasses import dataclass
import csv
from pathlib import Path
import re

from src.config import DATASET_DIR


@dataclass
class KnowledgeChunk:
    chunk_id: str
    section_id: str
    source_file: str
    title: str
    category: str
    status: str
    effective_from: str
    text: str
    authoritative: bool
    priority: int


def _metadata_value(text: str, key: str, default: str = "") -> str:
    pattern = rf"(?im)^\s*{re.escape(key)}\s*:\s*(.*?)\s*$"
    match = re.search(pattern, text)
    return match.group(1).strip() if match else default


def _priority(filename: str, category: str, status: str) -> int:
    name = filename.lower()
    category = category.lower()
    status = status.lower()

    if "superseded" in status or "superseded" in name:
        return 99
    if name.startswith("pol_") or category == "policy":
        return 0
    if "guide" in name or category == "guide":
        return 1
    if "help" in name or category == "help":
        return 2
    if "faq" in name or category == "faq":
        return 3
    if "marketing" in name or category == "marketing":
        return 4
    return 5


def _split_sections(content: str):
    """
    Split Markdown documents at headings such as:
    ## [REF-1] When a refund is initiated
    """
    pattern = re.compile(
        r"(?m)^#{1,6}\s+\[([A-Z][A-Z0-9-]*\d+)\]\s*(.*?)\s*$"
    )
    matches = list(pattern.finditer(content))

    if not matches:
        # Retain documents without explicit section IDs.
        return [("", "", content.strip())] if content.strip() else []

    sections = []

    # Preserve any introductory text before the first numbered section.
    preamble = content[:matches[0].start()].strip()
    if preamble:
        sections.append(("", "Document overview", preamble))

    for index, match in enumerate(matches):
        start = match.start()
        end = (
            matches[index + 1].start()
            if index + 1 < len(matches)
            else len(content)
        )

        section_id = match.group(1).strip()
        section_title = match.group(2).strip()
        section_text = content[start:end].strip()

        if section_text:
            sections.append((section_id, section_title, section_text))

    return sections

def _load_holidays_chunks() -> list[KnowledgeChunk]:
    """Load holiday dates as authoritative evidence for timeline calculations."""
    holidays_path = DATASET_DIR / "data" / "holidays.csv"
    if not holidays_path.is_file():
        return []

    text = "ZentroMart Official Holidays:\n"
    with holidays_path.open("r", encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            date = str(row.get("date", "")).strip()
            name = str(row.get("name", "")).strip()
            if date:
                text += f"- {date}: {name}\n"

    return [
        KnowledgeChunk(
            chunk_id="holidays.csv",
            section_id="holidays",
            source_file="holidays.csv",
            title="Official Holidays",
            category="calendar",
            status="current",
            effective_from="2026-10-20",
            text=text,
            authoritative=True,
            priority=1,
        )
    ]

def load_knowledge_base() -> list[KnowledgeChunk]:
    kb_dir = DATASET_DIR / "kb"

    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge-base directory not found: {kb_dir}")

    chunks = []

    for path in sorted(kb_dir.glob("*.md")):
        content = path.read_text(encoding="utf-8-sig").strip()
        if not content:
            continue

        title = _metadata_value(content, "title", path.stem)
        category = _metadata_value(content, "doc_type", "unknown")
        status = _metadata_value(content, "status", "unknown")
        effective_from = _metadata_value(content, "effective_from", "")

        authoritative = (
            status.lower() not in {"superseded", "obsolete", "retired"}
            and "superseded" not in path.stem.lower()
        )

        priority = _priority(path.name, category, status)

        for section_id, section_title, section_text in _split_sections(content):
            # Prefer the explicit section heading as the chunk's title.
            chunk_title = section_title or title
            chunk_id = (
                f"{path.stem}:{section_id}"
                if section_id
                else f"{path.stem}:overview"
            )

            chunks.append(
                KnowledgeChunk(
                    chunk_id=chunk_id,
                    section_id=section_id,
                    source_file=path.name,
                    title=chunk_title,
                    category=category,
                    status=status,
                    effective_from=effective_from,
                    text=section_text,
                    authoritative=authoritative,
                    priority=priority,
                )
            )

    if not chunks:
        raise ValueError(f"No Markdown knowledge chunks found in {kb_dir}")

    chunks.extend(_load_product_chunks())
    chunks.extend(_load_holidays_chunks())
    return chunks


def _load_product_chunks() -> list[KnowledgeChunk]:
    """Load catalog rows as authoritative evidence for product-specific answers."""
    products_path = DATASET_DIR / "data" / "products.csv"
    if not products_path.is_file():
        raise FileNotFoundError(f"Product catalog not found: {products_path}")

    chunks = []
    with products_path.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as file:
        for row in csv.DictReader(file):
            sku = str(row.get("sku", "")).strip()
            title = str(row.get("title", "")).strip()
            if not sku or not title:
                continue

            text = (
                f"SKU: {sku}\n"
                f"Product: {title}\n"
                f"Brand: {row.get('brand', '').strip()}\n"
                f"Return category: {row.get('return_category', '').strip()}\n"
                f"Price (INR): {row.get('price_inr', '').strip()}\n"
                f"Fulfilled by: {row.get('fulfilled_by', '').strip()}\n"
                f"Assured: {row.get('assured', '').strip()}\n"
                f"Brand warranty (months): {row.get('warranty_months', '').strip()}\n"
                f"Description: {row.get('description', '').strip()}"
            )
            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"products.csv:{sku}",
                    section_id=f"products.csv:{sku}",
                    source_file="products.csv",
                    title=title,
                    category="catalog",
                    status="current",
                    effective_from="2026-10-20",
                    text=text,
                    authoritative=True,
                    priority=1,
                )
            )

    if not chunks:
        raise ValueError(f"No product records found in {products_path}")

    return chunks
