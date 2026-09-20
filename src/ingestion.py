from pathlib import Path
import re
from typing import Any

import pymupdf


def clean_text(text: str) -> str:
    """Normalize extracted PDF text without removing meaningful content."""
    cleaned_lines = []

    for line in text.splitlines():
        line = re.sub(r"[ \t]+", " ", line).strip()

        if line:
            cleaned_lines.append(line)

    cleaned_text = "\n".join(cleaned_lines)
    cleaned_text = re.sub(r"\n{3,}", "\n\n", cleaned_text)

    return cleaned_text.strip()


def extract_pdf_pages(pdf_path: str | Path) -> list[dict[str, Any]]:
    """
    Extract text page by page.

    Each returned item contains:
    - text
    - metadata.source
    - metadata.page
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    pages = []

    with pymupdf.open(pdf_path) as pdf_document:
        for page_index, page in enumerate(pdf_document):
            raw_text = page.get_text("text")
            cleaned_text = clean_text(raw_text)

            if not cleaned_text:
                continue

            pages.append(
                {
                    "text": cleaned_text,
                    "metadata": {
                        "source": pdf_path.name,
                        "page": page_index + 1,
                    },
                }
            )

    return pages


if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent
    pdf_path = project_root / "data" / "Agent-as-Judge.pdf"

    extracted_pages = extract_pdf_pages(pdf_path)

    print(f"Extracted non-empty pages: {len(extracted_pages)}")

    if extracted_pages:
        first_page = extracted_pages[0]

        print(f"First page source: {first_page['metadata']['source']}")
        print(f"First page number: {first_page['metadata']['page']}")
        print("\nFirst page preview:\n")
        print(first_page["text"][:1000])