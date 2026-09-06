"""
preprocess_documents.py

Converts raw source documents in knowledge_base/documents/raw/ into cleaned
plain-text files in knowledge_base/documents/processed/, ready for Layer 2
chunking (build_layer2_index.py).

Scope note (dissertation-level):
    - Extension-based auto-routing only (.pdf, .md, .html/.htm)
    - No OCR — source PDFs are assumed to be text-based (official vendor docs)
    - No layout-recovery beyond basic multi-column heuristics in pdfplumber
    - Advanced ingestion (OCR, scanned docs, DOCX, structured web crawling)
      is out of scope here and is noted as future work for a production
      version of this framework.

Usage:
    python preprocess_documents.py
    python preprocess_documents.py --input-dir ../documents/raw --output-dir ../documents/processed
    python preprocess_documents.py --verbose
"""

import argparse
import re
import sys
from pathlib import Path
from collections import Counter


# --------------------------------------------------------------------------
# Format-specific extractors
# --------------------------------------------------------------------------

def extract_pdf(path: Path, verbose: bool = False) -> str:
    """Extract raw text from a PDF using pdfplumber, page by page."""
    try:
        import pdfplumber
    except ImportError:
        print("ERROR: pdfplumber is required for PDF extraction. "
              "Install with: pip install pdfplumber")
        sys.exit(1)

    pages_text = []
    with pdfplumber.open(str(path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            pages_text.append(text)
            if verbose:
                print(f"    Page {i + 1}/{len(pdf.pages)}: "
                      f"{len(text)} chars extracted")

    return _remove_repeated_headers_footers(pages_text)


def extract_markdown(path: Path, verbose: bool = False) -> str:
    """Strip markdown syntax down to plain readable text."""
    raw = path.read_text(encoding="utf-8", errors="ignore")

    text = raw
    # Fenced code blocks -> drop (implementation detail, not prose knowledge)
    text = re.sub(r"```.*?```", "", text, flags=re.DOTALL)
    # Inline code
    text = re.sub(r"`([^`]*)`", r"\1", text)
    # Images ![alt](url) -> drop entirely
    text = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", text)
    # Links [text](url) -> keep text only
    text = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", text)
    # Headers (#, ##, ###...) -> strip leading hashes, keep text
    text = re.sub(r"^#{1,6}\s*", "", text, flags=re.MULTILINE)
    # Bold / italic markers
    text = re.sub(r"(\*\*\*|\*\*|\*|___|__|_)", "", text)
    # Bullet / numbered list markers
    text = re.sub(r"^\s*[-*+]\s+", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*\d+\.\s+", "", text, flags=re.MULTILINE)
    # Blockquote markers
    text = re.sub(r"^\s*>\s?", "", text, flags=re.MULTILINE)
    # Horizontal rules
    text = re.sub(r"^\s*[-*_]{3,}\s*$", "", text, flags=re.MULTILINE)
    # Table separator rows, e.g. "|---|---|" or "| :--- | ---: |" -> drop
    text = re.sub(r"^\s*\|?[\s:|-]+\|[\s:|-]*$", "", text, flags=re.MULTILINE)
    # Table pipes -> spaces
    text = re.sub(r"\|", " ", text)

    return text


def extract_html(path: Path, verbose: bool = False) -> str:
    """Strip HTML tags down to visible text using BeautifulSoup."""
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        print("ERROR: beautifulsoup4 is required for HTML extraction. "
              "Install with: pip install beautifulsoup4")
        sys.exit(1)

    raw = path.read_text(encoding="utf-8", errors="ignore")
    soup = BeautifulSoup(raw, "html.parser")

    # Drop non-content tags entirely
    for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
        tag.decompose()

    return soup.get_text(separator="\n")


# --------------------------------------------------------------------------
# Common cleaning (applied after extraction, regardless of source format)
# --------------------------------------------------------------------------

def _remove_repeated_headers_footers(pages_text, min_repeats=3):
    """
    PDF-specific: lines that repeat identically across many pages are almost
    always headers/footers/page numbers, not content. Drop them.
    Only meaningful when there are multiple pages; for single-page PDFs this
    is a no-op.

    KNOWN LIMITATION: this is a frequency heuristic, not true layout
    detection. On short technical documents where a genuine content line
    happens to repeat verbatim across most pages (e.g. a repeated code
    snippet or defined term), it can be misclassified as boilerplate and
    removed. Spot-check processed output against the source for documents
    where this matters. A production version would use position-on-page
    (e.g. top/bottom N% of page) rather than pure repetition frequency.
    """
    if len(pages_text) < min_repeats:
        return "\n".join(pages_text)

    line_counts = Counter()
    per_page_lines = []
    for page in pages_text:
        lines = [ln.strip() for ln in page.split("\n") if ln.strip()]
        per_page_lines.append(lines)
        for ln in set(lines):  # count once per page, not per occurrence
            line_counts[ln] += 1

    threshold = max(min_repeats, int(len(pages_text) * 0.5))
    boilerplate = {ln for ln, count in line_counts.items() if count >= threshold}

    cleaned_pages = []
    for lines in per_page_lines:
        kept = [ln for ln in lines if ln not in boilerplate]
        cleaned_pages.append("\n".join(kept))

    return "\n".join(cleaned_pages)


def clean_text(text: str) -> str:
    """Common normalization applied to all extracted text before saving."""
    # De-hyphenate words broken across lines: "optimi-\nzation" -> "optimization"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Normalize whitespace within lines
    text = re.sub(r"[ \t]+", " ", text)

    # Collapse 3+ blank lines into a single blank line
    text = re.sub(r"\n\s*\n\s*\n+", "\n\n", text)

    # Strip leading/trailing whitespace per line
    lines = [ln.strip() for ln in text.split("\n")]
    text = "\n".join(lines)

    # Drop lines that are just page numbers or near-empty fragments
    lines = [ln for ln in text.split("\n") if not re.fullmatch(r"\d{1,4}", ln)]
    text = "\n".join(lines)

    return text.strip()


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------

EXTRACTORS = {
    ".pdf": extract_pdf,
    ".md": extract_markdown,
    ".html": extract_html,
    ".htm": extract_html,
}


def process_documents(input_dir: Path, output_dir: Path, verbose: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    source_files = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() in EXTRACTORS
    )

    skipped = sorted(
        f for f in input_dir.iterdir()
        if f.is_file() and f.suffix.lower() not in EXTRACTORS
    )

    print(f"[1/3] Scanning {input_dir} ...")
    print(f"      Found {len(source_files)} supported document(s) "
          f"({', '.join(sorted(EXTRACTORS.keys()))})")
    if skipped:
        print(f"      Skipped {len(skipped)} unsupported file(s): "
              f"{', '.join(f.name for f in skipped)}")

    if not source_files:
        print("      No supported documents found. Nothing to do.")
        return []

    print(f"[2/3] Extracting and cleaning text ...")
    results = []
    for f in source_files:
        ext = f.suffix.lower()
        extractor = EXTRACTORS[ext]
        print(f"    - {f.name} [{ext}]")

        try:
            raw_text = extractor(f, verbose=verbose)
            cleaned = clean_text(raw_text)
        except Exception as e:
            print(f"      FAILED: {e}")
            results.append((f.name, "FAILED", 0))
            continue

        out_path = output_dir / (f.stem + ".txt")
        out_path.write_text(cleaned, encoding="utf-8")

        word_count = len(cleaned.split())
        print(f"      -> {out_path.name} ({word_count} words)")
        results.append((f.name, "OK", word_count))

        if word_count < 50:
            print(f"      WARNING: very low word count — check extraction "
                  f"quality manually for this file.")

    print(f"[3/3] Done. {output_dir} now has "
          f"{len(list(output_dir.glob('*.txt')))} processed file(s).")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Convert raw source documents (PDF/MD/HTML) into cleaned "
                     "plain text for Layer 2 KB chunking."
    )
    parser.add_argument(
        "--input-dir", type=str, default="../documents/raw",
        help="Directory containing raw source documents (default: ../documents/raw)"
    )
    parser.add_argument(
        "--output-dir", type=str, default="../documents/processed",
        help="Directory to write cleaned .txt files (default: ../documents/processed)"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-page extraction detail for PDFs"
    )
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"ERROR: input directory not found: {input_dir}")
        sys.exit(1)

    results = process_documents(input_dir, output_dir, verbose=args.verbose)

    failed = [r for r in results if r[1] == "FAILED"]
    if failed:
        print(f"\n{len(failed)} file(s) failed extraction: "
              f"{', '.join(r[0] for r in failed)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
