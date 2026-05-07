import json
import re
import shutil
from typing import List

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

from config import (
    RAW_DOCS_DIR,
    PROCESSED_DIR,
    CHUNKS_PATH,
    VECTORSTORE_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    CHROMA_DISTANCE_METRIC,
)


# -----------------------------------------------------------------------------
# Cleaning
# -----------------------------------------------------------------------------

def remove_markup_noise(text: str) -> str:
    """Remove documentation markup noise while preserving useful technical content.

    This removes MkDocs/Jinja/HTML wrapper noise, but keeps useful technical
    snippets such as environment variables, YAML, JSON, shell commands,
    Docker commands, and API examples.
    """

    # Remove YAML frontmatter at the beginning of Markdown files.
    text = re.sub(r"^---\s*\n.*?\n---\s*\n", "", text, flags=re.DOTALL)

    # Remove fenced code blocks that only contain HTML/template wrapper remnants.
    # Keep useful code blocks such as YAML, JSON, shell commands, env variables.
    text = re.sub(
        r"(?is)```(?:html|xml|markdown)?\s*(?:</?(?:div|span|section|figure|figcaption)[^>]*>\s*)+```",
        "",
        text,
    )

    # Remove MkDocs/Jinja template blocks and lines.
    text = re.sub(r"\[\[%.*?%\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"\[\[.*?\]\]", "", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s*(\[%|{%|{{).*?(\]%|%}|}})?\s*$", "", text)

    # Remove MkDocs snippet includes: --8<-- "path"
    text = re.sub(r'--8<--\s*["\'].*?["\']', "", text)

    # Convert admonition blocks like "/// note | Title" to plain text title.
    text = re.sub(
        r"///\s*(note|info|warning|danger|tip|important)\s*\|\s*(.*)",
        r"\2",
        text,
    )
    text = re.sub(r"///\s*(note|info|warning|danger|tip|important).*", "", text)
    text = re.sub(r"///", "", text)

    # Remove standalone HTML wrapper lines.
    text = re.sub(
        r"(?im)^\s*</?(div|figure|figcaption|span|section|iframe|video|source|img)[^>]*>\s*$",
        "",
        text,
    )

    # Remove inline HTML wrapper tags but keep inner text where possible.
    text = re.sub(r"<br\s*/?>", "\n", text)
    text = re.sub(
        r"</?(div|figure|figcaption|span|section|iframe|video|source|img)[^>]*>",
        "",
        text,
    )

    # Remove icon shortcodes such as :material-arrow-right:.
    text = re.sub(r":[a-zA-Z0-9_-]+:", "", text)

    # Remove image links: ![alt](url)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", "", text)

    # Convert Markdown links [text](url) -> text.
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)

    return text

def remove_markdown_emphasis(text: str) -> str:
    """Remove Markdown emphasis markers without damaging technical tokens.

    Important:
    Do not remove underscores inside identifiers such as DB_TYPE,
    DB_POSTGRESDB_HOST, N8N_ENCRYPTION_KEY, or snake_case names.
    """

    # Bold with asterisks: **text**
    text = re.sub(r"\*\*([^*\n]+)\*\*", r"\1", text)

    # Bold with underscores: __text__
    # Only remove when the underscores are not part of a word/token.
    text = re.sub(r"(?<![A-Za-z0-9])__([^_\n]+)__(?![A-Za-z0-9])", r"\1", text)

    # Italic with underscores: _text_
    # Only remove when the underscores are not inside technical identifiers.
    text = re.sub(r"(?<![A-Za-z0-9])_([^_\n]+)_(?![A-Za-z0-9])", r"\1", text)

    return text

def normalize_whitespace_preserving_code_blocks(text: str) -> str:
    """Normalize whitespace outside fenced code blocks while preserving code indentation."""

    lines = text.splitlines()
    normalized_lines = []
    inside_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            normalized_lines.append(line.rstrip())
            continue

        if inside_code_block:
            # Preserve leading spaces inside code blocks.
            normalized_lines.append(line.rstrip())
        else:
            # Normalize repeated spaces in normal prose only.
            normalized_lines.append(re.sub(r"[ \t]+", " ", line).strip())

    normalized_text = "\n".join(normalized_lines)

    # Normalize excessive blank lines.
    normalized_text = re.sub(r"\n{3,}", "\n\n", normalized_text)

    return normalized_text.strip()

def clean_markdown(text: str) -> str:
    """Clean Markdown before splitting while preserving useful technical content."""

    text = remove_markup_noise(text)

    # Remove Markdown emphasis markers while keeping the text.
    text = remove_markdown_emphasis(text)

    # Normalize whitespace without damaging code block indentation.
    text = normalize_whitespace_preserving_code_blocks(text)

    return text

def clean_chunk_content(text: str) -> str:
    """Clean chunk text after Markdown header splitting."""

    text = remove_markup_noise(text)

    # Convert Markdown headings to plain text:
    # "### What you will learn" -> "What you will learn"
    text = re.sub(r"^(#{1,6})\s+", "", text, flags=re.MULTILINE)

    # Remove Markdown emphasis markers while keeping the text.
    text = remove_markdown_emphasis(text)

    # Normalize whitespace without damaging code block indentation.
    text = normalize_whitespace_preserving_code_blocks(text)

    return text

def extract_title(text: str, fallback: str) -> str:
    """Extract the first H1 or H2 title from a Markdown document."""

    for line in text.splitlines():
        line = line.strip()

        if line.startswith("# "):
            return line.replace("# ", "", 1).strip()

        if line.startswith("## "):
            return line.replace("## ", "", 1).strip()

    return fallback


# -----------------------------------------------------------------------------
# Semantic header embedding
# -----------------------------------------------------------------------------

def build_heading_path(metadata: dict) -> str:
    """Build a readable heading path from chunk metadata."""

    return " > ".join(
        h
        for h in [
            metadata.get("h1", ""),
            metadata.get("h2", ""),
            metadata.get("h3", ""),
        ]
        if h
    )


def build_embedding_text(content: str, metadata: dict) -> str:
    """Build enriched text used for embedding.

    The embedding text includes lightweight structural metadata. This helps the
    retriever match user questions to the right document topic, not only to the
    raw chunk body.

    The clean chunk body is still stored separately as metadata["display_content"]
    for LLM context and UI display.
    """

    title = metadata.get("title", "")
    category = metadata.get("category", "")
    source = metadata.get("source", "")
    heading_path = build_heading_path(metadata)

    parts = []

    if title:
        parts.append(f"Title: {title}")

    if heading_path:
        parts.append(f"Section: {heading_path}")

    if category:
        parts.append(f"Category: {category}")

    if source:
        parts.append(f"Path: {source}")

    parts.append(f"Content:\n{content}")

    return "\n".join(parts).strip()


# -----------------------------------------------------------------------------
# Loading and chunking
# -----------------------------------------------------------------------------

def load_markdown_documents() -> List[Document]:
    """Load selected n8n Markdown documents into LangChain Document objects."""

    if not RAW_DOCS_DIR.exists():
        raise FileNotFoundError(f"Raw data directory not found: {RAW_DOCS_DIR}")

    documents: List[Document] = []
    md_files = sorted(
        list(RAW_DOCS_DIR.rglob("*.md")) + list(RAW_DOCS_DIR.rglob("*.mdx"))
    )

    print(f"Found {len(md_files)} Markdown files.")

    for path in md_files:
        raw_text = path.read_text(encoding="utf-8", errors="ignore")
        clean_text = clean_markdown(raw_text)

        if len(clean_text) < 200:
            continue

        relative_path = path.relative_to(RAW_DOCS_DIR)
        category = relative_path.parts[0] if len(relative_path.parts) > 1 else "general"
        title = extract_title(clean_text, fallback=path.stem.replace("-", " ").title())

        documents.append(
            Document(
                page_content=clean_text,
                metadata={
                    "source": relative_path.as_posix(),
                    "category": category,
                    "title": title,
                    "doc_type": "n8n_product_documentation",
                },
            )
        )

    print(f"Loaded {len(documents)} documents after cleaning.")
    return documents


def split_documents_into_clean_chunks(documents: List[Document]) -> List[Document]:
    """Split Markdown documents into section-aware clean chunks.

    Strategy:
    - First split by Markdown headings h1/h2/h3.
    - Keep short sections as complete chunks.
    - Only recursively split sections that are too long.
    """

    MAX_SECTION_CHARS = 1800
    CHUNK_SIZE = 1400
    CHUNK_OVERLAP = 120
    MIN_CHUNK_CHARS = 80

    headers_to_split_on = [
        ("#", "h1"),
        ("##", "h2"),
        ("###", "h3"),
    ]

    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False,
    )

    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=[
            "\n\n",
            "\n",
            ". ",
            " ",
            "",
        ],
    )

    clean_chunks: List[Document] = []

    for doc in documents:
        sections = markdown_splitter.split_text(doc.page_content)

        for section in sections:
            merged_metadata = {
                **doc.metadata,
                **section.metadata,
            }

            clean_content = clean_chunk_content(section.page_content)

            if len(clean_content) < MIN_CHUNK_CHARS:
                continue

            # Keep short sections intact.
            if len(clean_content) <= MAX_SECTION_CHARS:
                clean_chunks.append(
                    Document(
                        page_content=clean_content,
                        metadata=merged_metadata,
                    )
                )
                continue

            # Only split long sections.
            long_section_doc = Document(
                page_content=clean_content,
                metadata=merged_metadata,
            )

            split_chunks = recursive_splitter.split_documents([long_section_doc])

            for split_chunk in split_chunks:
                split_content = clean_chunk_content(split_chunk.page_content)

                if len(split_content) < MIN_CHUNK_CHARS:
                    continue

                clean_chunks.append(
                    Document(
                        page_content=split_content,
                        metadata=dict(split_chunk.metadata),
                    )
                )

    for i, chunk in enumerate(clean_chunks):
        chunk.metadata["chunk_id"] = i

    return clean_chunks

def build_embedding_documents(clean_chunks: List[Document]) -> List[Document]:
    """Build documents used for vectorstore indexing.

    page_content is enriched for embedding.
    metadata["display_content"] keeps the clean chunk for generation/UI.
    """

    embedding_documents: List[Document] = []

    for chunk in clean_chunks:
        display_content = chunk.page_content
        metadata = dict(chunk.metadata)

        embedding_text = build_embedding_text(
            content=display_content,
            metadata=metadata,
        )

        metadata["display_content"] = display_content
        metadata["embedding_text_version"] = "semantic_header_section_v1"

        embedding_documents.append(
            Document(
                page_content=embedding_text,
                metadata=metadata,
            )
        )

    return embedding_documents


# -----------------------------------------------------------------------------
# Inspection output
# -----------------------------------------------------------------------------

def save_chunks_to_json(chunks: List[Document]) -> None:
    """Save cleaned and chunked documents for inspection and reproducibility."""

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    records = []

    for chunk in chunks:
        records.append(
            {
                "id": chunk.metadata.get("chunk_id"),
                "display_content": chunk.metadata.get("display_content", chunk.page_content),
                "embedding_text": chunk.page_content,
                "metadata": chunk.metadata,
            }
        )

    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved processed chunks to: {CHUNKS_PATH}")


# -----------------------------------------------------------------------------
# Vectorstore build
# -----------------------------------------------------------------------------

def build_vectorstore() -> None:
    """Build and persist a local Chroma vectorstore."""

    documents = load_markdown_documents()

    if not documents:
        raise ValueError("No documents were loaded. Please check the data directory.")

    clean_chunks = split_documents_into_clean_chunks(documents)
    embedding_documents = build_embedding_documents(clean_chunks)

    print(f"Created {len(embedding_documents)} chunks.")

    save_chunks_to_json(embedding_documents)

    # Rebuild the vectorstore from scratch each time.
    if VECTORSTORE_DIR.exists():
        shutil.rmtree(VECTORSTORE_DIR)

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    Chroma.from_documents(
        documents=embedding_documents,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(VECTORSTORE_DIR),
        collection_metadata={"hnsw:space": CHROMA_DISTANCE_METRIC},
    )

    print(f"Vectorstore saved to: {VECTORSTORE_DIR}")


if __name__ == "__main__":
    build_vectorstore()