import html
import re
import threading
import time
from pathlib import Path
from typing import Any

import streamlit as st

from config import (
    CONTEXT_MAX_CHARS,
    DENSE_WEIGHT,
    DISPLAY_SOURCE_K,
    FINAL_RETRIEVAL_K,
    INITIAL_RETRIEVAL_K,
    LLM_BACKEND,
    OLLAMA_MODEL,
    OPENAI_COMPATIBLE_MODEL,
    SPARSE_WEIGHT,
    TEMPERATURE,
    TOP_P,
    ENABLE_STREAMING,
)

from rag_pipeline import (
    get_rag_response, 
    warm_up_ollama,
    prepare_rag_context,
    stream_rag_answer,
    clean_generated_answer,
    normalize_no_answer_response,
)


@st.cache_resource
def start_vectorstore_warmup() -> bool:
    """Start vectorstore loading in a background thread."""

    def _warmup() -> None:
        try:
            from retrieval.dense_retriever import load_vectorstore

            load_vectorstore()
            print("[warmup] vectorstore loaded")
        except Exception as e:
            print(f"[warmup] vectorstore warm-up failed: {e}")

    thread = threading.Thread(
        target=_warmup,
        daemon=True,
    )
    thread.start()

    return True

def run_rag_query_stream(query: str) -> dict:
    """Run retrieval first, then stream the generated answer."""

    with st.status(
        "Searching n8n documentation and ranking evidence...",
        expanded=False,
    ) as status:
        result = prepare_rag_context(
            query=query,
            k=FINAL_RETRIEVAL_K,
            chat_history=st.session_state.get("messages", []),
        )

        if result.get("is_direct_response") or result.get("mode") == "retrieval_only":
            status.update(label="Answer ready.", state="complete", expanded=False)
            return result

        status.update(label="Sources selected. Generating answer...", state="running")

    prompt = result.get("prompt")

    if not prompt:
        return result

    chunks: list[str] = []

    answer_col, source_col = st.columns([0.58, 0.42], gap="large")

    with source_col:
        render_sources_panel(
            response=result,
            panel_key=f"pending_{len(st.session_state.get('messages', []))}",
        )

    with answer_col:
        def answer_stream():
            for chunk in stream_rag_answer(prompt):
                chunks.append(chunk)
                yield chunk

        streamed_answer = st.write_stream(answer_stream)

    if isinstance(streamed_answer, str) and streamed_answer.strip():
        final_answer = streamed_answer
    else:
        final_answer = "".join(chunks)

    final_answer = clean_generated_answer(final_answer)
    final_answer = normalize_no_answer_response(final_answer)

    result["answer"] = final_answer

    return result

def run_rag_query(query: str) -> dict[str, Any]:
    """Unified RAG entry point for the Streamlit UI."""

    if ENABLE_STREAMING:
        return run_rag_query_stream(query)

    return get_rag_response(
        query=query,
        k=FINAL_RETRIEVAL_K,
        chat_history=st.session_state.get("messages", []),
    )
# -----------------------------------------------------------------------------
# Page configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="n8n Docs Assistant",
    page_icon="◼",
    layout="wide",
    initial_sidebar_state="expanded",
)

start_vectorstore_warmup()

# -----------------------------------------------------------------------------
# CSS
# -----------------------------------------------------------------------------


def load_css(css_path: str = "assets/styles.css") -> None:
    """Load custom CSS from an external stylesheet."""

    css_file = Path(css_path)

    if not css_file.exists():
        st.warning(f"CSS file not found: {css_path}")
        return

    st.markdown(
        f"<style>{css_file.read_text(encoding='utf-8')}</style>",
        unsafe_allow_html=True,
    )


load_css()


# -----------------------------------------------------------------------------
# Background model warm-up
# -----------------------------------------------------------------------------

def start_background_warmup() -> None:
    """Start local Ollama warm-up once per Streamlit session."""

    if LLM_BACKEND != "ollama":
        return

    if st.session_state.get("ollama_warmup_started", False):
        return

    st.session_state["ollama_warmup_started"] = True

    thread = threading.Thread(target=warm_up_ollama, daemon=True)
    thread.start()


start_background_warmup()


# -----------------------------------------------------------------------------
# Session state
# -----------------------------------------------------------------------------
if "pending_query" not in st.session_state:
    st.session_state["pending_query"] = None

if "messages" not in st.session_state:
    st.session_state["messages"] = []

if "selected_source" not in st.session_state:
    st.session_state["selected_source"] = None

if "example_widget_version" not in st.session_state:
    st.session_state["example_widget_version"] = 0

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------


def source_sort_key(source: dict[str, Any]) -> tuple[int, float]:
    """Sort sources by retrieval rank first, then by score if available."""

    source_number = int(source.get("source_number", 9999))
    score = float(source.get("hybrid_score", 0.0))

    # source_number already reflects final ranking order in your pipeline.
    # Lower source_number = better rank.
    return (source_number, -score)


def select_display_sources(
    query: str,
    sources: list[dict[str, Any]],
    max_sources: int = DISPLAY_SOURCE_K,
) -> list[dict[str, Any]]:
    """Select the sources shown in the right panel.

    The pipeline already selects the final sources for generation. The UI only
    limits how many to display and avoids showing too many chunks from the same
    file.
    """

    if not sources:
        return []

    ranked_sources = sorted(sources, key=source_sort_key)

    selected: list[dict[str, Any]] = []
    file_counts: dict[str, int] = {}

    for source in ranked_sources:
        file_path = source.get("source", "Unknown file")

        if file_counts.get(file_path, 0) >= 2:
            continue

        selected.append(source)
        file_counts[file_path] = file_counts.get(file_path, 0) + 1

        if len(selected) >= max_sources:
            break

    return selected

def should_show_welcome() -> bool:
    """Show welcome/examples only before the first question is submitted."""

    return (
        len(st.session_state.get("messages", [])) == 0
        and st.session_state.get("pending_query") is None
    )


def get_latest_assistant_response() -> dict[str, Any] | None:
    """Return the latest assistant response from chat history."""

    for message in reversed(st.session_state.get("messages", [])):
        if message.get("role") == "assistant":
            return message.get("response", {})

    return None

def convert_markdown_tables_to_bullets(text: str) -> str:
    """Convert simple Markdown tables into compact bullet-style evidence.

    This is used only for the source preview panel. It makes table-heavy
    documentation easier to read in the narrow right-hand column.
    """

    if not text:
        return ""

    lines = text.splitlines()
    output_lines: list[str] = []

    i = 0

    while i < len(lines):
        line = lines[i]

        # Detect Markdown table header + separator.
        if (
            i + 1 < len(lines)
            and line.strip().startswith("|")
            and line.strip().endswith("|")
            and lines[i + 1].strip().startswith("|")
            and re.fullmatch(r"\s*\|[\s:|\-]+\|\s*", lines[i + 1])
        ):
            header_cells = [
                cell.strip()
                for cell in line.strip().strip("|").split("|")
            ]

            i += 2

            table_rows: list[list[str]] = []

            while (
                i < len(lines)
                and lines[i].strip().startswith("|")
                and lines[i].strip().endswith("|")
            ):
                row_cells = [
                    cell.strip()
                    for cell in lines[i].strip().strip("|").split("|")
                ]

                if len(row_cells) == len(header_cells):
                    table_rows.append(row_cells)

                i += 1

            for row_cells in table_rows:
                row = dict(zip(header_cells, row_cells))

                variable = (
                    row.get("Variable")
                    or row.get("Name")
                    or row_cells[0]
                )

                description = (
                    row.get("Description")
                    or row.get("Details")
                    or ""
                )

                variable = clean_table_cell_text(variable)
                description = clean_table_cell_text(description)

                extra_parts = []

                for key in ["Type", "Default"]:
                    value = row.get(key)
                    if value:
                        value = clean_table_cell_text(value)
                        if value and value != "-":
                            extra_parts.append(f"{key}: {value}")
                        elif value == "-":
                            extra_parts.append(f"{key}: -")

                if description:
                    bullet = f"- {variable}: {description}"
                else:
                    bullet = f"- {variable}"

                if extra_parts:
                    bullet += f" ({'; '.join(extra_parts)})"

                output_lines.append(bullet)

            continue

        output_lines.append(line)
        i += 1

    return "\n".join(output_lines)

def clean_markdown_table_fragments(text: str) -> str:
    """Convert broken Markdown table fragments into readable bullet points.

    This handles table rows that lost their header or were broken by line breaks,
    which can happen after documentation preprocessing.
    """

    if not text:
        return ""

    # Repair common broken table line breaks inside cells.
    text = re.sub(r"(`[^`\n]+`)\s*\n\s*/\s*(`_FILE`)", r"\1 / \2", text)
    text = re.sub(r"Enum string:\s*\n\s*", "Enum string: ", text)

    lines = text.splitlines()
    output_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Convert variable table rows such as:
        # | `DB_TYPE` / `_FILE` | Enum string: `sqlite`, `postgresdb` | `sqlite` | The database to use. |
        if stripped.startswith("|") and stripped.endswith("|") and "DB_" in stripped:
            cells = [
                clean_table_cell_text(cell)
                for cell in stripped.strip("|").split("|")
            ]

            if len(cells) >= 4:
                variable = cells[0]
                var_type = cells[1]
                default = cells[2]
                description = cells[3]

                bullet = f"- {variable}: {description}"

                extra_parts = []

                if var_type and var_type not in {"-", "*"}:
                    extra_parts.append(f"Type: {var_type}")

                if default and default != "-":
                    extra_parts.append(f"Default: {default}")

                if extra_parts:
                    bullet += f" ({'; '.join(extra_parts)})"

                output_lines.append(bullet)
                continue

        output_lines.append(line)

    return "\n".join(output_lines)

def clean_table_cell_text(text: str) -> str:
    """Clean Markdown table cell text for compact source previews."""

    cleaned = text.strip()

    cleaned = cleaned.replace("<br>", " / ")
    cleaned = cleaned.replace("<br/>", " / ")
    cleaned = cleaned.replace("<br />", " / ")

    # Remove Markdown code ticks but keep the technical token.
    cleaned = cleaned.replace("`", "")

    # Convert Markdown links [text](url) -> text.
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Normalize spacing around slash separators.
    cleaned = re.sub(r"\s*/\s*", " / ", cleaned)

    return cleaned.strip()

def clean_evidence_chunk(text: str, source_title: str = "") -> str:
    """Clean retrieved evidence for user-facing display."""

    if not text:
        return ""

    cleaned = text.strip()

    # Remove YAML frontmatter.
    cleaned = re.sub(r"(?s)^---\s*.*?\s*---\s*", "", cleaned)

    def remove_dirty_code_block(match: re.Match) -> str:
        """Remove only code blocks that contain raw HTML or template remnants."""

        block = match.group(0)

        dirty_patterns = [
            r"</?div[^>]*>",
            r"\{\{.*?\}\}",
            r"\{%.*?%\}",
            r"\[\%.*?%\]",
            r"\[\[.*?\]\]",
        ]

        if any(
            re.search(pattern, block, flags=re.IGNORECASE | re.DOTALL)
            for pattern in dirty_patterns
        ):
            return ""

        return block


    cleaned = re.sub(
        r"(?is)```(?:[a-zA-Z0-9_-]+)?\s*.*?\s*```",
        remove_dirty_code_block,
        cleaned,
    )

    # Remove standalone HTML tag lines.
    cleaned = re.sub(r"(?im)^\s*</?div[^>]*>\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*<br\s*/?>\s*$", "", cleaned)

    # Remove template / macro lines.
    cleaned = re.sub(
        r"(?im)^\s*(\[%|\[\[|{%|{{).*?(\]|\]\]|%}|}})?\s*$",
        "",
        cleaned,
    )

    lines = cleaned.splitlines()
    cleaned_lines = []

    normalized_title = source_title.strip().lower()

    for line in lines:
        stripped = line.strip()

        if not stripped and not cleaned_lines:
            continue

        # Remove markdown headings.
        if stripped.startswith("#"):
            continue

        # Remove duplicated plain source title.
        if normalized_title and stripped.lower() == normalized_title:
            continue

        # Remove raw HTML-only lines again after splitting.
        if re.fullmatch(r"</?div[^>]*>", stripped, flags=re.IGNORECASE):
            continue

        cleaned_lines.append(line.rstrip())

    cleaned = "\n".join(cleaned_lines).strip()
    cleaned = convert_markdown_tables_to_bullets(cleaned)
    cleaned = clean_markdown_table_fragments(cleaned)

    # Normalize blank lines, including lines that contain only spaces.
    # For the compact source panel, one blank line is enough.
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n\s*\n+", "\n", cleaned)


    return cleaned


def render_sources_panel(
    response: dict[str, Any] | None,
    panel_key: str,
) -> None:
    """Render selected relevant sources for one assistant answer.

    This panel displays a compact, filtered set of relevant sources rather than
    all retrieved top-k results or only model-cited sources.
    """

    if not response:
        return

    all_sources = response.get("sources", [])
    retrieved_results = response.get("retrieved_results", [])
    query = response.get("query", "")

    if not all_sources:
        return

    display_sources = select_display_sources(
        query=query,
        sources=all_sources,
        max_sources=DISPLAY_SOURCE_K,
    )

    if not display_sources:
        return

    # Count duplicate files among displayed sources.
    file_counts: dict[str, int] = {}
    for source in display_sources:
        file_path = source.get("source", "Unknown file")
        file_counts[file_path] = file_counts.get(file_path, 0) + 1

    header_html = (
        '<div class="source-panel-header">'
        '<div class="source-panel-title">Relevant sources</div>'
        '<div class="source-panel-subtitle">Documentation selected for this answer</div>'
        '</div>'
    )
    st.markdown(header_html, unsafe_allow_html=True)

    source_options: list[str] = []
    source_lookup: dict[str, dict[str, Any]] = {}

    for display_index, source in enumerate(display_sources, start=1):
        original_source_number = int(source.get("source_number", display_index))

        title = source.get("title", "Unknown title")
        section = source.get("section") or ""
        file_path = source.get("source", "Unknown file")

        is_duplicate_file = file_counts.get(file_path, 0) > 1

        short_title = title if len(title) <= 48 else title[:45] + "..."
        short_section = section if len(section) <= 34 else section[:31] + "..."

        if (
            is_duplicate_file
            and section
            and section.strip().lower() != title.strip().lower()
        ):
            option_label = f"[{display_index}] {short_title} · {short_section}"
        else:
            option_label = f"[{display_index}] {short_title}"

        retrieved_result = (
            retrieved_results[original_source_number - 1]
            if 1 <= original_source_number <= len(retrieved_results)
            else {}
        )

        source_options.append(option_label)
        source_lookup[option_label] = {
            "display_source_number": display_index,
            "original_source_number": original_source_number,
            "title": title,
            "section": section,
            "file_path": file_path,
            "is_duplicate_file": is_duplicate_file,
            "retrieved_result": retrieved_result,
        }

    selector_key = f"source_selector_{panel_key}"

    selected_label = st.radio(
        "Relevant sources",
        source_options,
        index=0,
        key=selector_key,
        label_visibility="collapsed",
    )

    selected = source_lookup[selected_label]

    display_source_number = selected["display_source_number"]
    title = selected["title"]
    section = selected["section"]
    file_path = selected["file_path"]
    is_duplicate_file = selected["is_duplicate_file"]
    retrieved_result = selected["retrieved_result"]
    chunk_content = retrieved_result.get("content", "")

    if not chunk_content:
        evidence_html = (
            '<div class="evidence-card">'
            f'<div class="evidence-title">[{display_source_number}] {html.escape(title)}</div>'
            f'<div class="evidence-path">{html.escape(file_path)}</div>'
            '<div class="evidence-body muted">Retrieved evidence chunk could not be found.</div>'
            '</div>'
        )
        st.markdown(evidence_html, unsafe_allow_html=True)
        return

    cleaned_chunk = clean_evidence_chunk(chunk_content, source_title=title)
    evidence_preview = cleaned_chunk[:1800]

    truncated_note = (
        '<div class="evidence-truncated-note">Evidence preview truncated.</div>'
        if len(cleaned_chunk) > 1800
        else ""
    )

    section_html = ""
    if (
        is_duplicate_file
        and section
        and section.strip().lower() != title.strip().lower()
    ):
        section_html = f'<div class="evidence-section">{html.escape(section)}</div>'

    evidence_html = (
        '<div class="evidence-card">'
        f'<div class="evidence-title">[{display_source_number}] {html.escape(title)}</div>'
        f'<div class="evidence-path">{html.escape(file_path)}</div>'
        f'{section_html}'
        f'<div class="evidence-body">{html.escape(evidence_preview)}</div>'
        f'{truncated_note}'
        '</div>'
    )

    st.markdown(evidence_html, unsafe_allow_html=True)


def render_user_message(content: str) -> None:
    """Render user message as a left-aligned chat bubble."""

    st.markdown(
        (
            '<div class="chat-row user-row-left">'
            f'<div class="user-bubble user-bubble-left">{html.escape(content)}</div>'
            '</div>'
        ),
        unsafe_allow_html=True,
    )


def clear_chat() -> None:
    """Clear chat history and pending query."""

    st.session_state["messages"] = []
    st.session_state["pending_query"] = None

    if "example_widget_version" in st.session_state:
        st.session_state["example_widget_version"] += 1


def get_relevance_score(source: dict[str, Any]) -> float:
    return float(source.get("hybrid_score", 0.0))


def format_retrieval_sources(source: dict[str, Any]) -> str:
    retrieval_sources = source.get("retrieval_sources", [])

    if not retrieval_sources:
        return "N/A"

    return ", ".join(retrieval_sources)





def render_technical_details(response: dict[str, Any] | None = None) -> None:
    """Render technical details for diagnostics."""

    st.markdown("**Retrieval**")
    st.write(f"Initial retrieval k: `{INITIAL_RETRIEVAL_K}`")
    st.write(f"Final sources: `{FINAL_RETRIEVAL_K}`")
    st.write(f"Dense weight: `{DENSE_WEIGHT}`")
    st.write(f"Sparse weight: `{SPARSE_WEIGHT}`")

    st.divider()

    st.markdown("**Generation**")
    st.write(f"Backend: `{LLM_BACKEND}`")
    

    if LLM_BACKEND == "openai_compatible":
        active_model = OPENAI_COMPATIBLE_MODEL
    elif LLM_BACKEND == "ollama":
        active_model = OLLAMA_MODEL
    else:
        active_model = "N/A"

    st.markdown(f"Model: `{active_model}`")

    st.write(f"Temperature: `{TEMPERATURE}`")
    st.write(f"Top-p: `{TOP_P}`")

    st.divider()

    st.markdown("**Runtime**")
    st.write(f"Max context chars: `{CONTEXT_MAX_CHARS}`")

    if response:
        st.write(f"Mode: `{response.get('mode', 'N/A')}`")
        elapsed = response.get("elapsed_seconds")
        if elapsed is not None:
            st.write(f"Response time: `{elapsed:.1f}s`")

def render_assistant_message(message: dict[str, Any], message_index: int) -> None:
    """Render assistant answer safely as Markdown."""

    response = message.get("response", {})
    answer = response.get("answer", message.get("content", ""))

    st.markdown(answer)


def render_welcome_message() -> None:
    """Render the initial assistant welcome message."""
    if not should_show_welcome():
        return

    avatar_col, content_col = st.columns([0.06, 0.94])

    with avatar_col:
        st.markdown(
            """
            <div class="assistant-avatar">🤖</div>
            """,
            unsafe_allow_html=True,
        )

    with content_col:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-title">Hi, I can help you answer questions about n8n documentation.</div>
                <div class="welcome-text">
                Ask about configuration, permissions, workflows, credentials, privacy/security, or AI features.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if not should_show_welcome():
            return
        
        examples = [
            "How can I configure environment variables in n8n?",
            "How does source control work in n8n?",
            "How can users manage permissions?",
            "What AI features does n8n provide?",
        ]

        example_key = f"example_question_pills_{st.session_state['example_widget_version']}"

        selected_example = st.pills(
            "Try asking:",
            examples,
            selection_mode="single",
            key=example_key,
        )

        if selected_example:
            st.session_state["messages"].append(
                {
                    "role": "user",
                    "content": selected_example,
                }
            )

            st.session_state["pending_query"] = selected_example

            # Change the widget key next time instead of modifying the widget state directly.
            st.session_state["example_widget_version"] += 1

            st.rerun()


# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="app-header">
        <div class="app-header-text">
            <div class="app-title">n8n Docs Assistant</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("Developer diagnostics")
    st.caption(
        "Inspect backend, retrieval weights, model settings, and runtime configuration."
    )

    st.divider()
    render_technical_details(None)

if st.session_state["messages"]:
    clear_col_1, clear_col_2 = st.columns([0.82, 0.18])

    with clear_col_2:
        if st.button("Clear history", key="clear_history_main", use_container_width=True):
            clear_chat()
            st.rerun()


# -----------------------------------------------------------------------------
# Chat history
# -----------------------------------------------------------------------------
def render_chat_area() -> None:
    """Render welcome message and chat history."""

    if should_show_welcome():
        render_welcome_message()
        return

    for message_index, message in enumerate(st.session_state["messages"]):
        role = message["role"]

        if role == "user":
            render_user_message(message["content"])
            continue

        response = message.get("response", {})
        sources = response.get("sources", [])

        if sources:
            answer_col, source_col = st.columns([0.58, 0.42], gap="large")

            with answer_col:
                render_assistant_message(message, message_index)

            with source_col:
                render_sources_panel(response, panel_key=f"message_{message_index}")
        else:
            render_assistant_message(message, message_index)



# -----------------------------------------------------------------------------
# Main chat workspace
# -----------------------------------------------------------------------------

chat_workspace = st.empty()

with chat_workspace.container():
    render_chat_area()

    if st.session_state.get("pending_query"):
        pending_query = st.session_state["pending_query"]

        start_time = time.perf_counter()
        response = run_rag_query(pending_query)
        response_time = time.perf_counter() - start_time

        response["elapsed_seconds"] = response_time

        st.session_state["messages"].append(
            {
                "role": "assistant",
                "content": response["answer"],
                "response": response,
            }
        )

        st.session_state["pending_query"] = None
        st.rerun()
# -----------------------------------------------------------------------------
# Chat input
# -----------------------------------------------------------------------------

user_query = st.chat_input(
    "Ask about n8n docs, permissions, workflows, credentials..."
)

if user_query:
    st.session_state["messages"].append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    st.session_state["pending_query"] = user_query
    st.rerun()