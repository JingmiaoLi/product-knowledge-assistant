import time
import re
from typing import Any


from config import (
    CONTEXT_MAX_CHARS,
    FINAL_RETRIEVAL_K,
    INITIAL_RETRIEVAL_K,
    LLM_BACKEND,
)

from retrieval.hybrid_retriever import hybrid_search
from retrieval.source_metadata import (
    describe_source_scope,
    infer_source_type,
)
from llm_client import (
    generate_answer as generate_llm_answer,
    stream_generate_answer,
)


# -----------------------------------------------------------------------------
# Direct response handling
# -----------------------------------------------------------------------------

SMALL_TALK_PATTERNS = [
    "hi",
    "hello",
    "hey",
    "good morning",
    "good afternoon",
    "good evening",
    "who are you",
    "what are you",
    "what's your name",
    "what is your name",
    "what can you do",
    "what can i ask",
    "help",
    "how do you work",
    "what's the weather like",
]


def is_small_talk_or_identity(query: str) -> bool:
    """Detect simple greeting, identity, or help questions."""

    query_lower = query.strip().lower().rstrip("?!.")
    return any(pattern in query_lower for pattern in SMALL_TALK_PATTERNS)


def build_direct_response(query: str) -> dict[str, Any] | None:
    """Handle simple assistant identity or help questions without retrieval."""

    if not is_small_talk_or_identity(query):
        return None

    return {
        "query": query,
        "answer": (
            "I’m the n8n Product Knowledge Assistant. "
            "I’m designed to answer questions about selected n8n documentation. "
            "Try asking about n8n configuration, permissions, workflows, source control, "
            "credentials, privacy/security, or AI features."
        ),
        "sources": [],
        "retrieved_results": [],
        "mode": "direct_response",
    }


# -----------------------------------------------------------------------------
# Follow-up query rewriting
# -----------------------------------------------------------------------------

FOLLOW_UP_PREFIXES = [
    "how about the",
    "what about the",
    "how about",
    "what about",
    "what if",
    "also",
    "then",
    "and ",
]


def normalize_query_text(query: str) -> str:
    """Normalize user query text for lightweight routing."""

    return re.sub(r"\s+", " ", query.lower().strip())


def get_previous_user_query(
    chat_history: list[dict[str, Any]] | None,
    current_query: str,
) -> str | None:
    """Return the most recent previous user query, excluding the current query."""

    if not chat_history:
        return None

    current_query_normalized = normalize_query_text(current_query)

    for message in reversed(chat_history):
        if message.get("role") != "user":
            continue

        content = str(message.get("content", "")).strip()

        if not content:
            continue

        if normalize_query_text(content) == current_query_normalized:
            continue

        return content

    return None


def is_follow_up_query(query: str) -> bool:
    """Detect short follow-up questions that depend on previous context."""

    q = normalize_query_text(query)

    if not q:
        return False

    if any(q.startswith(prefix) for prefix in FOLLOW_UP_PREFIXES):
        return True

    words = q.rstrip("?!.").split()

    if len(words) <= 3 and not any(
        term in q
        for term in [
            "install",
            "configure",
            "configuration",
            "permission",
            "permissions",
            "source control",
            "environment variable",
            "environment variables",
            "ai feature",
            "ai features",
        ]
    ):
        return True

    return False


def strip_follow_up_prefix(query: str) -> str:
    """Remove common follow-up prefixes while keeping the topic phrase."""

    q = query.strip().rstrip("?!.")
    q_lower = q.lower()

    for prefix in sorted(FOLLOW_UP_PREFIXES, key=len, reverse=True):
        if q_lower.startswith(prefix):
            return q[len(prefix):].strip(" :,-")

    return q.strip()


def rewrite_follow_up_query(
    query: str,
    previous_user_query: str | None,
) -> str:
    """Rewrite a context-dependent follow-up into a standalone retrieval query."""

    if not previous_user_query:
        return query

    if not is_follow_up_query(query):
        return query

    topic = strip_follow_up_prefix(query)

    if not topic:
        return query

    previous = normalize_query_text(previous_user_query)
    topic_lower = normalize_query_text(topic)

    if is_environment_variable_query(previous_user_query):
        return f"How do I configure {topic_lower} environment variables in n8n?"

    if is_installation_query(previous_user_query):
        return f"How do I install n8n using {topic_lower}?"

    if is_permissions_query(previous_user_query):
        return f"How can users manage {topic_lower} permissions in n8n?"

    if "source control" in previous:
        return f"How does {topic_lower} work for source control in n8n?"

    if is_ai_features_query(previous_user_query):
        return f"What AI features does n8n provide for {topic_lower}?"

    return f"{previous_user_query.rstrip('?!.')} — {query}"


# -----------------------------------------------------------------------------
# Optional Ollama warm-up compatibility
# -----------------------------------------------------------------------------

def warm_up_ollama() -> None:
    """Warm up Ollama if the configured LLM backend supports it.

    Kept here for backward compatibility if app.py imports warm_up_ollama
    from rag_pipeline.py.
    """

    if LLM_BACKEND != "ollama":
        return

    try:
        from llm_client import warm_up_ollama as llm_warm_up_ollama

        llm_warm_up_ollama()
    except Exception as e:
        print(f"Ollama warm-up failed: {e}")


# -----------------------------------------------------------------------------
# Query-aware source routing / boosting
# -----------------------------------------------------------------------------

ENV_VAR_TOPIC_TO_PATHS = {
    "database": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "db": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "postgres": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "postgresql": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "sqlite": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "mysql": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "mariadb": [
        "hosting/configuration/environment-variables/database.md",
        "hosting/configuration/supported-databases-settings.md",
    ],
    "log": [
        "hosting/configuration/environment-variables/logs.md",
    ],
    "logs": [
        "hosting/configuration/environment-variables/logs.md",
    ],
    "logging": [
        "hosting/configuration/environment-variables/logs.md",
    ],
    "timezone": [
        "hosting/configuration/environment-variables/timezone-localization.md",
    ],
    "queue": [
        "hosting/configuration/environment-variables/queue-mode.md",
    ],
    "security": [
        "hosting/configuration/environment-variables/security.md",
    ],
    "credentials": [
        "hosting/configuration/environment-variables/credentials.md",
    ],
    "credential": [
        "hosting/configuration/environment-variables/credentials.md",
    ],
    "execution": [
        "hosting/configuration/environment-variables/executions.md",
    ],
    "executions": [
        "hosting/configuration/environment-variables/executions.md",
    ],
    "binary": [
        "hosting/configuration/environment-variables/binary-data.md",
    ],
    "binary data": [
        "hosting/configuration/environment-variables/binary-data.md",
    ],
    "workflow": [
        "hosting/configuration/environment-variables/workflows.md",
    ],
    "workflows": [
        "hosting/configuration/environment-variables/workflows.md",
    ],
    "smtp": [
        "hosting/configuration/environment-variables/user-management-smtp-2fa.md",
    ],
    "2fa": [
        "hosting/configuration/environment-variables/user-management-smtp-2fa.md",
    ],
    "user management": [
        "hosting/configuration/environment-variables/user-management-smtp-2fa.md",
    ],
    "source control": [
        "hosting/configuration/environment-variables/source-control.md",
    ],
    "external secrets": [
        "hosting/configuration/environment-variables/external-secrets.md",
    ],
    "external data": [
        "hosting/configuration/environment-variables/external-data-storage.md",
    ],
    "task runner": [
        "hosting/configuration/environment-variables/task-runners.md",
    ],
    "task runners": [
        "hosting/configuration/environment-variables/task-runners.md",
    ],
}


GENERAL_ENV_VAR_PATHS = [
    "hosting/configuration/environment-variables/index.md",
    "hosting/configuration/configuration-methods.md",
]


GENERAL_INSTALL_PATHS = [
    "hosting/installation/npm.md",
]


def is_environment_variable_query(query: str) -> bool:
    """Return True for environment-variable-related questions."""

    q = normalize_query_text(query)

    return any(
        phrase in q
        for phrase in [
            "environment variable",
            "environment variables",
            "env variable",
            "env variables",
            "env var",
            "env vars",
        ]
    )


def is_installation_query(query: str) -> bool:
    """Return True for installation-related questions."""

    q = normalize_query_text(query)

    install_terms = [
        "install",
        "installation",
        "set up",
        "setup",
    ]

    return any(term in q for term in install_terms)


def get_desired_source_paths_for_query(query: str) -> list[str]:
    """Return source paths that should be boosted for the query."""

    q = normalize_query_text(query)

    if is_environment_variable_query(query):
        desired_paths: list[str] = []

        for topic, paths in ENV_VAR_TOPIC_TO_PATHS.items():
            if topic in q:
                desired_paths.extend(paths)

        if desired_paths:
            return desired_paths

        return GENERAL_ENV_VAR_PATHS

    if is_installation_query(query):
        scenario_terms = [
            "aws",
            "amazon web services",
            "azure",
            "gke",
            "google kubernetes",
            "kubernetes",
            "k8s",
            "openshift",
            "crc",
            "docker",
            "cli",
            "claude",
            "server cli",
            "oem",
        ]

        if not any(term in q for term in scenario_terms):
            return GENERAL_INSTALL_PATHS

    return []


def get_matched_env_var_topics(query: str) -> list[str]:
    """Return matched environment-variable topics from the query."""

    q = normalize_query_text(query)

    matched_topics = []

    for topic in ENV_VAR_TOPIC_TO_PATHS:
        if topic in q:
            matched_topics.append(topic)

    return matched_topics


def is_specific_environment_variable_query(query: str) -> bool:
    """Return True when the query asks about a specific environment-variable category."""

    return is_environment_variable_query(query) and bool(get_matched_env_var_topics(query))


def asks_for_command_line_env_setting(query: str) -> bool:
    """Return True when the user explicitly asks about shell/terminal syntax."""

    q = normalize_query_text(query)

    shell_terms = [
        "terminal",
        "command line",
        "command-line",
        "shell",
        "bash",
        "cmd",
        "cmd.exe",
        "powershell",
        "export",
        "set variable",
        "set environment variable",
    ]

    return any(term in q for term in shell_terms)


def result_source_path(result: dict[str, Any]) -> str:
    """Return normalized source path from a retrieved result."""

    metadata = result.get("metadata", {})
    return str(metadata.get("source", "")).lower()


def is_configuration_methods_result(result: dict[str, Any]) -> bool:
    """Return True for the generic environment variable setting methods page."""

    path = result_source_path(result)
    return path.endswith("hosting/configuration/configuration-methods.md")


def is_permissions_query(query: str) -> bool:
    """Return True for general permissions / roles / RBAC questions."""

    q = normalize_query_text(query)

    permission_terms = [
        "permission",
        "permissions",
        "role",
        "roles",
        "rbac",
        "access control",
        "account type",
        "account types",
    ]

    return any(term in q for term in permission_terms)


def is_chat_hub_query(query: str) -> bool:
    """Return True when the user specifically asks about Chat Hub."""

    q = normalize_query_text(query)

    return "chat hub" in q or "chat-hub" in q


def is_ai_features_query(query: str) -> bool:
    """Return True for broad AI feature questions."""

    q = normalize_query_text(query)

    ai_terms = [
        "ai feature",
        "ai features",
        "artificial intelligence feature",
        "artificial intelligence features",
        "what ai",
        "ai in n8n",
        "ai agent",
        "llm",
        "large language model",
    ]

    return any(term in q for term in ai_terms)


def is_ai_privacy_query(query: str) -> bool:
    """Return True when the user asks about AI privacy or data collection."""

    q = normalize_query_text(query)

    privacy_terms = [
        "privacy",
        "data collection",
        "data shared",
        "data sent",
        "opt in",
        "opt-in",
        "assistant data",
        "telemetry",
    ]

    return any(term in q for term in privacy_terms)


def is_chat_hub_result(result: dict[str, Any]) -> bool:
    """Return True for Chat Hub-specific documentation."""

    path = result_source_path(result)
    return path.endswith("advanced-ai/chat-hub.md") or "advanced-ai/chat-hub" in path


def is_privacy_security_result(result: dict[str, Any]) -> bool:
    """Return True for privacy/security documentation."""

    path = result_source_path(result)
    return path.startswith("privacy-security/") or "privacy-security/" in path


def is_root_index_result(result: dict[str, Any]) -> bool:
    """Return True for the general docs landing page."""

    path = result_source_path(result)
    return path == "index.md" or (path.endswith("/index.md") and path.count("/") == 0)


def path_matches(source_path: str, target_path: str) -> bool:
    """Return True when a retrieved source path matches a desired path."""

    source_path = source_path.lower()
    target_path = target_path.lower()

    return source_path.endswith(target_path) or target_path in source_path


def should_exclude_result_for_query(
    query: str,
    result: dict[str, Any],
    desired_paths: list[str],
) -> bool:
    """Return True for results that should not enter the final LLM context."""

    path = result_source_path(result)

    if is_specific_environment_variable_query(query):
        if desired_paths:
            matches_desired_path = any(
                path_matches(path, target_path)
                for target_path in desired_paths
            )

            if not matches_desired_path:
                return True

        if is_configuration_methods_result(result) and not asks_for_command_line_env_setting(query):
            return True

    if is_tangential_for_broad_install_query(query, result):
        return True

    if is_permissions_query(query) and not is_chat_hub_query(query):
        if is_chat_hub_result(result):
            return True

    if is_ai_features_query(query) and not is_ai_privacy_query(query):
        if is_privacy_security_result(result):
            return True

        if is_root_index_result(result):
            return True

    return False


def get_topic_boost_rank(
    result: dict[str, Any],
    desired_paths: list[str],
) -> int:
    """Return a rank for topic-specific source boosting. Lower is better."""

    if not desired_paths:
        return 100

    source_path = result_source_path(result)

    for rank, target_path in enumerate(desired_paths):
        if path_matches(source_path, target_path):
            return rank

    return 100


def is_tangential_for_broad_install_query(
    query: str,
    result: dict[str, Any],
) -> bool:
    """Detect overly specific sources for broad installation questions."""

    if not is_installation_query(query):
        return False

    q = normalize_query_text(query)

    specific_terms = [
        "openshift",
        "crc",
        "kubernetes",
        "aws",
        "azure",
        "gke",
        "oem",
        "oem-deployment",
        "n8n-cli",
        "server cli",
        "server-cli",
        "claude",
        "hetzner",
        "digitalocean",
        "digital ocean",
        "server-setups",
        "server setups",
    ]

    if any(term in q for term in specific_terms):
        return False

    metadata = result.get("metadata", {})
    text = " ".join(
        [
            str(metadata.get("source", "")),
            str(metadata.get("title", "")),
            str(metadata.get("h1", "")),
            str(metadata.get("h2", "")),
            str(metadata.get("h3", "")),
        ]
    ).lower()

    tangential_terms = [
        "openshift",
        "crc",
        "kubernetes",
        "aws",
        "azure",
        "gke",
        "oem",
        "oem-deployment",
        "n8n-cli",
        "server cli",
        "server-cli",
        "claude",
        "hetzner",
        "digitalocean",
        "digital ocean",
        "server-setups",
        "server setups",
    ]

    return any(term in text for term in tangential_terms)


# -----------------------------------------------------------------------------
# Retrieval result enrichment and prioritization
# -----------------------------------------------------------------------------

def search_docs(query: str, k: int = FINAL_RETRIEVAL_K) -> list[dict[str, Any]]:
    """Retrieve a larger candidate set before final source selection."""

    candidate_k = max(INITIAL_RETRIEVAL_K, k * 3)
    return hybrid_search(query, final_k=candidate_k)


def enrich_retrieved_results_with_source_metadata(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Add source_type and source_scope to each retrieved result's metadata."""

    enriched_results = []

    for item in results:
        enriched_item = dict(item)
        metadata = dict(enriched_item.get("metadata", {}))

        section = " > ".join(
            h
            for h in [
                metadata.get("h1", ""),
                metadata.get("h2", ""),
                metadata.get("h3", ""),
            ]
            if h
        )

        source_info = {
            "source": metadata.get("source", ""),
            "title": metadata.get("title", ""),
            "section": section,
        }

        source_type = infer_source_type(source_info)
        source_scope = describe_source_scope(source_type)

        metadata["source_type"] = source_type
        metadata["source_scope"] = source_scope

        enriched_item["metadata"] = metadata
        enriched_results.append(enriched_item)

    return enriched_results


def prioritize_retrieved_results_for_query(
    query: str,
    results: list[dict[str, Any]],
    max_results: int = FINAL_RETRIEVAL_K,
) -> list[dict[str, Any]]:
    """Prioritize and filter retrieved results using query-aware source boosting."""

    if not results:
        return []

    desired_paths = get_desired_source_paths_for_query(query)

    filtered_results = [
        item
        for item in results
        if not should_exclude_result_for_query(
            query=query,
            result=item,
            desired_paths=desired_paths,
        )
    ]

    if not filtered_results:
        filtered_results = results

    scored_results = []

    for original_rank, item in enumerate(filtered_results):
        metadata = item.get("metadata", {})
        source_type = metadata.get("source_type", "general")
        hybrid_score = float(item.get("hybrid_score", 0.0))

        topic_rank = get_topic_boost_rank(item, desired_paths)

        if topic_rank < 100:
            priority_group = 0
        elif source_type == "general":
            priority_group = 1
        else:
            priority_group = 2

        scored_results.append(
            (
                priority_group,
                topic_rank,
                original_rank,
                -hybrid_score,
                item,
            )
        )

    scored_results.sort(key=lambda row: row[:4])

    prioritized = [row[-1] for row in scored_results]

    return prioritized[:max_results]


# -----------------------------------------------------------------------------
# Context cleaning and formatting
# -----------------------------------------------------------------------------

def clean_context_content_for_llm(text: str) -> str:
    """Clean retrieved chunk content before sending it to the LLM."""

    if not text:
        return ""

    cleaned = text.strip()

    # Remove leftover HTML tags and broken fenced-code remnants before other cleanup.
    cleaned = cleaned.replace("```</div>", "")
    cleaned = cleaned.replace("</div>", "")
    cleaned = cleaned.replace("<div>", "")
    cleaned = re.sub(r"</?div[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*```</div>\s*$", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*</?div[^>]*>\s*$", "", cleaned)

    pointer_patterns = [
        r"[^.\n]*\bfor the complete and most up-to-date list\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bfor the complete list\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bfor a full list\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bsee the full reference\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bfull reference\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\brefer to the\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bsee the .*documentation\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bsee the .*docs\b[^.\n]*(?:\.|$)",
        r"[^.\n]*\bcan be found in the .*docs repository\b[^.\n]*(?:\.|$)",
    ]

    for pattern in pointer_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)

    # Normalize spacing.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()

def format_context(
    results: list[dict[str, Any]],
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """Format retrieved chunks as structured context for the LLM."""

    context_parts = []

    for i, item in enumerate(results, start=1):
        metadata = item["metadata"]

        title = metadata.get("title", "Unknown title")
        source = metadata.get("source", "Unknown source")
        category = metadata.get("category", "Unknown category")
        source_type = metadata.get("source_type", "general")
        source_scope = metadata.get("source_scope", "general product documentation")

        heading_path = " > ".join(
            h
            for h in [
                metadata.get("h1", ""),
                metadata.get("h2", ""),
                metadata.get("h3", ""),
            ]
            if h
        )

        content = clean_context_content_for_llm(item.get("content", ""))

        if not content:
            continue

        context_parts.append(
            f"[Source {i}]\n"
            f"Title: {title}\n"
            f"Category: {category}\n"
            f"Source type: {source_type}\n"
            f"Scope: {source_scope}\n"
            f"Section: {heading_path if heading_path else 'N/A'}\n"
            f"File: {source}\n"
            f"Content:\n{content}\n"
        )

    context = "\n\n---\n\n".join(context_parts)

    return context[:max_chars]

def normalize_spacing_preserving_code_blocks(text: str) -> str:
    """Normalize excessive spaces outside fenced code blocks only."""

    lines = text.splitlines()
    normalized_lines = []
    inside_code_block = False

    for line in lines:
        stripped = line.strip()

        if stripped.startswith("```"):
            inside_code_block = not inside_code_block
            normalized_lines.append(line)
            continue

        if inside_code_block:
            # Preserve indentation inside code blocks.
            normalized_lines.append(line)
        else:
            # Normalize repeated spaces only in normal prose.
            normalized_lines.append(re.sub(r"[ \t]{2,}", " ", line))

    return "\n".join(normalized_lines)


def clean_generated_answer(answer: str) -> str:
    """Clean generated answer before returning it to any UI or API."""

    if not answer:
        return ""

    cleaned = answer.strip()
    # Remove leftover HTML tags and broken fenced-code remnants from model output.
    cleaned = cleaned.replace("```</div>", "```")
    cleaned = cleaned.replace("</div>", "")
    cleaned = cleaned.replace("<div>", "")
    cleaned = re.sub(r"</?div[^>]*>", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"(?im)^\s*</?div[^>]*>\s*$", "", cleaned)

    cleaned = re.sub(
        r"\s*\[(?:Source|Sources)\s+[^\]]+\]\([^)]+\)",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(
        r"\s*\[(?:Source|Sources)\s+[^\]]+\]",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    referral_clause_patterns = [
        r"\s*Refer to [^.]* for more information\.",
        r"\s*For more information,? [^.]*\.",
        r"\s*See the [^.]*documentation[^.]*\.",
        r"\s*See the [^.]*docs[^.]*\.",
        r"\s*Check the [^.]*documentation[^.]*\.",
        r"\s*as listed in [^.]*\.",
        r"\s*as listed in the [^.]*section[^.]*\.",
        r"\s*as listed in this section[^.]*\.",
    ]

    for pattern in referral_clause_patterns:
        cleaned = re.sub(pattern, "", cleaned, flags=re.IGNORECASE)

    standalone_referral_patterns = [
        r"(?im)^\s*Refer to .* for more information\.\s*$",
        r"(?im)^\s*For more information,?.*\.\s*$",
        r"(?im)^\s*See the .*documentation.*\.\s*$",
        r"(?im)^\s*See the .*docs.*\.\s*$",
        r"(?im)^\s*Check the .*documentation.*\.\s*$",
    ]

    for pattern in standalone_referral_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    cleaned = re.sub(r"(?im)^\s*Note that\s+", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*Note:\s*", "", cleaned)

    cleaned = re.sub(
        r"(?im)^\s*Refer to [^\n.]*\.?\s*$",
        "",
        cleaned,
    )
    cleaned = re.sub(
        r"\s*Refer to [^\n.]*\.?",
        "",
        cleaned,
        flags=re.IGNORECASE,
    )

    cleaned = re.sub(r"(?m)^you can", "You can", cleaned)
    cleaned = re.sub(r"(?m)^you must", "You must", cleaned)
    cleaned = re.sub(r"(?m)^you should", "You should", cleaned)
    cleaned = re.sub(r"(?m)^n8n requires", "n8n requires", cleaned)

    cleaned = re.sub(
        r"(?ms)\n?\|[^\n]+\|\s*\n\|[\s:\-|]+\|\s*(?=\n\n|\n[A-Z]|\Z)",
        "\n",
        cleaned,
    )

    cleaned = re.sub(r"(?m)^\s*[-*]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*$", "", cleaned)
    cleaned = normalize_spacing_preserving_code_blocks(cleaned)
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


NO_ANSWER_MESSAGE = (
    "The retrieved documentation does not contain enough information to answer this question."
)


def is_no_answer_response(answer: str) -> bool:
    """Return True if the answer indicates insufficient retrieved documentation."""

    if not answer:
        return False

    normalized = normalize_query_text(answer)

    no_answer_signals = [
        "the retrieved documentation does not contain enough information",
        "retrieved documentation does not contain enough information",
        "does not contain enough information to answer",
        "does not contain enough information to configure",
        "does not contain enough information for",
        "provided sources do not",
        "provided sources cover",
        "do not specifically address",
        "does not specifically address",
        "not specifically address",
    ]

    return any(signal in normalized for signal in no_answer_signals)


def normalize_no_answer_response(answer: str) -> str:
    """Normalize no-answer responses to one clean product message."""

    if is_no_answer_response(answer):
        return NO_ANSWER_MESSAGE

    return answer


def build_no_answer_response(
    query: str,
    retrieval_query: str,
    previous_user_query: str | None,
    retrieved_results: list[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    """Build a no-answer response without showing irrelevant sources."""

    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "previous_user_query": previous_user_query,
        "answer": NO_ANSWER_MESSAGE,
        "sources": [],
        "retrieved_results": retrieved_results,
        "mode": mode,
        "no_answer": True,
    }


# -----------------------------------------------------------------------------
# Prompt and generation
# -----------------------------------------------------------------------------

def build_prompt(query: str, results: list[dict[str, Any]]) -> str:
    """Build the grounded RAG prompt shared by all generation backends."""

    context = format_context(results)

    return f"""
You are an internal product knowledge assistant for a software product.

Answer the user's question using only the retrieved documentation context.

Strict rules:
- Use only the retrieved context. Do not guess or use outside knowledge.
- Preserve technical tokens exactly, including environment variables, commands, paths, file names, and code snippets.
- Answer directly. Do not mention sources, retrieved context, or documentation panels.
- If the answer is not supported, say exactly: "The retrieved documentation does not contain enough information to answer this question."
- For table fragments, rewrite them into clean bullet points.
- When the context describes multiple configuration methods, include all major methods instead of selecting only one or two.
- If the context contains Markdown table fragments, summarize each row as one bullet point. Do not create separate bullets for Type, Default, and Description.
- For configuration variables, use this format: `VARIABLE_NAME`: purpose. Include type/default values in the same bullet only when available.
- Keep the answer concise and practical.
- Do not end with a generic closing phrase such as "For more information" or "For more detailed guidance."
- Make sure the final sentence is complete.
- For broad conceptual questions, summarize the main concepts first and avoid step-by-step setup instructions unless the user explicitly asks how to configure or create something.
- For permission-related questions, use at most three sections: Account-level roles, Project-level RBAC roles, and Custom project roles. Keep each section to one concise bullet unless the user asks for details.
- If one source contains detailed setup steps but the user asks a broad question, use that source only as supporting detail, not as the main structure of the answer.
- Keep broad answers concise: usually 3 to 5 bullets. Do not expand every permission or setting unless the user asks for details.
- Avoid generic closing sentences. End with a concrete summary only when it adds useful information.
- Prefer product-specific terms such as "workflow automation" over generic phrases such as "traditional programming" when the context is about n8n.
- Do not add examples or capabilities that are not explicitly supported by the retrieved context.
- Answer in English.

User question:
{query}

Retrieved documentation context:
{context}

Answer:
""".strip()


def build_retrieval_only_answer(results: list[dict[str, Any]]) -> str:
    """Fallback answer when no LLM backend is available."""

    if not results:
        return "No relevant documentation passages were retrieved."

    answer_parts = [
        "I could not generate an LLM answer, but here are the most relevant retrieved documentation passages:"
    ]

    for i, item in enumerate(results[:3], start=1):
        metadata = item["metadata"]
        title = metadata.get("title", "Unknown title")
        source = metadata.get("source", "Unknown source")

        answer_parts.append(
            f"\n[Source {i}] {title} — {source}\n"
            f"{item['content'][:700]}"
        )

    return "\n".join(answer_parts)


def generate_answer(
    query: str,
    results: list[dict[str, Any]],
) -> tuple[str, str]:
    """Generate an answer using the configured LLM backend."""

    if LLM_BACKEND == "retrieval_only":
        answer = build_retrieval_only_answer(results)
        answer = clean_generated_answer(answer)
        return answer, "retrieval_only"

    prompt = build_prompt(query, results)

    try:
        answer = generate_llm_answer(prompt)
        answer = clean_generated_answer(answer)
        return answer, LLM_BACKEND

    except Exception as e:
        fallback_answer = build_retrieval_only_answer(results)
        fallback_answer = clean_generated_answer(fallback_answer)
        return fallback_answer, f"retrieval_only_fallback: {e}"


# -----------------------------------------------------------------------------
# Response formatting
# -----------------------------------------------------------------------------

def get_sources(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Create a source list that keeps the same numbering as the LLM context."""

    sources = []

    for i, item in enumerate(results, start=1):
        metadata = item["metadata"]

        source = metadata.get("source", "Unknown source")
        title = metadata.get("title", "Unknown title")
        category = metadata.get("category", "Unknown category")

        section = " > ".join(
            h
            for h in [
                metadata.get("h1", ""),
                metadata.get("h2", ""),
                metadata.get("h3", ""),
            ]
            if h
        )

        sources.append(
            {
                "source_number": i,
                "title": title,
                "category": category,
                "section": section,
                "source": source,
                "distance": item.get("distance", 1.0),
                "hybrid_score": item.get("hybrid_score", 1.0 - item.get("distance", 1.0)),
                "retrieval_sources": item.get("retrieval_sources", []),
                "source_type": metadata.get("source_type", "general"),
                "source_scope": metadata.get("source_scope", "general product documentation"),
            }
        )

    return sources

def prepare_rag_context(
    query: str,
    k: int = FINAL_RETRIEVAL_K,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run retrieval and prepare the prompt without generating the final answer."""

    direct_response = build_direct_response(query)
    if direct_response is not None:
        return {
            **direct_response,
            "prompt": None,
            "is_direct_response": True,
        }

    previous_user_query = get_previous_user_query(
        chat_history=chat_history,
        current_query=query,
    )

    retrieval_query = rewrite_follow_up_query(
        query=query,
        previous_user_query=previous_user_query,
    )

    candidate_results = search_docs(retrieval_query, k=k)
    candidate_results = enrich_retrieved_results_with_source_metadata(candidate_results)

    reranked_results = prioritize_retrieved_results_for_query(
        query=retrieval_query,
        results=candidate_results,
        max_results=len(candidate_results),
    )

    final_results = reranked_results[:k]

    if LLM_BACKEND == "retrieval_only":
        answer = build_retrieval_only_answer(final_results)
        answer = clean_generated_answer(answer)

        return {
            "query": query,
            "retrieval_query": retrieval_query,
            "previous_user_query": previous_user_query,
            "answer": answer,
            "sources": get_sources(final_results),
            "retrieved_results": final_results,
            "mode": "retrieval_only",
            "no_answer": False,
            "prompt": None,
            "is_direct_response": False,
        }

    prompt = build_prompt(retrieval_query, final_results)

    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "previous_user_query": previous_user_query,
        "answer": "",
        "sources": get_sources(final_results),
        "retrieved_results": final_results,
        "mode": LLM_BACKEND,
        "no_answer": False,
        "prompt": prompt,
        "is_direct_response": False,
    }


def stream_rag_answer(prompt: str):
    """Stream and clean a RAG answer.

    Note: final cleanup happens after streaming in app.py because cleaning partial
    chunks during streaming may break formatting.
    """

    yield from stream_generate_answer(prompt)


def get_rag_response(
    query: str,
    k: int = FINAL_RETRIEVAL_K,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    total_start = time.perf_counter()
    print("\n========== RAG TIMING START ==========")

    direct_response = build_direct_response(query)
    if direct_response is not None:
        print(f"[timing] direct response total: {time.perf_counter() - total_start:.2f}s")
        print("========== RAG TIMING END ==========\n")
        return direct_response

    t0 = time.perf_counter()
    previous_user_query = get_previous_user_query(
        chat_history=chat_history,
        current_query=query,
    )

    retrieval_query = rewrite_follow_up_query(
        query=query,
        previous_user_query=previous_user_query,
    )
    print(f"[timing] query rewrite: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    candidate_results = search_docs(retrieval_query, k=k)
    print(f"[timing] search_docs total: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    candidate_results = enrich_retrieved_results_with_source_metadata(candidate_results)

    reranked_results = prioritize_retrieved_results_for_query(
        query=retrieval_query,
        results=candidate_results,
        max_results=len(candidate_results),
    )

    final_results = reranked_results[:k]
    print(f"[timing] enrich + rerank: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    answer, mode = generate_answer(retrieval_query, final_results)
    print(f"[timing] generation: {time.perf_counter() - t0:.2f}s")

    t0 = time.perf_counter()
    answer = normalize_no_answer_response(answer)
    print(f"[timing] answer normalization: {time.perf_counter() - t0:.2f}s")

    if is_no_answer_response(answer):
        print(f"[timing] total rag_pipeline: {time.perf_counter() - total_start:.2f}s")
        print("========== RAG TIMING END ==========\n")

        return build_no_answer_response(
            query=query,
            retrieval_query=retrieval_query,
            previous_user_query=previous_user_query,
            retrieved_results=final_results,
            mode=mode,
        )

    t0 = time.perf_counter()
    sources = get_sources(final_results)
    print(f"[timing] source formatting: {time.perf_counter() - t0:.2f}s")

    print(f"[timing] total rag_pipeline: {time.perf_counter() - total_start:.2f}s")
    print("========== RAG TIMING END ==========\n")

    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "previous_user_query": previous_user_query,
        "answer": answer,
        "sources": sources,
        "retrieved_results": final_results,
        "mode": mode,
        "no_answer": False,
    }

# -----------------------------------------------------------------------------
# Terminal testing
# -----------------------------------------------------------------------------

def print_rag_response(response: dict[str, Any]) -> None:
    """Pretty-print the RAG response for terminal testing."""

    print("\nQuestion:")
    print(response["query"])

    retrieval_query = response.get("retrieval_query")
    if retrieval_query and retrieval_query != response["query"]:
        print("\nRetrieval query:")
        print(retrieval_query)

    print("\nAnswer:")
    print(response["answer"])

    print("\nSources:")
    for source in response["sources"]:
        section = source["section"] if source["section"] else "N/A"
        retrieval_sources = source.get("retrieval_sources", [])

        print(
            f"{source['source_number']}. {source['title']} | "
            f"Section: {section} | "
            f"File: {source['source']} | "
            f"Source type: {source.get('source_type', 'general')} | "
            f"Scope: {source.get('source_scope', 'general product documentation')} | "
            f"Relevance score: {source.get('hybrid_score', 0.0):.4f} | "
            f"Retrieval: {retrieval_sources}"
        )

    print("\nMode:")
    print(response["mode"])


if __name__ == "__main__":
    chat_history: list[dict[str, Any]] = []

    test_queries = [
        "How can I configure environment variables in n8n?",
        "How about database?",
        "What about logs?",
        "How to install n8n?",
        "What about npm?",
        "How can users manage permissions?",
        "What about custom roles?",
    ]

    for test_query in test_queries:
        print("\n" + "#" * 100)

        chat_history.append(
            {
                "role": "user",
                "content": test_query,
            }
        )

        response = get_rag_response(
            query=test_query,
            k=FINAL_RETRIEVAL_K,
            chat_history=chat_history,
        )

        print_rag_response(response)

        chat_history.append(
            {
                "role": "assistant",
                "content": response["answer"],
                "response": response,
            }
        )