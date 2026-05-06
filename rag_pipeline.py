import os
import re
from typing import Any

import ollama
from dotenv import load_dotenv
from openai import OpenAI

from config import (
    CONTEXT_MAX_CHARS,
    FINAL_RETRIEVAL_K,
    INITIAL_RETRIEVAL_K,
    LLM_BACKEND,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OPENAI_COMPATIBLE_API_KEY_ENV,
    OPENAI_COMPATIBLE_BASE_URL,
    OPENAI_COMPATIBLE_MODEL,
    TEMPERATURE,
    TOP_P,
)
from retrieval.hybrid_retriever import hybrid_search
from retrieval.source_metadata import (
    describe_source_scope,
    infer_source_type,
)


load_dotenv()


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

        # Skip the current query if it is already stored in chat history.
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

    # Very short noun-like follow-ups, such as "database?" or "permissions?"
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
    """Rewrite a context-dependent follow-up into a standalone retrieval query.

    This is intentionally lightweight and rule-based. It improves retrieval for
    common follow-up questions without adding another LLM call.
    """

    if not previous_user_query:
        return query

    if not is_follow_up_query(query):
        return query

    topic = strip_follow_up_prefix(query)

    if not topic:
        return query

    previous = normalize_query_text(previous_user_query)
    topic_lower = normalize_query_text(topic)

    # Environment-variable follow-ups:
    # Previous: "How can I configure environment variables in n8n?"
    # Current: "How about database?"
    # Rewritten: "How do I configure database environment variables in n8n?"
    if is_environment_variable_query(previous_user_query):
        return f"How do I configure {topic_lower} environment variables in n8n?"

    # Installation follow-ups:
    # Previous: "How to install n8n?"
    # Current: "What about npm?"
    if is_installation_query(previous_user_query):
        return f"How do I install n8n using {topic_lower}?"

    # Permission follow-ups:
    # Previous: "How can users manage permissions?"
    # Current: "What about custom roles?"
    if is_permissions_query(previous_user_query):
        return f"How can users manage {topic_lower} permissions in n8n?"

    # Source-control follow-ups:
    # Previous: "How does source control work in n8n?"
    # Current: "How about setup?"
    if "source control" in previous:
        return f"How does {topic_lower} work for source control in n8n?"

    # AI feature follow-ups:
    # Previous: "What AI features does n8n provide?"
    # Current: "What about agents?"
    if is_ai_features_query(previous_user_query):
        return f"What AI features does n8n provide for {topic_lower}?"

    # Generic fallback: keep both pieces of context.
    return f"{previous_user_query.rstrip('?!.')} — {query}"

# -----------------------------------------------------------------------------
# Ollama warm-up
# -----------------------------------------------------------------------------

def warm_up_ollama() -> None:
    """Warm up the local Ollama model without running a full RAG query."""

    if LLM_BACKEND != "ollama":
        return

    try:
        ollama.chat(
            model=OLLAMA_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": "Reply with exactly one word: ready",
                }
            ],
            options={
                "temperature": 0.0,
                "top_p": TOP_P,
            },
            keep_alive=OLLAMA_KEEP_ALIVE,
        )
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


def normalize_query_text(query: str) -> str:
    """Normalize user query text for lightweight routing."""

    return re.sub(r"\s+", " ", query.lower().strip())


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
    """Return source paths that should be boosted for the query.

    This is soft routing: matching documents are promoted, not forced as the
    only context.
    """

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
    return path == "index.md" or path.endswith("/index.md") and path.count("/") == 0



def should_exclude_result_for_query(
    query: str,
    result: dict[str, Any],
    desired_paths: list[str],
) -> bool:
    """Return True for results that should not enter the final LLM context.

    This is a lightweight guardrail layer. It removes clearly off-scope sources
    for known broad query types without replacing semantic retrieval.
    """

    path = result_source_path(result)

    # If the user asks about a specific environment-variable category,
    # keep the final context focused on the matching reference docs.
    if is_specific_environment_variable_query(query):
        if desired_paths:
            matches_desired_path = any(
                path_matches(path, target_path)
                for target_path in desired_paths
            )

            if not matches_desired_path:
                return True

        # Do not include generic shell syntax unless explicitly requested.
        if is_configuration_methods_result(result) and not asks_for_command_line_env_setting(query):
            return True

    # For broad install questions, remove overly specific deployment scenarios.
    if is_tangential_for_broad_install_query(query, result):
        return True

    # For general permissions questions, avoid feature-specific Chat Hub docs
    # unless the user explicitly asks about Chat Hub.
    if is_permissions_query(query) and not is_chat_hub_query(query):
        if is_chat_hub_result(result):
            return True

    # For broad AI feature questions, avoid privacy/security docs unless the
    # user asks about privacy, data collection, opt-in behavior, or assistant data.
    if is_ai_features_query(query) and not is_ai_privacy_query(query):
        if is_privacy_security_result(result):
            return True

        # The docs landing page is usually too broad to answer AI feature
        # questions if more specific advanced-ai sources are available.
        if is_root_index_result(result):
            return True

    return False

def result_source_path(result: dict[str, Any]) -> str:
    """Return normalized source path from a retrieved result."""

    metadata = result.get("metadata", {})
    return str(metadata.get("source", "")).lower()


def path_matches(source_path: str, target_path: str) -> bool:
    """Return True when a retrieved source path matches a desired path."""

    source_path = source_path.lower()
    target_path = target_path.lower()

    return source_path.endswith(target_path) or target_path in source_path


def get_topic_boost_rank(
    result: dict[str, Any],
    desired_paths: list[str],
) -> int:
    """Return a rank for topic-specific source boosting.

    Lower is better. A return value of 100 means no boost.
    """

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
    """Retrieve a larger candidate set before final source selection.

    We retrieve more candidates than the final LLM context size, then apply
    metadata enrichment and lightweight source prioritization before trimming
    back to FINAL_RETRIEVAL_K.
    """

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
    """Prioritize and filter retrieved results using query-aware source boosting.

    For specific environment-variable topics, such as database/logs/timezone,
    this keeps the context focused on the corresponding reference files.

    For broad installation questions, this removes overly specific deployment
    pages such as OpenShift, Hetzner, OEM, CLI, or Kubernetes unless the user
    explicitly asks about those scenarios.
    """

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

    # Safety fallback: never return an empty context if retrieval found results.
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
    """Clean retrieved chunk content before sending it to the LLM.

    This removes documentation navigation/pointer sentences that are useful in
    the original docs but unhelpful in a synthesized RAG answer.
    """

    if not text:
        return ""

    cleaned = text.strip()

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
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]+\n", "\n", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def format_context(
    results: list[dict[str, Any]],
    max_chars: int = CONTEXT_MAX_CHARS,
) -> str:
    """Format retrieved chunks as structured context for the LLM.

    The context includes source type and scope so the generator can distinguish
    general product documentation from scenario-specific hosting or CLI pages.
    It also removes navigation-only documentation sentences before generation.
    """

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


def clean_generated_answer(answer: str) -> str:
    """Clean generated answer before returning it to any UI or API.

    This keeps the pipeline output product-ready and avoids duplicating cleanup
    logic in Streamlit or other frontends.
    """

    if not answer:
        return ""

    cleaned = answer.strip()

    # Remove inline source links or citations if the model produces them.
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

    # Remove referral clauses without deleting useful factual content before them.
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

    # Remove standalone referral/meta lines.
    standalone_referral_patterns = [
        r"(?im)^\s*Refer to .* for more information\.\s*$",
        r"(?im)^\s*For more information,?.*\.\s*$",
        r"(?im)^\s*See the .*documentation.*\.\s*$",
        r"(?im)^\s*See the .*docs.*\.\s*$",
        r"(?im)^\s*Check the .*documentation.*\.\s*$",
    ]

    for pattern in standalone_referral_patterns:
        cleaned = re.sub(pattern, "", cleaned)

    # Remove meta openings while preserving the factual sentence.
    cleaned = re.sub(r"(?im)^\s*Note that\s+", "", cleaned)
    cleaned = re.sub(r"(?im)^\s*Note:\s*", "", cleaned)

    # Remove leftover referral fragments, such as:
    # "Refer to Environments in n8n"
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

    # Fix lowercase sentence starts caused by removing "Note that" or similar openings.
    cleaned = re.sub(r"(?m)^you can", "You can", cleaned)
    cleaned = re.sub(r"(?m)^you must", "You must", cleaned)
    cleaned = re.sub(r"(?m)^you should", "You should", cleaned)
    cleaned = re.sub(r"(?m)^n8n requires", "n8n requires", cleaned)

    # Remove incomplete Markdown tables that only contain a header and separator.
    cleaned = re.sub(
        r"(?ms)\n?\|[^\n]+\|\s*\n\|[\s:\-|]+\|\s*(?=\n\n|\n[A-Z]|\Z)",
        "\n",
        cleaned,
    )

    # Remove empty bullets/numbered lines that may remain after cleanup.
    cleaned = re.sub(r"(?m)^\s*[-*]\s*$", "", cleaned)
    cleaned = re.sub(r"(?m)^\s*\d+\.\s*$", "", cleaned)

    # Normalize spacing.
    cleaned = re.sub(r"\s+([.,;:])", r"\1", cleaned)
    cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
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
- Use only information explicitly supported by the retrieved context. Do not use outside knowledge or guess product behavior.
- Preserve technical tokens exactly as written in the retrieved context, including environment variable names, commands, file names, configuration keys, paths, and code snippets. Do not remove underscores, hyphens, capitalization, or suffixes such as `_FILE`.
- When listing environment variables, copy the variable names exactly from the retrieved context.
- Answer directly. Do not start with phrases like "Based on the retrieved context", "It appears", or "To answer your question".
- Synthesize the retrieved documentation into a direct, self-contained answer.
- Do not include inline source citations. Relevant documentation is shown separately in the source panel.
- Do not tell the user to refer to, see, check, or look at documentation unless the user explicitly asks where to find something.
- Do not write meta-comments about the answer, the retrieved context, or the sources.
- Do not write phrases such as "Note that this answer...", "the retrieved documentation does not...", "this answer does not provide...", or "Source X provides...".
- Do not use sources that are only tangentially related to the user's question.
- Do not create Markdown tables unless the retrieved context contains the complete table rows needed for the answer. Prefer bullet points for partial table information.
- Do not list general AI concepts as product features. Only describe n8n capabilities, nodes, workflow features, or integrations that are explicitly supported by the retrieved context.
- Do not list LLMs themselves as n8n product features unless the retrieved context explicitly says n8n provides the model. When appropriate, say that n8n AI workflows can use LLMs.
- Preserve indentation in code blocks, YAML, JSON, and shell commands exactly when possible.
- For broad/general questions, prioritize sources marked as general. Use scenario-specific sources only when the user asks about that scenario or when clearly presenting them as scenario-specific alternatives.
- For “how does it work” questions, explain the mechanism briefly, not only setup steps.
- If the user asks about a specific environment-variable category, such as database, logs, timezone, security, credentials, or queue mode, focus on that category. Do not explain generic shell syntax unless the user asks how to set variables in a terminal, shell, Bash, cmd.exe, or PowerShell.
- If the retrieved documentation does not contain enough relevant information to answer the question at all, say exactly: "The retrieved documentation does not contain enough information to answer this question."
- Keep the answer practical and easy for a business analyst or product user to understand.
- Answer in English.

User question:
{query}

Retrieved documentation context:
{context}

Answer:
""".strip()


def generate_answer_with_ollama(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    """Generate an answer using a local Ollama model."""

    prompt = build_prompt(query, results)

    response = ollama.chat(
        model=OLLAMA_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        options={
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
        },
        keep_alive=OLLAMA_KEEP_ALIVE,
    )

    return response["message"]["content"].strip()


def generate_answer_with_openai_compatible_api(
    query: str,
    results: list[dict[str, Any]],
) -> str:
    """Generate an answer using any OpenAI-compatible chat completions API.

    This can support OpenAI, OpenRouter, DeepSeek, Together, Groq, etc.,
    as long as the provider exposes an OpenAI-compatible endpoint.
    """

    api_key = os.getenv(OPENAI_COMPATIBLE_API_KEY_ENV)

    if not api_key:
        raise ValueError(f"{OPENAI_COMPATIBLE_API_KEY_ENV} is not set.")

    client = OpenAI(
        api_key=api_key,
        base_url=OPENAI_COMPATIBLE_BASE_URL,
    )

    prompt = build_prompt(query, results)

    response = client.chat.completions.create(
        model=OPENAI_COMPATIBLE_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are a source-grounded internal product knowledge assistant.",
            },
            {
                "role": "user",
                "content": prompt,
            },
        ],
        temperature=TEMPERATURE,
        top_p=TOP_P,
    )

    content = response.choices[0].message.content

    if content is None:
        raise ValueError("The OpenAI-compatible API returned an empty response.")

    return content.strip()


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
    """Generate an answer using the configured backend."""

    if LLM_BACKEND == "ollama":
        try:
            answer = generate_answer_with_ollama(query, results)
            answer = clean_generated_answer(answer)
            return answer, "ollama"
        except Exception as e:
            fallback_answer = build_retrieval_only_answer(results)
            fallback_answer = clean_generated_answer(fallback_answer)
            return fallback_answer, f"retrieval_only_fallback: {e}"

    if LLM_BACKEND == "openai_compatible":
        try:
            answer = generate_answer_with_openai_compatible_api(query, results)
            answer = clean_generated_answer(answer)
            return answer, "openai_compatible"
        except Exception as e:
            fallback_answer = build_retrieval_only_answer(results)
            fallback_answer = clean_generated_answer(fallback_answer)
            return fallback_answer, f"retrieval_only_fallback: {e}"

    if LLM_BACKEND == "retrieval_only":
        answer = build_retrieval_only_answer(results)
        answer = clean_generated_answer(answer)
        return answer, "retrieval_only"

    fallback_answer = build_retrieval_only_answer(results)
    fallback_answer = clean_generated_answer(fallback_answer)
    return fallback_answer, f"retrieval_only_fallback: unsupported backend '{LLM_BACKEND}'"

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

def get_rag_response(
    query: str,
    k: int = FINAL_RETRIEVAL_K,
    chat_history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run retrieval, source metadata enrichment, source prioritization, and generation."""

    direct_response = build_direct_response(query)
    if direct_response is not None:
        return direct_response

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

    answer, mode = generate_answer(retrieval_query, final_results)
    answer = normalize_no_answer_response(answer)

    if is_no_answer_response(answer):
        return build_no_answer_response(
            query=query,
            retrieval_query=retrieval_query,
            previous_user_query=previous_user_query,
            retrieved_results=final_results,
            mode=mode,
        )

    return {
        "query": query,
        "retrieval_query": retrieval_query,
        "previous_user_query": previous_user_query,
        "answer": answer,
        "sources": get_sources(final_results),
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