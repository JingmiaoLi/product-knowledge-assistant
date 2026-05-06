
from __future__ import annotations

from typing import Any


def infer_source_type(source_info: dict[str, Any]) -> str:
    """Infer a coarse source type from source path/title/section.

    This is used for UI labels and lightweight query-aware prioritization.
    """

    source = str(source_info.get("source", "")).lower()
    title = str(source_info.get("title", "")).lower()
    section = str(source_info.get("section", "")).lower()

    text = f"{source} {title} {section}"

    if "hosting/installation/" in source or "/installation/" in source:
        if "docker" in text or "docker-compose" in text or "compose" in text:
            return "docker_hosting"
        if "kubernetes" in text or "helm" in text or "k8s" in text:
            return "kubernetes_hosting"
        if "npm" in text:
            return "npm_installation"
        return "installation"

    if "hosting/configuration/environment-variables/" in source:
        return "environment_variables"

    if "source-control-environments/" in source:
        return "source_control"

    if "user-management/" in source:
        return "user_management"

    if "advanced-ai/" in source:
        return "advanced_ai"

    if "privacy-security/" in source or "privacy" in text or "security" in text:
        return "privacy_security"

    if "workflows/" in source:
        return "workflows"

    return "general"


def describe_source_scope(source_type: str) -> str:
    """Return a short human-readable description for a source type."""

    scope_map = {
        "general": "general product documentation",
        "installation": "installation documentation",
        "npm_installation": "npm installation documentation",
        "docker_hosting": "Docker-specific hosting setup",
        "kubernetes_hosting": "Kubernetes-specific hosting setup",
        "environment_variables": "environment variables configuration reference",
        "source_control": "source control and environments documentation",
        "user_management": "user management and permissions documentation",
        "advanced_ai": "AI features and AI workflow documentation",
        "privacy_security": "privacy and security documentation",
        "workflows": "workflow-related documentation",
    }

    return scope_map.get(source_type, "general product documentation")


def _contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def infer_query_intent(query: str) -> str:
    """Infer a coarse intent label from the user query."""

    q = query.lower().strip()

    if _contains_any(q, ["source control", "git branch", "git repo", "push and pull", "environments"]):
        return "source_control"

    if _contains_any(q, ["permission", "permissions", "role", "roles", "rbac", "access control"]):
        return "permissions"

    if _contains_any(q, ["install n8n", "how to install", "installation", "install "]):
        return "installation"

    if _contains_any(q, ["database environment variable", "database env", "dbtype", "dbpostgres", "postgres"]):
        return "database_environment_variables"

    if _contains_any(q, ["environment variable", "environment variables", "env var", "env vars", ".env"]):
        return "environment_variables"

    if _contains_any(q, ["ai feature", "ai features", "artificial intelligence", "ai agent", "llm"]):
        return "ai_features"

    return "general"


def _intent_priority(intent: str, source: dict[str, Any]) -> tuple[int, int, int]:
    """Return a sortable priority tuple for one source.

    Lower tuple values mean higher priority.
    The tuple structure is:
    1. primary intent match bucket
    2. secondary preference bucket
    3. original index bucket (preserved outside this function)
    """

    source_path = str(source.get("source", "")).lower()
    title = str(source.get("title", "")).lower()
    section = str(source.get("section", "")).lower()
    category = str(source.get("category", "")).lower()
    source_type = str(source.get("source_type", "")).lower()

    text = f"{source_path} {title} {section} {category} {source_type}"

    # Default bucket: keep things in the middle unless boosted or penalized.
    primary = 5
    secondary = 5

    if intent == "environment_variables":
        if "hosting/configuration/environment-variables/" in source_path:
            primary = 0
            secondary = 0 if source_path.endswith("/index.md") else 1
        elif "hosting/configuration/" in source_path:
            primary = 1
            secondary = 0
        elif "hosting/installation/" in source_path:
            primary = 3
            secondary = 0
        elif "advanced-ai/" in source_path:
            primary = 8
            secondary = 0

    elif intent == "database_environment_variables":
        if source_path.endswith("hosting/configuration/environment-variables/database.md"):
            primary = 0
            secondary = 0 if "postgresql" in text else 1
        elif "supported-databases-settings.md" in source_path:
            primary = 1
            secondary = 0
        elif "hosting/configuration/environment-variables/" in source_path:
            primary = 2
            secondary = 0
        elif "hosting/architecture/database-structure.md" in source_path:
            primary = 4
            secondary = 0
        else:
            primary = 7
            secondary = 0

    elif intent == "installation":
        if source_path.endswith("hosting/installation/npm.md"):
            primary = 0
            secondary = 0
        elif "hosting/installation/" in source_path:
            primary = 1
            secondary = 0
        elif "hosting/configuration/" in source_path:
            primary = 4
            secondary = 0
        else:
            primary = 7
            secondary = 0

    elif intent == "permissions":
        if "user-management/" in source_path:
            primary = 0
            secondary = 0 if "rbac" in source_path else 1
        elif "workflows/sharing" in source_path or "permission" in text:
            primary = 1
            secondary = 0
        elif "advanced-ai/chat-hub" in source_path:
            primary = 3
            secondary = 0
        elif "advanced-ai/" in source_path:
            primary = 7
            secondary = 0
        else:
            primary = 5
            secondary = 0

    elif intent == "source_control":
        if "source-control-environments/" in source_path:
            primary = 0
            secondary = 0
        elif source_path.endswith("hosting/configuration/environment-variables/source-control.md"):
            primary = 1
            secondary = 0
        elif "hosting/configuration/" in source_path:
            primary = 3
            secondary = 0
        else:
            primary = 6
            secondary = 0

    elif intent == "ai_features":
        if "advanced-ai/" in source_path:
            primary = 0
            secondary = 0
        elif source_path.endswith("workflows/templates.md"):
            primary = 1
            secondary = 0
        elif "privacy-security/" in source_path or "privacy" in text:
            primary = 3
            secondary = 0
        else:
            primary = 6
            secondary = 0

    else:
        # General queries: prefer broader/general docs over highly scenario-specific setup docs.
        if source_type in {"general", ""}:
            primary = 0
            secondary = 0
        elif source_type in {"environment_variables", "source_control", "user_management", "advanced_ai", "workflows"}:
            primary = 1
            secondary = 0
        elif source_type in {"installation", "npm_installation"}:
            primary = 2
            secondary = 0
        elif source_type in {"docker_hosting", "kubernetes_hosting"}:
            primary = 4
            secondary = 0
        else:
            primary = 5
            secondary = 0

    return primary, secondary, 0


def prioritize_sources_for_query(
    query: str,
    sources: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Re-rank retrieved sources using lightweight query-aware heuristics.

    This does not discard results completely. It just moves more relevant
    source families to the top in a deterministic, explainable way.
    """

    intent = infer_query_intent(query)

    indexed_sources = []

    for index, source in enumerate(sources):
        primary, secondary, _ = _intent_priority(intent, source)

        indexed_sources.append(
            (
                primary,
                secondary,
                index,   # preserve original retrieval order within same bucket
                source,
            )
        )

    indexed_sources.sort(key=lambda item: (item[0], item[1], item[2]))

    return [item[-1] for item in indexed_sources]