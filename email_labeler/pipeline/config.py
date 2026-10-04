"""Configuration classes for the ETL pipeline.

These are pydantic models validated at load time. Unknown/typo'd keys are
rejected (``extra="forbid"``) so a misspelled setting (e.g. ``sender_rule:``
instead of ``sender_rules:``) fails fast at boot with a clear error instead of
being silently swallowed into a default. Fields with a sensible default stay
optional; only genuinely required values must be supplied.
"""

import os
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, ConfigDict, ValidationError

from ..config import PathsConfig


class ConfigError(ValueError):
    """Raised when a pipeline config file is missing, malformed, or has unknown keys
    or invalid values (e.g. sender rules whose values are not configured categories)."""


class ExtractConfig(BaseModel):
    """Configuration for the Extract stage."""

    model_config = ConfigDict(extra="forbid")

    source: str = "gmail"  # Options: "gmail", "database"
    gmail_query: str = "is:unread"
    batch_size: int = 100
    max_results: Optional[int] = None


class EscalationConfig(BaseModel):
    """Tiered body escalation: a second LLM pass that adds body head.

    Production classifies from headers alone (llm_body_mode=none), but cold
    outreach is engineered to have no distinguishing headers — the tell-tale
    cues (Series A, equity, "brief chat") live in the body. When the first pass
    returns `main` for an unruled company-domain sender, re-classify once with
    the first `body_head_lines` lines of the body included.
    """

    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    body_head_lines: int = 10


class TransformConfig(BaseModel):
    """Configuration for the Transform stage."""

    # validate_assignment: from_yaml loads the external rules file into
    # sender_rules/personal_domains AFTER construction; re-validating those
    # assignments makes a wrongly-typed rules file fail at load, not at runtime.
    model_config = ConfigDict(extra="forbid", validate_assignment=True)
    llm_service: str = "openai"  # Options: "openai", "ollama"
    model: str = "gpt-4o-mini"
    temperature: float = 0.0  # Sampling temperature; 0 for deterministic classification
    max_content_length: int = 4000
    timeout: int = 30
    skip_on_error: bool = True
    categories: List[str] = [
        "Marketing",
        "Response Needed / High Priority",
        "Bills",
        "Subscriptions",
        "Newsletters",
        "Personal",
        "Work",
        "Events",
        "Travel",
        "Receipts",
        "Low quality",
        "Notifications",
        "Other",
    ]
    system_prompt: Optional[str] = None  # Custom system prompt (supports templating)
    user_prompt: Optional[str] = None  # Custom user prompt (supports templating)
    sender_rules: Dict[str, str] = {}  # full address or registered domain -> category
    # Freemail/personal domains: in `review-senders` these are broken out per individual
    # sender (you'd never rule the whole domain), rather than collapsed into one domain row.
    personal_domains: List[str] = []
    # Optional external file (path relative to the main config) holding top-level
    # `sender_rules:` and `personal_domains:`. When set, from_yaml loads it into the two
    # fields above so the growing reviewed-rules list stays out of the main pipeline config.
    sender_rules_file: Optional[str] = None
    llm_body_mode: str = "none"  # Options: "none", "head", "full"
    llm_body_head_lines: int = 20  # used when llm_body_mode == "head"
    escalation: EscalationConfig = EscalationConfig()


GMAIL_TAB_LABEL_IDS = {
    "primary": "CATEGORY_PERSONAL",
    "updates": "CATEGORY_UPDATES",
    "forums": "CATEGORY_FORUMS",
    "promotions": "CATEGORY_PROMOTIONS",
    "social": "CATEGORY_SOCIAL",
}


class LoadConfig(BaseModel):
    """Configuration for the Load stage."""

    model_config = ConfigDict(extra="forbid")

    apply_labels: bool = True
    create_missing_labels: bool = True
    category_actions: Dict[str, List[str]] = {
        "Marketing": ["apply_label", "archive"],
        "Response Needed / High Priority": ["apply_label", "star"],
        "Bills": ["apply_label", "star"],
        "Newsletters": ["apply_label", "archive"],
        "Low quality": ["apply_label", "archive", "mark_as_read"],
        "Notifications": ["apply_label", "mark_as_read"],
    }
    default_actions: List[str] = ["apply_label"]
    category_tab_map: Dict[str, str] = {}


class SyncConfig(BaseModel):
    """Configuration for the Sync stage."""

    model_config = ConfigDict(extra="forbid")

    database_path: str = "email_pipeline.db"
    save_metrics: bool = True
    track_history: bool = True
    batch_size: int = 100
    track_metrics: bool = True


class MonitoringConfig(BaseModel):
    """Configuration for monitoring and observability."""

    model_config = ConfigDict(extra="forbid")

    log_level: str = "INFO"
    metrics_export: str = "json"  # Options: "json", "csv", "prometheus"
    metrics_path: str = "pipeline_metrics.json"
    enable_tracing: bool = False


class TopLevelConfig(BaseModel):
    """Top level of a config file: a `pipeline:` mapping and a `paths:` block.

    Unknown top-level keys are rejected here (extra="forbid") and the `paths:`
    block is validated by the nested PathsConfig model (owned by
    email_labeler.config, beside its consumer PathConfig). The `pipeline`
    mapping is validated next by PipelineConfig itself: its own
    extra="forbid" rejects unknown pipeline keys, so no hand-maintained
    allowlist of pipeline keys (which would have to track the model fields
    in lockstep) is needed.
    """

    model_config = ConfigDict(extra="forbid")

    pipeline: Optional[dict] = None
    paths: Optional[PathsConfig] = None


class PipelineConfig(BaseModel):
    """Main pipeline configuration."""

    model_config = ConfigDict(extra="forbid")

    extract: ExtractConfig = ExtractConfig()
    transform: TransformConfig = TransformConfig()
    load: LoadConfig = LoadConfig()
    sync: SyncConfig = SyncConfig()
    monitoring: MonitoringConfig = MonitoringConfig()
    dry_run: bool = False
    continue_on_error: bool = True
    max_retries: int = 3

    @classmethod
    def from_yaml(cls, path: str) -> "PipelineConfig":
        """Load configuration from YAML file.

        An empty file, or a missing/null `pipeline:` block, loads all defaults.

        Raises:
            ConfigError: if the file is missing/unreadable, or if it contains an
                unknown/typo'd key, a wrongly-typed value (including in the external
                sender-rules file), or a sender rule whose value is not a configured
                category. The message names the config file (and the rules file, when
                the offender came from it) and the offending field, so a boot-time
                failure is self-explanatory.
        """
        try:
            with open(path) as f:
                data = yaml.safe_load(f) or {}
        except OSError as e:
            raise ConfigError(f"Could not read config file {path!r}: {e}") from e

        if not isinstance(data, dict):
            raise ConfigError(
                f"Config file {path!r} must contain a top-level mapping, got {type(data).__name__}"
            )

        # The file's shape is declared by models: TopLevelConfig rejects unknown
        # top-level keys and validates the `paths:` block, then PipelineConfig's own
        # extra="forbid" rejects unknown keys under `pipeline`. A missing/null
        # `pipeline:` block (or an empty file) loads all defaults, via `or {}`.
        try:
            top = TopLevelConfig(**data)
            config = cls(**(top.pipeline or {}))
        except ValidationError as e:
            raise ConfigError(f"Invalid config in {path!r}:\n{e}") from e

        # An external sender-rules file (path relative to this config) is the source of
        # truth for sender_rules/personal_domains when present.
        rules_source = path  # where the effective sender_rules came from (for errors)
        if config.transform.sender_rules_file:
            rules_path = os.path.join(
                os.path.dirname(os.path.abspath(path)), config.transform.sender_rules_file
            )
            try:
                with open(rules_path) as rf:
                    rules_data = yaml.safe_load(rf) or {}
            except OSError as e:
                raise ConfigError(
                    f"Could not read sender_rules_file {config.transform.sender_rules_file!r} "
                    f"(resolved to {rules_path!r}, from {path!r}): {e}"
                ) from e
            if not isinstance(rules_data, dict):
                raise ConfigError(
                    f"sender_rules_file {rules_path!r} must contain a top-level mapping"
                )
            unknown_rules = set(rules_data) - {"sender_rules", "personal_domains"}
            if unknown_rules:
                raise ConfigError(
                    f"Unknown key(s) in sender_rules_file {rules_path!r}: "
                    f"{', '.join(sorted(unknown_rules))}. "
                    "Expected only 'sender_rules' and 'personal_domains'."
                )
            # TransformConfig validates on assignment, so a sender_rules that is a
            # list/string (or personal_domains as a dict/string) fails right here,
            # named with the rules file. The `or {}`/`or []` keep empty values tolerated.
            try:
                config.transform.sender_rules = rules_data.get("sender_rules") or {}
                config.transform.personal_domains = rules_data.get("personal_domains") or []
            except ValidationError as e:
                raise ConfigError(
                    f"Invalid sender_rules_file {rules_path!r} (from {path!r}):\n{e}"
                ) from e
            rules_source = rules_path

        # Every rule VALUE must be one of the configured categories, case-sensitively:
        # the runtime sender-rule lookups are case-sensitive, so e.g. a capitalized
        # "Marketing" against a lowercase "marketing" category would silently never
        # match. List every offender so one load error shows the whole cleanup needed.
        invalid_rules = sorted(
            f"{sender}: {category}"
            for sender, category in config.transform.sender_rules.items()
            if category not in config.transform.categories
        )
        if invalid_rules:
            source = (
                f"{rules_source!r} (referenced from {path!r})"
                if rules_source != path
                else f"{rules_source!r}"
            )
            raise ConfigError(
                f"Invalid sender_rules in {source}: rule values must be one of the "
                f"configured categories {config.transform.categories} (case-sensitive). "
                f"Offending rule(s):\n  " + "\n  ".join(invalid_rules)
            )

        return config

    @classmethod
    def from_env(cls) -> "PipelineConfig":
        """Create configuration from environment variables and defaults."""
        config = cls()

        # Override with environment variables if present
        llm_service = os.getenv("LLM_SERVICE")
        if llm_service:
            config.transform.llm_service = llm_service.lower()

        openai_model = os.getenv("OPENAI_MODEL")
        if openai_model:
            config.transform.model = openai_model
        else:
            ollama_model = os.getenv("OLLAMA_MODEL")
            if ollama_model:
                config.transform.model = ollama_model

        database_path = os.getenv("DATABASE_PATH")
        if database_path:
            config.sync.database_path = database_path

        log_level = os.getenv("LOG_LEVEL")
        if log_level:
            config.monitoring.log_level = log_level

        return config

    def to_yaml(self, path: str):
        """Save configuration to YAML file."""
        data = {
            "pipeline": {
                "dry_run": self.dry_run,
                "continue_on_error": self.continue_on_error,
                "max_retries": self.max_retries,
                "extract": {
                    "source": self.extract.source,
                    "gmail_query": self.extract.gmail_query,
                    "batch_size": self.extract.batch_size,
                    "max_results": self.extract.max_results,
                },
                "transform": {
                    "llm_service": self.transform.llm_service,
                    "model": self.transform.model,
                    "temperature": self.transform.temperature,
                    "max_content_length": self.transform.max_content_length,
                    "timeout": self.transform.timeout,
                    "skip_on_error": self.transform.skip_on_error,
                    "categories": self.transform.categories,
                    "system_prompt": self.transform.system_prompt,
                    "user_prompt": self.transform.user_prompt,
                    # An external file owns the rules when set; otherwise inline them.
                    **(
                        {"sender_rules_file": self.transform.sender_rules_file}
                        if self.transform.sender_rules_file
                        else {
                            "sender_rules": self.transform.sender_rules,
                            "personal_domains": self.transform.personal_domains,
                        }
                    ),
                    "llm_body_mode": self.transform.llm_body_mode,
                    "llm_body_head_lines": self.transform.llm_body_head_lines,
                    "escalation": {
                        "enabled": self.transform.escalation.enabled,
                        "body_head_lines": self.transform.escalation.body_head_lines,
                    },
                },
                "load": {
                    "apply_labels": self.load.apply_labels,
                    "create_missing_labels": self.load.create_missing_labels,
                    "category_actions": self.load.category_actions,
                    "default_actions": self.load.default_actions,
                    "category_tab_map": self.load.category_tab_map,
                },
                "sync": {
                    "database_path": self.sync.database_path,
                    "save_metrics": self.sync.save_metrics,
                    "track_history": self.sync.track_history,
                    "batch_size": self.sync.batch_size,
                    "track_metrics": self.sync.track_metrics,
                },
                "monitoring": {
                    "log_level": self.monitoring.log_level,
                    "metrics_export": self.monitoring.metrics_export,
                    "metrics_path": self.monitoring.metrics_path,
                    "enable_tracing": self.monitoring.enable_tracing,
                },
            }
        }

        with open(path, "w") as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
