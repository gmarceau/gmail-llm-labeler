"""Configuration module for email auto-labeler."""

import logging
import os
from pathlib import Path
from typing import Optional

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, ValidationError

# Load environment variables
load_dotenv()


class ConfigError(ValueError):
    """Raised when a config file is missing, malformed, or has unknown keys or
    invalid values (e.g. a typo'd ``paths:`` key, or sender rules whose values
    are not configured categories). Shared by PathConfig (the CONFIG_FILE/env
    entry point) and PipelineConfig.from_yaml."""


def get_default_data_dir() -> Path:
    """Get platform-appropriate default data directory.

    Returns:
        Path to default data directory:
        - Linux/Mac: ~/.local/share/gmail-llm-labeler/
        - Windows: %LOCALAPPDATA%/gmail-llm-labeler/
        - Fallback: ./data/
    """
    if os.name == "posix":  # Unix-like systems
        base = Path.home() / ".local" / "share" / "gmail-llm-labeler"
    elif os.name == "nt":  # Windows
        base = (
            Path(os.getenv("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "gmail-llm-labeler"
        )
    else:
        base = Path("./data")

    return base


def get_default_log_dir() -> Path:
    """Get platform-appropriate default log directory.

    Returns:
        Path to default log directory:
        - Linux/Mac: ~/.local/share/gmail-llm-labeler/logs/
        - Windows: %LOCALAPPDATA%/gmail-llm-labeler/logs/
        - Fallback: ./logs/
    """
    return get_default_data_dir() / "logs"


class PathsConfig(BaseModel):
    """Schema for the top-level `paths:` block of a config file.

    Single source of truth for the set of configurable path keys: PathConfig
    below resolves each of these fields to a concrete absolute path, and
    PipelineConfig.from_yaml validates the block against this model so a
    typo'd key fails fast at boot instead of being silently ignored.
    """

    model_config = ConfigDict(extra="forbid")

    database_file: Optional[str] = None
    llm_log_file: Optional[str] = None
    error_log_file: Optional[str] = None
    test_output_file: Optional[str] = None
    test_summary_file: Optional[str] = None


class PathConfig:
    """Manages configurable file paths for the application.

    Priority order:
    1. Environment variables
    2. YAML configuration file
    3. Default values

    The set of path keys is owned by PathsConfig.model_fields (the `paths:`
    block schema above); this class resolves each of those keys to an absolute
    path — the env var named KEY.upper() wins, then the YAML value, then the
    per-key default. Parent directories are created automatically if they
    don't exist.
    """

    # Default (directory, filename) per path key, keyed by PathsConfig field name.
    # PathsConfig owns the KEY SET; this table only supplies each key's default
    # location. A key added to PathsConfig but missing here fails loudly (KeyError).
    _DEFAULT_LOCATIONS = {
        "database_file": ("data", "email_pipeline.db"),
        "llm_log_file": ("logs", "llm_interactions.json"),
        "error_log_file": ("logs", "categorization_errors.log"),
        "test_output_file": ("data", "test_results.csv"),
        "test_summary_file": ("data", "test_summary.json"),
    }

    def __init__(self, config_file: Optional[str] = None):
        """Initialize path configuration.

        Args:
            config_file: Optional path to YAML config file to load paths from.
        """
        # Load from YAML if provided
        yaml_paths = {}
        if config_file and Path(config_file).exists():
            with open(config_file) as f:
                # An empty file loads as None — same tolerance as from_yaml's `or {}`
                data = yaml.safe_load(f) or {}
            if not isinstance(data, dict):
                raise ConfigError(
                    f"Config file {config_file!r} must contain a top-level mapping, "
                    f"got {type(data).__name__}"
                )
            yaml_paths = data.get("paths") or {}  # a bare `paths:` also loads as None
            if not isinstance(yaml_paths, dict):
                raise ConfigError(
                    f"'paths' block in {config_file!r} must be a mapping, "
                    f"got {type(yaml_paths).__name__}"
                )
            # Validate the key set with the same model from_yaml uses: PathsConfig
            # forbids extra keys, so a typo'd path key fails fast here — named with
            # its file — instead of being silently ignored by the .get(key) lookups
            # in the resolution loop below.
            try:
                PathsConfig(**yaml_paths)
            except ValidationError as e:
                raise ConfigError(f"Invalid 'paths' block in {config_file!r}:\n{e}") from e

        # Get default directories
        default_dirs = {
            "data": get_default_data_dir(),
            "logs": get_default_log_dir(),
        }

        # Configure paths with priority: env var > yaml > default
        for key in PathsConfig.model_fields:
            default_dir, default_file = self._DEFAULT_LOCATIONS[key]
            setattr(
                self,
                key,
                self._resolve_path(
                    os.getenv(key.upper()),
                    yaml_paths.get(key),
                    default_dirs[default_dir] / default_file,
                ),
            )

        # Create directories if they don't exist
        self._ensure_directories()

    def _resolve_path(
        self, env_value: Optional[str], yaml_value: Optional[str], default_value: Path
    ) -> Path:
        """Resolve a path from environment, YAML, or default.

        Args:
            env_value: Value from environment variable
            yaml_value: Value from YAML config
            default_value: Default path value

        Returns:
            Resolved absolute Path object
        """
        if env_value:
            return Path(env_value).expanduser().resolve()
        elif yaml_value:
            return Path(yaml_value).expanduser().resolve()
        else:
            return default_value.resolve()

    def _ensure_directories(self):
        """Create parent directories for all configured paths if they don't exist."""
        for key in PathsConfig.model_fields:
            path = getattr(self, key)
            path.parent.mkdir(parents=True, exist_ok=True)

    def to_dict(self) -> dict:
        """Export configuration as dictionary.

        Returns:
            Dictionary of path configurations as strings
        """
        return {key: str(getattr(self, key)) for key in PathsConfig.model_fields}


# Initialize path configuration
# Check for custom config file from environment
_config_file = os.getenv("CONFIG_FILE")
_path_config = PathConfig(config_file=_config_file)

# Resolved paths, exported as module-level defaults: EmailDatabase, LLMService
# (its interaction/error logs), and MetricsTracker consume these when constructed
# without explicit paths. Key set and resolution are owned by PathsConfig/
# PathConfig above.
DATABASE_FILE = str(_path_config.database_file)
LLM_LOG_FILE = str(_path_config.llm_log_file)
ERROR_LOG_FILE = str(_path_config.error_log_file)
TEST_OUTPUT_FILE = str(_path_config.test_output_file)
TEST_SUMMARY_FILE = str(_path_config.test_summary_file)

# Setup logging
log_level = os.getenv("LOG_LEVEL", "INFO")
logging.basicConfig(
    level=getattr(logging, log_level.upper(), logging.INFO),
    format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
)

# LLM wiring, consumed at call time by email_labeler/llm_service.py (LLMService):
# which backend to call (LLM_SERVICE), where (OLLAMA_BASE_URL / OPENAI_API_KEY),
# the model fallbacks when a caller doesn't pass model= (OPENAI_MODEL /
# OLLAMA_MODEL), and the gpt-oss reasoning-effort level (GPT_OSS_REASONING).
# NOT v1 leftovers — the pipeline's LLMService reads these even though
# TransformConfig also carries llm_service/model fields; wiring those yaml
# fields through to LLMService is tracked in a follow-up bead.
LLM_SERVICE = os.getenv("LLM_SERVICE", "OpenAI")  # "OpenAI" or "Ollama"

# OpenAI configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

# Ollama configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434/v1")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1")
GPT_OSS_REASONING = os.getenv("GPT_OSS_REASONING", "medium")
