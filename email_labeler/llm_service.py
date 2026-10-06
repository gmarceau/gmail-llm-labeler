"""LLM service for email categorization."""

import json
import logging
import os
import shutil
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import httpx
from openai import OpenAI
from plumbum import local

from .config import (
    ERROR_LOG_FILE,
    LLM_LOG_FILE,
    OLLAMA_BASE_URL,
    OPENAI_API_KEY,
)


class LLMCategorizationError(Exception):
    """Exception raised when LLM categorization fails."""

    pass


# Default prompts
DEFAULT_SYSTEM_PROMPT = "You are an email categorization assistant. Always respond with a valid JSON object containing 'category' and 'explanation' fields."

DEFAULT_SYSTEM_PROMPT_GPT_OSS = """Reasoning: {reasoning}
You are an email categorization assistant.An email that is a notification should always be categorized as 'Notifications'.Always respond with a valid JSON object containing 'category' and 'explanation' fields."""

DEFAULT_USER_PROMPT = """Categorize this email into exactly ONE of these categories:

{categories}

Email content:
{email_content}

Respond with a JSON object:
{{
    "explanation": "<brief reason for this categorization>",
    "category": "<exact category name from the list>",

}}"""


# The two supported backends. Which one to call is pipeline config
# (TransformConfig.llm_service -> LLMService(service=...)), not environment.
_SUPPORTED_SERVICES = ("openai", "ollama")

# Model used when the caller doesn't pass one. The pipeline always passes
# config.transform.model; these defaults only serve standalone LLMService use.
_DEFAULT_MODELS = {"openai": "gpt-4o-mini", "ollama": "llama3.1"}

# Absolute fallbacks for scheduled-run environments whose PATH omits
# Homebrew (observed: PATH had ~/.cargo/bin but not /opt/homebrew/bin,
# so `local["ollama"]` raised CommandNotFound while ollama sat installed).
_OLLAMA_BIN_CANDIDATES = (
    "/opt/homebrew/bin/ollama",  # macOS Apple Silicon (Homebrew)
    "/usr/local/bin/ollama",  # macOS Intel (Homebrew) / manual install
    "/usr/bin/ollama",  # Linux distro package
)


def _resolve_ollama_binary() -> str:
    """Find the ollama executable, falling back to absolute paths when PATH is stripped.

    Raises RuntimeError naming the searched locations when nothing is found,
    so the error says "ollama binary not found" instead of a bare CommandNotFound.
    """
    found = shutil.which("ollama")
    if found:
        return found
    for candidate in _OLLAMA_BIN_CANDIDATES:
        if os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "ollama binary not found: not on PATH and none of "
        + ", ".join(_OLLAMA_BIN_CANDIDATES)
        + " are executable. Install ollama or add it to PATH."
    )


class LLMService:
    """Handles email categorization using LLM (OpenAI or Ollama)."""

    def __init__(
        self,
        categories: List[str],
        max_content_length: int = 4000,
        llm_client: Optional[OpenAI] = None,
        model: Optional[str] = None,
        service: str = "openai",
        gpt_oss_reasoning: str = "medium",
        lazy_init: bool = False,
        system_prompt: Optional[str] = None,
        user_prompt: Optional[str] = None,
        temperature: float = 0.0,
        timeout: int = 30,
    ):
        """Initialize the LLM client.

        Args:
            categories: List of category labels for email classification.
            max_content_length: Maximum length of email content before truncation.
            llm_client: Optional OpenAI client instance. If not provided, creates
                one for the selected service at init (or at first use, if lazy).
            model: Optional model name. If not provided, uses the per-service
                default — the pipeline always passes config.transform.model.
            service: Which backend to call: "openai" or "ollama" (case-insensitive).
                Authoritative — comes from TransformConfig.llm_service in the pipeline.
            gpt_oss_reasoning: Reasoning-effort level ("low"/"medium"/"high") rendered
                into the default gpt-oss system prompt; unused with a custom
                system_prompt. Mirrors TransformConfig.gpt_oss_reasoning.
            lazy_init: If True, delay LLM client initialization until first use.
            system_prompt: Optional custom system prompt with template support.
            user_prompt: Optional custom user prompt with template support.
            temperature: Sampling temperature.
            timeout: Request timeout (seconds) for the constructed client — a hung
                call fails fast instead of parking the pipeline forever. Mirrors the
                TransformConfig.timeout default; the pipeline always passes the
                config value.
        """
        service = service.lower()
        if service not in _SUPPORTED_SERVICES:
            raise ValueError(
                f"Unknown LLM service {service!r}: expected one of {sorted(_SUPPORTED_SERVICES)}"
            )
        self.service = service
        self.categories = categories
        self.max_content_length = max_content_length
        self._lazy_init = lazy_init
        self.system_prompt = system_prompt
        self.user_prompt = user_prompt
        self.temperature = temperature
        self.timeout = timeout
        self.gpt_oss_reasoning = gpt_oss_reasoning
        self.model = model or _DEFAULT_MODELS[service]
        self.llm_client: Optional[OpenAI] = llm_client
        if self.llm_client is None and not lazy_init:
            self.llm_client = self._get_llm_client()

    def _ensure_llm_client(self):
        """Ensure LLM client is initialized (for lazy initialization)."""
        if self.llm_client is None and self._lazy_init:
            self.llm_client = self._get_llm_client()

    def _ensure_ollama_running(self):
        """Start ollama serve if it's not already reachable."""
        try:
            httpx.get(OLLAMA_BASE_URL.replace("/v1", ""), timeout=2)
            return  # Already running
        except Exception:
            pass

        logging.info("Ollama not reachable, starting ollama serve...")
        local[_resolve_ollama_binary()].popen(["serve"])

        # Wait up to 10s for it to become ready
        for _ in range(20):
            time.sleep(0.5)
            try:
                httpx.get(OLLAMA_BASE_URL.replace("/v1", ""), timeout=1)
                logging.info("Ollama started successfully.")
                return
            except Exception:
                pass

        raise RuntimeError("Timed out waiting for ollama serve to start")

    def _get_llm_client(self) -> OpenAI:
        """Get the client for the selected service (self.service).

        Both paths pass self.timeout: a client without a request timeout parks
        its thread in select() forever when the call never returns (observed:
        a scheduled run hung for 27 days on one such call).
        """
        if self.service == "ollama":
            self._ensure_ollama_running()
            logging.debug(f"Using Ollama at {OLLAMA_BASE_URL} with model {self.model}")
            return OpenAI(  # Dummy key for Ollama
                base_url=OLLAMA_BASE_URL, api_key="ollama", timeout=self.timeout
            )
        else:
            logging.info(f"Using OpenAI with model {self.model}")
            return OpenAI(api_key=OPENAI_API_KEY, timeout=self.timeout)

    def _render_template(self, template: str, variables: Dict[str, str]) -> str:
        """Render a template string with provided variables.

        Args:
            template: Template string with {variable} placeholders.
            variables: Dictionary of variable names to values.

        Returns:
            Rendered template string.
        """
        try:
            return template.format(**variables)
        except KeyError as e:
            logging.warning(f"Template variable {e} not found, using empty string")
            # Try again with missing variables as empty strings
            import re

            var_names = re.findall(r"\{(\w+)\}", template)
            safe_vars = {k: variables.get(k, "") for k in var_names}
            return template.format(**safe_vars)

    def categorize_email(self, email_content: str) -> Tuple[str, str]:
        """
        Categorizes an email using the configured LLM.
        Returns tuple of (category, explanation)
        Raises LLMCategorizationError if the LLM service fails.
        """
        self._ensure_llm_client()

        # Extract subject for debugging (email_content format: "Subject: ...\nFrom: ...\n\n...")
        subject = "Unknown"
        if email_content.startswith("Subject: "):
            subject_line = email_content.split("\n", 1)[0]
            subject = subject_line.replace("Subject: ", "").strip()

        # Smart truncation: keep beginning + end to preserve footer (unsubscribe, signatures)
        if len(email_content) > self.max_content_length:
            max_len = self.max_content_length
            keep_start = int(max_len * 0.6)  # 60% from beginning
            keep_end = int(max_len * 0.4)    # 40% from end
            email_content = (
                email_content[:keep_start]
                + "\n\n...[middle content truncated]...\n\n"
                + email_content[-keep_end:]
            )
            logging.debug(f"Smart truncated email: kept first {keep_start} and last {keep_end} chars")

        # Build messages
        messages = self._build_messages(email_content)

        try:
            # Make API call
            response = self._call_llm(messages, email_content)

            # Parse and validate response (pass subject for logging)
            category, explanation = self._parse_response(response, subject)

            return category, explanation

        except Exception as e:
            logging.error(f"Error in LLM categorization with {self.model}: {str(e)}")
            logging.exception("Full exception details:")
            self._log_error(email_content, str(e))
            raise LLMCategorizationError(f"LLM categorization failed: {str(e)}") from e

    def _build_messages(self, email_content: str) -> list:
        """Build messages for the LLM based on the service type."""
        messages = []

        # Prepare template variables
        template_vars = {
            "categories": ", ".join(self.categories),
            "email_content": email_content,
            "reasoning": self.gpt_oss_reasoning,
        }

        # Determine which system prompt to use
        if self.system_prompt:
            # Use custom system prompt
            system_content = self._render_template(self.system_prompt, template_vars)
        elif self.service == "ollama" and "gpt-oss" in self.model:
            # Use default GPT-OSS system prompt
            system_content = self._render_template(DEFAULT_SYSTEM_PROMPT_GPT_OSS, template_vars)
        else:
            # Use default system prompt
            system_content = self._render_template(DEFAULT_SYSTEM_PROMPT, template_vars)

        messages.append({"role": "system", "content": system_content})

        # User prompt
        if self.user_prompt:
            # Use custom user prompt
            user_content = self._render_template(self.user_prompt, template_vars)
        else:
            # Use default user prompt
            user_content = self._render_template(DEFAULT_USER_PROMPT, template_vars)

        messages.append({"role": "user", "content": user_content})
        return messages

    def _call_llm(self, messages: list, email_content: str) -> str:
        """Make the API call to the LLM."""
        start_time = time.time()

        # Prepare completion kwargs
        completion_kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": 500,
        }

        # Add response_format for OpenAI (not supported by Ollama)
        if self.service == "openai":
            completion_kwargs["response_format"] = {"type": "json_object"}

        logging.debug(f"Calling {self.service} API with model {self.model}")
        assert self.llm_client is not None, "LLM client must be initialized"
        response = self.llm_client.chat.completions.create(**completion_kwargs)  # type: ignore[call-overload]

        end_time = time.time()

        # Log the interaction
        self._log_interaction(start_time, end_time, response.choices[0].message.content, email_content)

        return response.choices[0].message.content  # type: ignore[no-any-return]

    def _parse_response(self, response_text: str, subject: str = "Unknown") -> Tuple[str, str]:
        """Parse and validate the LLM response.

        Empty, unparseable, or unknown-category responses raise ValueError;
        categorize_email turns that into an error-log entry and an
        LLMCategorizationError — never a silently invented category.
        """
        response_text = response_text.strip()
        logging.debug(f"Subject: {subject}")
        logging.debug(f"LLM response: {response_text[:500]}")

        # Try to parse as JSON
        try:
            response_json = json.loads(response_text)
            category = response_json.get("category", "").strip()
            explanation = response_json.get("explanation", "").strip()
            logging.debug(f"Categorized as: {category} - {explanation}")
        except json.JSONDecodeError as e:
            logging.warning("Failed to parse JSON response, attempting text extraction")
            # Fallback: try to extract category from text
            for label in self.categories:
                if label.lower() in response_text.lower():
                    logging.info(f"Extracted category '{label}' from non-JSON response")
                    return label, "Extracted from response"
            raise ValueError(
                f"Unparseable LLM response (not JSON, no configured category name in text): "
                f"{response_text[:200]!r}"
            ) from e

        # Validate category
        if category in self.categories:
            return category, explanation

        if not category:
            # An empty 'category' field fuzzy-matches the first configured label
            # ("" is "in" every string) — a silent invention, not a match.
            raise ValueError("LLM response JSON has an empty 'category' field")

        # Try fuzzy matching
        category_lower = category.lower()
        for label in self.categories:
            if label.lower() in category_lower or category_lower in label.lower():
                logging.info(f"Fuzzy matched '{category}' to '{label}'")
                return label, explanation
        logging.warning(f"Category '{category}' not in predefined list")
        raise ValueError(
            f"Unknown category {category!r}: not among configured categories {self.categories}"
        )

    def _log_interaction(self, start_time: float, end_time: float, response: str, email_content: str):
        """Log the LLM interaction for debugging."""
        log_entry = {
            "request_timestamp": start_time,
            "response_timestamp": end_time,
            "duration": end_time - start_time,
            "model": self.model,
            "service": self.service,
            "response": response,
            "processed_email": email_content,
        }
        with open(LLM_LOG_FILE, "a") as f:
            f.write(json.dumps(log_entry, indent=2) + "\n")

    def _log_error(self, email_content: str, error: str):
        """Log categorization errors for debugging."""
        error_entry = {
            "timestamp": datetime.now().isoformat(),
            "model": self.model,
            "error": error,
            "email_preview": email_content[:500] if email_content else "No content",
        }
        with open(ERROR_LOG_FILE, "a") as f:
            f.write(json.dumps(error_entry) + "\n")
