"""Email Auto-Labeler package for Gmail."""

from .database import EmailDatabase
from .email_processor import EmailProcessor
from .llm_service import LLMCategorizationError, LLMService
from .metrics import MetricsTracker

__version__ = "2.1.0"
__all__ = [
    "EmailDatabase",
    "EmailProcessor",
    "LLMService",
    "LLMCategorizationError",
    "MetricsTracker",
]
