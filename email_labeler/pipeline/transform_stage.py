"""Transform stage implementation for the ETL pipeline."""

import logging
import time
from datetime import datetime
from typing import Any, List, Optional
from ..progress import SmartBar
from ..email_processor import EmailProcessor
from ..gmail_utils import extract_address, extract_domain, format_classification_headers
from ..llm_service import LLMService
from .base import EmailRecord, EnrichedEmailRecord, PipelineContext, PipelineStage
from .config import TransformConfig


logger = logging.getLogger(__name__)


class TransformStage(PipelineStage):
    """Handles email categorization and enrichment."""

    def __init__(
        self,
        config: TransformConfig,
        llm_service: Optional[LLMService] = None,
        email_processor: Optional[EmailProcessor] = None,
    ):
        """Initialize the transform stage.

        Args:
            config: Transform stage configuration.
            llm_service: Optional LLMService instance. If not provided, creates a new one.
            email_processor: Optional EmailProcessor instance. If not provided, creates a new one.
        """
        super().__init__()
        self.config = config
        self.llm_service = llm_service or LLMService(
            categories=config.categories,
            max_content_length=config.max_content_length,
            system_prompt=config.system_prompt,
            user_prompt=config.user_prompt,
        )
        self.email_processor = email_processor or EmailProcessor()

    def execute(
        self, input_data: List[EmailRecord], context: PipelineContext
    ) -> List[EnrichedEmailRecord]:
        """Transform emails by adding categorization."""
        if not input_data:
            logger.info("No emails to transform")
            return []

        logger.debug(f"Starting transformation of {len(input_data)} emails")
        start_time = datetime.now()

        enriched_emails = []
        success_count = 0
        error_count = 0

        bar = SmartBar("Categorizing...", max=len(input_data), initial_eta_seconds=8.0 * len(input_data))
        bar.start()
        for email in input_data:
            try:
                if context.preview_mode:
                    logger.info(f"PREVIEW: Would categorize email {email.id}: {email.subject}")
                    # Create a mock enriched email for preview
                    enriched = EnrichedEmailRecord(
                        **email.__dict__,
                        category="[Preview Mode]",
                        explanation="[Preview Mode - No actual categorization]",
                        confidence=0.0,
                        processing_time=0.0,
                    )
                else:
                    # dry_run still runs real categorization (sender-shortcut/LLM) so the
                    # would-apply summary reflects real decisions; it just never writes.
                    enriched = self._categorize_email(email, context)

                enriched_emails.append(enriched)
                success_count += 1

            except Exception as e:
                error_count += 1
                error_msg = f"Failed to categorize email {email.id}: {e}"
                logger.error(error_msg)
                context.add_error(error_msg)

                if self.config.skip_on_error :
                    continue
                elif not context.config.continue_on_error:
                    raise

            finally:
                bar.next()

        bar.finish()

        # Update metrics
        elapsed = (datetime.now() - start_time).total_seconds()
        self.metrics["emails_transformed"] = success_count
        self.metrics["transformation_errors"] = error_count
        self.metrics["transformation_time"] = elapsed

        context.add_metric("transform_success_count", success_count)
        context.add_metric("transform_error_count", error_count)
        context.add_metric("transform_time", elapsed)

        logger.info(
            f"Transformed {success_count} emails in {elapsed:.2f} seconds ({error_count} errors)"
        )

        return enriched_emails

    def _categorize_email(
        self, email: EmailRecord, context: PipelineContext
    ) -> EnrichedEmailRecord:
        """Categorize a single email."""
        start_time = time.time()

        shortcut = self._try_sender_shortcut(email, context, start_time)
        if shortcut is not None:
            return shortcut
        context.increment_metric("transform_llm_calls")

        email_content = self._build_email_content(email)

        if context.test_mode:
            # In test mode, use a mock categorization
            category = "Test Category"
            explanation = "Test mode - mock categorization"
        else:
            category, explanation = self.llm_service.categorize_email(email_content)

        # Validate category
        if category not in self.config.categories:
            logger.warning(f"Unknown category '{category}' for email {email.id}, using 'main'")
            category = "main"

        confidence = self._calculate_confidence(category, explanation)
        processing_time = time.time() - start_time

        return EnrichedEmailRecord(
            **email.__dict__,
            category=category,
            explanation=explanation,
            confidence=confidence,
            processing_time=processing_time,
        )

    def _try_sender_shortcut(
        self, email: EmailRecord, context: PipelineContext, start_time: float
    ) -> Optional[EnrichedEmailRecord]:
        """Known-sender shortcut: sender_rules takes precedence over the LLM.

        A rule key may be a full sender address (recruiter@gmail.com) or a registered
        domain (substack.com). The exact address is matched first, so a single sender on
        a personal domain can be routed even when the domain itself has no rule (or a
        different one).

        Returns None (falling through to the LLM) if neither the sender's address nor its
        domain has a rule, or the matched rule's category isn't a configured category. A
        matched-but-invalid address rule does not fall back to the domain rule.
        """
        rules = self.config.sender_rules
        for key in (extract_address(email.sender), extract_domain(email.sender)):
            if key and key in rules:
                rule_category = rules[key]
                break
        else:
            return None

        if rule_category not in self.config.categories:
            return None

        context.increment_metric("transform_sender_shortcut")
        return EnrichedEmailRecord(
            **email.__dict__,
            category=rule_category,
            explanation=f"known sender: {key}",
            confidence=1.0,
            processing_time=time.time() - start_time,
        )

    def _build_email_content(self, email: EmailRecord) -> str:
        """Build the LLM input: header-first (no fixed rules — the LLM judges from
        headers directly), with body inclusion gated by llm_body_mode.
        """
        header_block = format_classification_headers(email.headers)
        email_content = f"Subject: {email.subject}\nFrom: {email.sender}\n{header_block}"

        body = ""
        if self.config.llm_body_mode == "head":
            clean_content = self.email_processor.strip_html(email.content)
            body = "\n".join(clean_content.splitlines()[: self.config.llm_body_head_lines])
        elif self.config.llm_body_mode == "full":
            body = self._smart_truncate(self.email_processor.strip_html(email.content))

        if body:
            email_content += f"\n\n{body}"
        return email_content

    def _smart_truncate(self, content: str) -> str:
        """Keep beginning + end to preserve footer (unsubscribe, signatures) when over length."""
        if len(content) <= self.config.max_content_length:
            return content

        max_len = self.config.max_content_length
        keep_start = int(max_len * 0.7)  # 70% from beginning
        keep_end = int(max_len * 0.3)    # 30% from end
        logger.debug(f"Smart truncated email: kept first {keep_start} and last {keep_end} chars")
        return (
            content[:keep_start]
            + "\n\n...[middle content truncated]...\n\n"
            + content[-keep_end:]
        )

    def _calculate_confidence(self, category: str, explanation: str) -> float:
        """Calculate confidence score for categorization."""
        # Simple heuristic: longer explanations tend to be more confident
        # Categories like "Other" or empty explanations get lower confidence

        if category == "Other":
            base_confidence = 0.5
        elif category in ["Response Needed / High Priority", "Bills"]:
            base_confidence = 0.9
        else:
            base_confidence = 0.7

        # Adjust based on explanation length
        explanation_factor = min(len(explanation) / 200, 1.0) * 0.2

        confidence = min(base_confidence + explanation_factor, 1.0)
        return round(confidence, 2)

    def validate_input(self, input_data: Any) -> bool:
        """Validate stage input."""
        if not isinstance(input_data, list):
            return False

        # Check if all items are EmailRecord instances
        for item in input_data:
            if not isinstance(item, EmailRecord):
                return False

        return True
