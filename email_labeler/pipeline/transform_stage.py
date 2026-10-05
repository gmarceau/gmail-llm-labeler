"""Transform stage implementation for the ETL pipeline."""

import logging
import time
from datetime import datetime
from typing import Any, List, Optional, Tuple

from ..email_processor import EmailProcessor
from ..gmail_utils import (
    compute_sender_signals,
    extract_address,
    extract_domain,
    format_classification_headers,
    format_signals_block,
    strip_reply_prefix,
)
from ..llm_service import LLMService
from ..progress import SmartBar
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
            model=config.model,
            temperature=config.temperature,
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

                if self.config.skip_on_error:
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

        email_content, signals_injected = self._build_email_content(email)
        if signals_injected:
            context.increment_metric("transform_signals_injected")

        if context.test_mode:
            # In test mode, use a mock categorization
            category = "Test Category"
            explanation = "Test mode - mock categorization"
        else:
            category, explanation = self.llm_service.categorize_email(email_content)

            # Tiered escalation: a header-only "main" for an unruled company-domain
            # sender is exactly the recruiter/cold-mail leak — re-classify once with
            # the body head, where the tell-tale cues live. Max one extra pass.
            # "main" is the production personal-correspondence category name —
            # hardcoded by design: this gate is coupled to the configured category
            # naming, so renaming the category means updating this trigger too.
            if category == "main" and self._should_escalate(email):
                context.increment_metric("transform_escalation_second_pass")
                # The second pass rebuilds content directly; its signals flag is
                # discarded — the block is identical to the first pass's, and the
                # metric counts one injection per email.
                escalated_content, _ = self._build_email_content(
                    email,
                    body_mode="head",
                    body_head_lines=self.config.escalation.body_head_lines,
                )
                category, explanation = self.llm_service.categorize_email(escalated_content)

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

    def _build_email_content(
        self,
        email: EmailRecord,
        body_mode: Optional[str] = None,
        body_head_lines: Optional[int] = None,
    ) -> Tuple[str, bool]:
        """Build the LLM input: header-first (no fixed rules — the LLM judges from
        headers directly), with body inclusion gated by llm_body_mode.

        For header-poor mail a compact "Signals:" block of deterministic facts is
        appended (advisory only — the LLM still judges). See _build_signals.

        Returns (content, signals_injected): the flag is True when a Signals
        block was appended — a real flag rather than text-sniffing, so literal
        "Signals (" text in the email itself cannot fake a count.

        `body_mode`/`body_head_lines` override the configured llm_body_mode for a
        single call (used by escalation's second pass, which forces "head").
        """
        mode = body_mode if body_mode is not None else self.config.llm_body_mode
        head_lines = (
            body_head_lines if body_head_lines is not None else self.config.llm_body_head_lines
        )

        header_block = format_classification_headers(email.headers)
        subject = strip_reply_prefix(email.subject)
        email_content = f"Subject: {subject}\nFrom: {email.sender}\n{header_block}"

        signals_block = self._build_signals(email)
        if signals_block:
            email_content += f"\n\n{signals_block}"

        body = ""
        if mode == "head":
            clean_content = self.email_processor.strip_html(email.content)
            body = "\n".join(clean_content.splitlines()[:head_lines])
        elif mode == "full":
            body = self._smart_truncate(self.email_processor.strip_html(email.content))

        if body:
            email_content += f"\n\n{body}"
        return email_content, bool(signals_block)

    def _is_ruled_or_personal_sender(self, email: EmailRecord) -> bool:
        """Whether the user has already made a judgment about this sender: a
        personal/freemail domain match, or a sender_rules entry for its exact
        address or its registered domain.

        Shared by the escalation and signals gates — the two copies of this
        check once drifted apart, so there is exactly one now.
        """
        domain = extract_domain(email.sender)
        return (
            domain in self.config.personal_domains
            or domain in self.config.sender_rules
            or extract_address(email.sender) in self.config.sender_rules
        )

    def _should_escalate(self, email: EmailRecord) -> bool:
        """Whether a header-only `main` verdict warrants a body-head second pass.

        Only for an unruled company-domain sender: sender-rule shortcut hits never
        reach the LLM, and freemail/personal senders are legitimately personal, so
        neither is escalated. Requires a body with actual text to escalate.
        """
        if not self.config.escalation.enabled:
            return False
        # Header-only mode only: in head/full the first pass already saw MORE
        # body than the 10-line escalation pass, so a second pass adds no
        # information and can only flip verdicts on less context.
        if self.config.llm_body_mode != "none":
            return False
        domain = extract_domain(email.sender)
        if not domain or self._is_ruled_or_personal_sender(email):
            return False
        # Gate on the text the LLM would see, not the raw bytes: a body that is
        # only HTML markup (or whitespace) strips to nothing, and the escalated
        # pass would repeat the identical header-only input at temperature 0.
        if not self.email_processor.strip_html(email.content).strip():
            return False
        return True

    def _build_signals(self, email: EmailRecord) -> str:
        """Deterministic advisory signals for the LLM, or "" when not applicable.

        Only emails that would otherwise reach the LLM as unruled company-domain
        mail get signals: sender-rule shortcut hits never reach here, and freemail/
        personal senders are excluded (their mail is legitimately personal). This
        keeps the block focused on the header-poor cold-mail case it was built for.

        A sender the user has *already made a judgment about* is excluded too — an
        address or domain with any sender_rules entry, not just a personal_domains
        match. The signals exist to help the LLM on senders nobody has ruled on;
        injecting a block for a ruled sender (e.g. repucci.org: main, an
        individual's own vanity domain) would contradict the prompt's statement
        that the block only appears for unruled senders.
        """
        if self._is_ruled_or_personal_sender(email):
            return ""
        signals = compute_sender_signals(
            sender=email.sender,
            subject=email.subject,
            headers=email.headers,
        )
        return format_signals_block(signals)

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

    # Base confidence per category, keyed by the PRODUCTION category names
    # (transaction/newsletter/marketing/cold-outreach/main) — coupled to the
    # configured category naming, like the escalation gate's "main" trigger:
    # renaming a category means re-keying this table. Personal correspondence
    # ("main") and transactional mail ("transaction") are the high-stakes tiers —
    # missing either is costly — while the bulk categories sit at the neutral
    # base. A configured category absent from the table (e.g. a dev config on
    # the legacy capitalized defaults) gets the same neutral base.
    _BASE_CONFIDENCE = {
        "main": 0.9,
        "transaction": 0.9,
        "newsletter": 0.7,
        "marketing": 0.7,
        "cold-outreach": 0.7,
    }

    def _calculate_confidence(self, category: str, explanation: str) -> float:
        """Confidence score: a per-category base tier, nudged up by explanation length."""
        base_confidence = self._BASE_CONFIDENCE.get(category, 0.7)

        # Longer explanations read as more considered verdicts (bump capped below)
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
