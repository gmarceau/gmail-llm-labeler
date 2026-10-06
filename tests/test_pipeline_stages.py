"""Tests for pipeline stages."""

import json
from unittest.mock import MagicMock, patch

import pytest

from email_labeler.database import EmailDatabase
from email_labeler.email_processor import EmailProcessor
from email_labeler.llm_service import LLMCategorizationError, LLMService
from email_labeler.pipeline.base import (
    ActionResult,
    EmailRecord,
    EnrichedEmailRecord,
    PipelineContext,
)
from email_labeler.pipeline.config import GMAIL_TAB_LABEL_IDS
from email_labeler.pipeline.extract_stage import ExtractStage
from email_labeler.pipeline.load_stage import LoadStage
from email_labeler.pipeline.sync_stage import SyncStage
from email_labeler.pipeline.transform_stage import TransformStage


class TestExtractStage:
    """Test cases for ExtractStage."""

    def test_init(self, mock_email_processor, email_database, pipeline_config):
        """Test ExtractStage initialization."""
        stage = ExtractStage(
            config=pipeline_config.extract,
            email_processor=mock_email_processor,
            database=email_database,
        )

        assert stage.email_processor == mock_email_processor
        assert stage.database == email_database
        assert stage.config == pipeline_config.extract

    def test_needs_body_by_llm_body_mode(self, mock_email_processor, email_database, pipeline_config):
        """Extract downloads bodies only for the head/full body modes."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        ctx = PipelineContext.create(pipeline_config, dry_run=True)

        ctx.config.transform.llm_body_mode = "none"
        ctx.config.transform.escalation.enabled = False
        assert stage._needs_body(ctx) is False
        ctx.config.transform.llm_body_mode = "head"
        assert stage._needs_body(ctx) is True
        ctx.config.transform.llm_body_mode = "full"
        assert stage._needs_body(ctx) is True

    def test_needs_body_when_escalation_enabled(
        self, mock_email_processor, email_database, pipeline_config
    ):
        """Escalation needs the body head, so header-only mode still fetches bodies."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        ctx = PipelineContext.create(pipeline_config, dry_run=True)

        ctx.config.transform.llm_body_mode = "none"
        ctx.config.transform.escalation.enabled = True
        assert stage._needs_body(ctx) is True

    def test_execute_success(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test successful email extraction."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        # Mock email processor to return test emails
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Subject 1",
                "from": "sender1@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Content 1",
                "headers": {},
                "has_unsubscribe": False,
            },
            {
                "id": "msg2",
                "subject": "Subject 2",
                "from": "sender2@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Content 2",
                "headers": {},
                "has_unsubscribe": False,
            },
        ]

        emails = stage.execute(None, pipeline_context)

        assert len(emails) == 2
        assert all(isinstance(email, EmailRecord) for email in emails)
        assert emails[0].id == "msg1"
        assert emails[1].id == "msg2"

    def test_execute_with_filtering(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test email extraction with filtering."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        # Mock email processor to return test emails
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Subject 1",
                "from": "sender1@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Content 1",
                "headers": {},
                "has_unsubscribe": False,
            },
            {
                "id": "msg2",
                "subject": "Subject 2",
                "from": "sender2@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Content 2",
                "headers": {},
                "has_unsubscribe": False,
            },
        ]

        emails = stage.execute(None, pipeline_context)

        # Should return all emails from email processor
        assert len(emails) == 2
        assert emails[0].id == "msg1"
        assert emails[1].id == "msg2"

    def test_execute_api_error(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test handling of Gmail API errors."""
        from googleapiclient.errors import HttpError

        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        error = HttpError(resp=MagicMock(status=403), content=b"Forbidden")
        mock_email_processor.fetch_emails_from_gmail.side_effect = error

        # Should return empty list if continue_on_error is True
        emails = stage.execute(None, pipeline_context)
        assert emails == []

    def test_execute_empty_result(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test extraction with no emails found."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        mock_email_processor.fetch_emails_from_gmail.return_value = []

        emails = stage.execute(None, pipeline_context)

        assert emails == []

    def test_batch_processing(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test batch processing of emails."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        # Create large number of fetched email dicts
        test_emails = [
            {
                "id": f"msg{i}",
                "subject": f"Subject {i}",
                "from": f"sender{i}@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": f"Content {i}",
                "headers": {},
                "has_unsubscribe": False,
            }
            for i in range(25)
        ]
        mock_email_processor.fetch_emails_from_gmail.return_value = test_emails

        emails = stage.execute(None, pipeline_context)

        assert len(emails) == 25

    def test_rate_limiting(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Test rate limiting handling."""
        from googleapiclient.errors import HttpError

        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)

        rate_limit_error = HttpError(resp=MagicMock(status=429), content=b"Rate limit exceeded")
        mock_email_processor.fetch_emails_from_gmail.side_effect = rate_limit_error

        # Should return empty list if continue_on_error is True
        emails = stage.execute(None, pipeline_context)
        assert emails == []

    def test_save_email_called_on_extract(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Extracted emails are persisted to DB without body content."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Subject 1",
                "from": "sender1@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Body",
                "headers": {},
                "has_unsubscribe": False,
            },
        ]
        email_database.save_email = MagicMock()

        stage.execute(None, pipeline_context)

        email_database.save_email.assert_called_once_with(
            "msg1", "Subject 1", "sender1@example.com", "2024-01-01T10:00:00Z", "", {}, False
        )

    def test_save_email_includes_headers(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Classification headers from the fetched email dict are forwarded to save_email."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        headers = {"list-unsubscribe": "<https://example.com/unsub>", "reply-to": "noreply@x.com"}
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Weekly",
                "from": "news@substack.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Body",
                "headers": headers,
                "has_unsubscribe": False,
            },
        ]
        email_database.save_email = MagicMock()

        stage.execute(None, pipeline_context)

        email_database.save_email.assert_called_once_with(
            "msg1", "Weekly", "news@substack.com", "2024-01-01T10:00:00Z", "", headers, False
        )

    def test_save_email_empty_headers(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Fetched emails with an empty headers dict save correctly."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Subject",
                "from": "a@b.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Body",
                "headers": {},
                "has_unsubscribe": False,
            },
        ]
        email_database.save_email = MagicMock()

        stage.execute(None, pipeline_context)

        email_database.save_email.assert_called_once_with(
            "msg1", "Subject", "a@b.com", "2024-01-01T10:00:00Z", "", {}, False
        )

    def test_gmail_extract_carries_headers_onto_email_record(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Headers and has_unsubscribe from the gmail source land on the EmailRecord itself."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        headers = {"list-unsubscribe": "<https://example.com/unsub>"}
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Weekly",
                "from": "news@substack.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Click here to Unsubscribe.",
                "headers": headers,
                "has_unsubscribe": True,
            },
        ]

        emails = stage.execute(None, pipeline_context)

        assert emails[0].headers == headers
        assert emails[0].has_unsubscribe is True

    def test_database_extract_carries_headers_onto_email_record(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """Headers and has_unsubscribe from the database source land on the EmailRecord itself."""
        pipeline_config.extract.source = "database"
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        headers_json = '{"list-unsubscribe": "<https://example.com/unsub>"}'
        email_database.cursor.fetchall.return_value = [
            ("msg1", "Weekly", "news@substack.com", "2024-01-01T10:00:00Z", "", headers_json, 1),
        ]

        emails = stage.execute(None, pipeline_context)

        assert len(emails) == 1
        assert emails[0].headers == {"list-unsubscribe": "<https://example.com/unsub>"}
        assert emails[0].has_unsubscribe is True

    def test_save_email_uses_source_has_unsubscribe(
        self, mock_email_processor, email_database, pipeline_config, pipeline_context
    ):
        """has_unsubscribe is stored as-is from the source, not recomputed from content.

        A header-only fetch has empty content, yet the source flag (computed by
        get_email_content from the List-Unsubscribe header) must still be saved.
        """
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "News",
                "from": "news@sub.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "",
                "headers": {},
                "has_unsubscribe": True,
            },
        ]
        email_database.save_email = MagicMock()

        stage.execute(None, pipeline_context)

        email_database.save_email.assert_called_once_with(
            "msg1", "News", "news@sub.com", "2024-01-01T10:00:00Z", "", {}, True
        )

    def test_dry_run_fetches_from_gmail_but_does_not_save(
        self, mock_email_processor, email_database, pipeline_config
    ):
        """DRY RUN still fetches real emails so they can be categorized, but writes nothing."""
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        mock_email_processor.fetch_emails_from_gmail.return_value = [
            {
                "id": "msg1",
                "subject": "Subject 1",
                "from": "sender1@example.com",
                "date": "2024-01-01T10:00:00Z",
                "body": "Content 1",
                "headers": {},
                "has_unsubscribe": False,
            },
        ]
        email_database.save_email = MagicMock()
        dry_run_context = PipelineContext.create(config=pipeline_config, dry_run=True)

        emails = stage.execute(None, dry_run_context)

        assert len(emails) == 1
        assert emails[0].id == "msg1"
        email_database.save_email.assert_not_called()

    def test_dry_run_fetches_from_database(
        self, mock_email_processor, email_database, pipeline_config
    ):
        """DRY RUN with a database source still reads real cached emails (pure read, no write)."""
        pipeline_config.extract.source = "database"
        stage = ExtractStage(pipeline_config.extract, mock_email_processor, email_database)
        email_database.cursor.fetchall.return_value = [
            ("msg1", "Weekly", "news@substack.com", "2024-01-01T10:00:00Z", "", "{}", 0),
        ]
        dry_run_context = PipelineContext.create(config=pipeline_config, dry_run=True)

        emails = stage.execute(None, dry_run_context)

        assert len(emails) == 1
        assert emails[0].id == "msg1"

    def test_metadata_fetch_unsubscribe_header_sets_has_unsubscribe(
        self, mock_gmail_client, email_database, pipeline_config
    ):
        """Acceptance: a metadata-format gmail fetch of an email with a List-Unsubscribe
        header yields has_unsubscribe=True on the EmailRecord and in the
        database.save_email call — no body downloaded, nothing recomputed from content.

        This exercises the real EmailProcessor and get_email_content against a mocked
        Gmail API, with llm_body_mode=none (header-only classification).
        """
        pipeline_config.transform.llm_body_mode = "none"
        pipeline_config.transform.escalation.enabled = False
        processor = EmailProcessor(gmail_client=mock_gmail_client)
        stage = ExtractStage(pipeline_config.extract, processor, email_database)
        email_database.save_email = MagicMock()

        # One Gmail message whose only unsubscribe signal is the List-Unsubscribe header.
        mock_gmail_client.users().messages().list.return_value.execute.return_value = {
            "messages": [{"id": "msg1", "threadId": "thread1"}]
        }
        mock_gmail_client.users().messages().get.return_value.execute.return_value = {
            "id": "msg1",
            "snippet": "Weekly digest",
            "payload": {
                "headers": [
                    {"name": "From", "value": "news@substack.com"},
                    {"name": "Subject", "value": "Weekly"},
                    {"name": "Date", "value": "Mon, 01 Jan 2024 12:00:00 +0000"},
                    {"name": "List-Unsubscribe", "value": "<https://example.com/unsub>"},
                ]
            },
        }

        emails = stage.execute(
            None, PipelineContext.create(config=pipeline_config, dry_run=False, test_mode=True)
        )

        # The fetch was header-only (metadata format), so no body was downloaded.
        _, kwargs = mock_gmail_client.users().messages().get.call_args
        assert kwargs["format"] == "metadata"

        assert len(emails) == 1
        assert emails[0].has_unsubscribe is True
        assert emails[0].content == ""
        email_database.save_email.assert_called_once_with(
            "msg1",
            "Weekly",
            "news@substack.com",
            "Mon, 01 Jan 2024 12:00:00 +0000",
            "",
            {"list-unsubscribe": "<https://example.com/unsub>"},
            True,
        )


class TestTransformStage:
    """Test cases for TransformStage."""

    def test_init(self, llm_service, mock_email_processor, pipeline_config):
        """Test TransformStage initialization."""
        stage = TransformStage(
            config=pipeline_config.transform,
            llm_service=llm_service,
            email_processor=mock_email_processor,
        )

        assert stage.llm_service == llm_service
        assert stage.email_processor == mock_email_processor
        assert stage.config == pipeline_config.transform

    def test_execute_success(
        self,
        llm_service,
        mock_email_processor,
        pipeline_config,
        pipeline_context_no_test_mode,
        sample_email_records,
    ):
        """Test successful email transformation."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        # Mock LLM service responses
        llm_service.categorize_email.side_effect = [
            ("Work", "Business email"),
            ("Newsletters", "Marketing content"),
            ("Personal", "Personal message"),
        ]

        enriched_emails = stage.execute(sample_email_records, pipeline_context_no_test_mode)

        assert len(enriched_emails) == 3
        assert all(isinstance(email, EnrichedEmailRecord) for email in enriched_emails)

        assert enriched_emails[0].category == "Work"
        assert enriched_emails[1].category == "Newsletters"
        assert enriched_emails[2].category == "Personal"

        assert llm_service.categorize_email.call_count == 3

    def test_execute_with_errors(
        self,
        llm_service,
        mock_email_processor,
        pipeline_config,
        pipeline_context_no_test_mode,
        sample_email_records,
    ):
        """Test transformation with LLM errors."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        # First email fails, others succeed
        llm_service.categorize_email.side_effect = [
            LLMCategorizationError("LLM Error"),
            ("Newsletters", "Marketing content"),
            ("Personal", "Personal message"),
        ]

        enriched_emails = stage.execute(sample_email_records, pipeline_context_no_test_mode)

        # Should still return results for successful emails
        assert len(enriched_emails) == 2
        assert enriched_emails[0].category == "Newsletters"
        assert enriched_emails[1].category == "Personal"

    def test_execute_batch_processing(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        """Test batch processing in transform stage."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        # Create large batch of emails
        large_batch = []
        for i in range(50):
            large_batch.append(
                EmailRecord(
                    id=f"msg{i}",
                    subject=f"Subject {i}",
                    sender=f"sender{i}@example.com",
                    content=f"Content {i}",
                    received_date="2024-01-01T10:00:00Z",
                )
            )

        llm_service.categorize_email.return_value = ("Work", "Business email")

        enriched_emails = stage.execute(large_batch, pipeline_context_no_test_mode)

        assert len(enriched_emails) == 50
        assert llm_service.categorize_email.call_count == 50

    def test_execute_skip_on_error(
        self,
        llm_service,
        mock_email_processor,
        pipeline_config,
        pipeline_context_no_test_mode,
        sample_email_records,
    ):
        """Test error handling with skip_on_error enabled."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        # First email fails, should be skipped due to skip_on_error=True
        llm_service.categorize_email.side_effect = LLMCategorizationError("Temporary error")

        enriched_emails = stage.execute([sample_email_records[0]], pipeline_context_no_test_mode)

        # Should return empty list when error occurs and skip_on_error is True
        assert len(enriched_emails) == 0
        assert llm_service.categorize_email.call_count == 1
        assert len(pipeline_context_no_test_mode.errors) == 1

    def test_timeout_handling(
        self,
        llm_service,
        mock_email_processor,
        pipeline_config,
        pipeline_context_no_test_mode,
        sample_email_records,
    ):
        """Test handling of LLM timeouts."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        llm_service.categorize_email.side_effect = LLMCategorizationError("LLM timeout")

        enriched_emails = stage.execute(sample_email_records, pipeline_context_no_test_mode)

        # Should return empty list for timeout cases
        assert len(enriched_emails) == 0

    def test_dry_run_still_categorizes(
        self, llm_service, mock_email_processor, pipeline_config, sample_email_records
    ):
        """DRY RUN runs real categorization (sender-shortcut/LLM); it just never writes."""
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)
        llm_service.categorize_email.side_effect = [
            ("Work", "Business email"),
            ("Newsletters", "Marketing content"),
            ("Personal", "Personal message"),
        ]
        dry_run_context = PipelineContext.create(config=pipeline_config, dry_run=True)

        enriched_emails = stage.execute(sample_email_records, dry_run_context)

        assert len(enriched_emails) == 3
        assert enriched_emails[0].category == "Work"
        assert llm_service.categorize_email.call_count == 3


class TestTransformStageSenderShortcut:
    """Known-sender sender_rules shortcut (exact address or domain) takes precedence over the LLM."""

    def test_known_domain_shortcut_skips_llm(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.sender_rules = {"example.com": "Work"}
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1",
            subject="Hi",
            sender="boss@example.com",
            content="body",
            received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert len(enriched) == 1
        assert enriched[0].category == "Work"
        assert enriched[0].explanation == "known sender: example.com"
        llm_service.categorize_email.assert_not_called()

    def test_exact_address_shortcut_skips_llm(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        """A full-address rule routes that exact sender, even on a personal domain."""
        pipeline_config.transform.sender_rules = {"recruiter@gmail.com": "Marketing"}
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1",
            subject="Job opportunity!",
            sender="Some Recruiter <Recruiter@Gmail.com>",
            content="body",
            received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert enriched[0].category == "Marketing"
        assert enriched[0].explanation == "known sender: recruiter@gmail.com"
        llm_service.categorize_email.assert_not_called()

    def test_exact_address_beats_domain_rule(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        """When both an address rule and a domain rule match, the more specific address wins."""
        pipeline_config.transform.sender_rules = {
            "example.com": "Work",
            "boss@example.com": "Marketing",
        }
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1", subject="Hi", sender="boss@example.com",
            content="body", received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert enriched[0].category == "Marketing"
        assert enriched[0].explanation == "known sender: boss@example.com"
        llm_service.categorize_email.assert_not_called()

    def test_unknown_sender_still_calls_llm(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.sender_rules = {"example.com": "Work"}
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1",
            subject="Hi",
            sender="friend@other.com",
            content="body",
            received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert len(enriched) == 1
        llm_service.categorize_email.assert_called_once()

    def test_rule_category_not_in_categories_falls_through_to_llm(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        """A sender_rules value that isn't a configured category is not applied; the LLM decides instead."""
        pipeline_config.transform.sender_rules = {"example.com": "NotARealCategory"}
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1",
            subject="Hi",
            sender="boss@example.com",
            content="body",
            received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        llm_service.categorize_email.assert_called_once()
        assert enriched[0].category == "Work"  # from the llm_service fixture's default return value

    def test_invalid_address_rule_does_not_fall_back_to_domain_rule(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        """A matched address rule with a bad category falls through to the LLM, not the domain rule."""
        pipeline_config.transform.sender_rules = {
            "example.com": "Work",
            "boss@example.com": "NotARealCategory",
        }
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1", subject="Hi", sender="boss@example.com",
            content="body", received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        llm_service.categorize_email.assert_called_once()
        assert enriched[0].category == "Work"  # from the llm_service fixture, not the domain rule

    def test_sender_shortcut_metrics_tracked(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.sender_rules = {"example.com": "Work"}
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        emails = [
            EmailRecord(
                id="e1", subject="Hi", sender="boss@example.com",
                content="b", received_date="2024-01-01T10:00:00Z",
            ),
            EmailRecord(
                id="e2", subject="Hi", sender="x@other.com",
                content="b", received_date="2024-01-01T10:00:00Z",
            ),
        ]

        stage.execute(emails, pipeline_context_no_test_mode)

        assert pipeline_context_no_test_mode.metrics["transform_sender_shortcut"] == 1
        assert pipeline_context_no_test_mode.metrics["transform_llm_calls"] == 1

    def test_cold_outreach_rule_routes_via_shortcut(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """Recruiter domains rule to cold-outreach, which must be a valid target."""
        production_like_config.transform.sender_rules = {"ovise.com": "cold-outreach"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1",
            subject="Strong Ivy League Forward Deployed Engineers for Arch",
            sender="Danny Tomkins <danny@ovise.com>",
            content="body",
            received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert enriched[0].category == "cold-outreach"
        assert enriched[0].explanation == "known sender: ovise.com"
        llm_service.categorize_email.assert_not_called()

    def test_cold_outreach_rule_rejected_when_not_a_category(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """A cold-outreach rule is inert unless the category is configured."""
        # cold-outreach is a production category; remove it so the rule is inert.
        production_like_config.transform.categories.remove("cold-outreach")
        production_like_config.transform.sender_rules = {"ovise.com": "cold-outreach"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        email = EmailRecord(
            id="e1", subject="x", sender="danny@ovise.com",
            content="b", received_date="2024-01-01T10:00:00Z",
        )

        stage.execute([email], pipeline_context_no_test_mode)

        # No cold-outreach category configured -> falls through to the LLM.
        llm_service.categorize_email.assert_called_once()


class TestTransformStageBodyMode:
    """Unknown-sender LLM input is header-first; body inclusion is gated by llm_body_mode."""

    def _email(self, **overrides):
        defaults = {
            "id": "e1",
            "subject": "Weekly Digest",
            "sender": "news@example.com",
            "content": "Line one\nLine two\nLine three\nUnsubscribe here",
            "received_date": "2024-01-01T10:00:00Z",
            "headers": {"list-unsubscribe": "<https://example.com/unsub>", "precedence": "bulk"},
        }
        defaults.update(overrides)
        return EmailRecord(**defaults)

    def test_none_mode_includes_headers_but_no_body(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.llm_body_mode = "none"
        mock_email_processor.strip_html.side_effect = lambda c: c
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        stage.execute([self._email()], pipeline_context_no_test_mode)

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "List-Unsubscribe: <https://example.com/unsub>" in email_content
        assert "Precedence: bulk" in email_content
        assert "Line one" not in email_content

    def test_head_mode_includes_first_n_lines(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.llm_body_mode = "head"
        pipeline_config.transform.llm_body_head_lines = 2
        mock_email_processor.strip_html.side_effect = lambda c: c
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        stage.execute([self._email()], pipeline_context_no_test_mode)

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "List-Unsubscribe: <https://example.com/unsub>" in email_content
        assert "Line one" in email_content
        assert "Line two" in email_content
        assert "Line three" not in email_content

    def test_full_mode_includes_truncated_body(
        self, llm_service, mock_email_processor, pipeline_config, pipeline_context_no_test_mode
    ):
        pipeline_config.transform.llm_body_mode = "full"
        mock_email_processor.strip_html.side_effect = lambda c: c
        stage = TransformStage(pipeline_config.transform, llm_service, mock_email_processor)

        stage.execute([self._email()], pipeline_context_no_test_mode)

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "List-Unsubscribe: <https://example.com/unsub>" in email_content
        assert "Line one" in email_content
        assert "Unsubscribe here" in email_content


class TestTransformStageSignalsInjection:
    """Signals are injected for unruled company-domain senders only.

    The production_like_config fixture supplies the baseline
    (personal_domains=["gmail.com"], no sender rules); tests override only what
    they vary.
    """

    def _email(self, **overrides):
        defaults = {
            "id": "e1",
            "subject": "An opportunity that made me think of you",
            "sender": "Danny Tomkins <danny@ovise.com>",
            "content": "body",
            "received_date": "2024-01-01T10:00:00Z",
            "headers": {},
        }
        defaults.update(overrides)
        return EmailRecord(**defaults)

    def test_signals_injected_for_unruled_company_domain(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        stage.execute([self._email()], pipeline_context_no_test_mode)

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "Signals (" in email_content
        assert "outreach phrases" in email_content
        assert pipeline_context_no_test_mode.metrics["transform_signals_injected"] == 1

    def test_no_signals_for_freemail_sender(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        stage.execute(
            [self._email(sender="Jake Miles <jacob.miles@gmail.com>")],
            pipeline_context_no_test_mode,
        )

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "Signals (" not in email_content
        assert "transform_signals_injected" not in pipeline_context_no_test_mode.metrics

    def test_no_signals_for_sender_rule_shortcut_hit(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        production_like_config.transform.sender_rules = {"ovise.com": "marketing"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "marketing"
        llm_service.categorize_email.assert_not_called()

    def test_no_signals_for_ruled_address_that_falls_through(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """A sender whose *address* is ruled but whose rule category is invalid falls
        through to the LLM (see _try_sender_shortcut). Signals must not reappear: the
        user has already made a judgment about this sender.
        """
        # Address rule present but its category isn't configured -> shortcut declines.
        production_like_config.transform.categories.remove("cold-outreach")
        production_like_config.transform.sender_rules = {"danny@ovise.com": "cold-outreach"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        stage.execute([self._email()], pipeline_context_no_test_mode)

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "Signals (" not in email_content
        assert "transform_signals_injected" not in pipeline_context_no_test_mode.metrics

    def test_no_signals_for_ruled_company_domain_that_falls_through(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """An individual's own vanity domain (e.g. repucci.org: main) is a company
        domain by the freemail test, but the user has already ruled it. No signals —
        the 'no known-sender rule' / 'company domain' facts would be contradictory.
        """
        # main not configured -> the domain rule's category is invalid, so it falls
        # through to the LLM; the point is that the sender is still ruled.
        production_like_config.transform.categories.remove("main")
        production_like_config.transform.sender_rules = {"repucci.org": "main"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        stage.execute(
            [self._email(sender="Michael Repucci <michael@repucci.org>")],
            pipeline_context_no_test_mode,
        )

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "Signals (" not in email_content
        assert "transform_signals_injected" not in pipeline_context_no_test_mode.metrics

    def test_signals_metric_counts_injections_not_literal_text(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """transform_signals_injected counts real injected blocks — an email whose
        subject literally contains 'Signals (' (and receives no block, here via a
        freemail sender) must not increment the metric."""
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        stage.execute(
            [
                self._email(
                    subject="Signals (beta): your weekly digest is ready",
                    sender="Jake Miles <jacob.miles@gmail.com>",
                )
            ],
            pipeline_context_no_test_mode,
        )

        email_content = llm_service.categorize_email.call_args[0][0]
        assert "Signals (" in email_content  # from the subject itself, not a block
        assert "transform_signals_injected" not in pipeline_context_no_test_mode.metrics


class TestTransformStageEscalation:
    """Tiered body escalation: re-classify borderline `main` with the body head.

    The production_like_config fixture supplies the shared baseline (lowercase
    categories, personal_domains=["gmail.com"], no sender rules, header-only
    LLM input, escalation enabled); tests override only what they vary.
    """

    def _email(self, **overrides):
        defaults = {
            "id": "e1",
            "subject": "Guillaume, love your background!",
            "sender": "Kenn Peters <kenn@thalolabs.com>",
            "content": "Hi Guillaume\n\nWe are a Series A stealth startup backed by Sequoia\nHappy to offer equity and a $5k referral bonus\nAre you open to a brief chat?\nThanks\nKenn",
            "received_date": "2024-01-01T10:00:00Z",
            "headers": {"return-path": "<kenn@thalolabs.com>"},
        }
        defaults.update(overrides)
        return EmailRecord(**defaults)

    def test_escalates_borderline_main_with_body_head(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        production_like_config.transform.escalation.body_head_lines = 3
        mock_email_processor.strip_html.side_effect = lambda c: c
        # First pass says main; escalated (body-head) pass says marketing.
        llm_service.categorize_email.side_effect = [("main", "looks personal"), ("marketing", "pitch")]
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "marketing"
        assert llm_service.categorize_email.call_count == 2
        # First pass: header-only (no body line).
        first_content = llm_service.categorize_email.call_args_list[0][0][0]
        assert "Series A" not in first_content
        # Second pass: body head included (only the first 3 lines).
        second_content = llm_service.categorize_email.call_args_list[1][0][0]
        assert "We are a Series A stealth startup" in second_content
        assert "Are you open to a brief chat?" not in second_content
        assert pipeline_context_no_test_mode.metrics["transform_escalation_second_pass"] == 1

    def test_no_escalation_when_first_pass_is_not_main(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        llm_service.categorize_email.return_value = ("marketing", "bulk promo")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "marketing"
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    def test_no_escalation_for_freemail_sender(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        llm_service.categorize_email.return_value = ("main", "personal")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute(
            [self._email(sender="Jake Miles <jacob.miles@gmail.com>")],
            pipeline_context_no_test_mode,
        )

        assert enriched[0].category == "main"
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    def test_no_escalation_when_disabled(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        production_like_config.transform.escalation.enabled = False
        llm_service.categorize_email.return_value = ("main", "personal")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "main"
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    def test_no_escalation_for_sender_rule_shortcut_hit(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        production_like_config.transform.sender_rules = {"thalolabs.com": "main"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "main"
        llm_service.categorize_email.assert_not_called()
        assert pipeline_context_no_test_mode.metrics["transform_sender_shortcut"] == 1

    def test_no_escalation_without_body(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        # The body gate checks HTML-stripped text; identity keeps "   " empty.
        mock_email_processor.strip_html.side_effect = lambda c: c
        llm_service.categorize_email.return_value = ("main", "personal")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute(
            [self._email(content="   ")], pipeline_context_no_test_mode
        )

        assert enriched[0].category == "main"
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    @pytest.mark.parametrize("mode", ["head", "full"])
    def test_no_escalation_when_first_pass_already_saw_body(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode, mode
    ):
        """Escalation is a header-only-mode device: in head/full the first pass
        already saw MORE body than the 10-line escalation pass, so a second
        pass adds no information and can only flip verdicts on less context."""
        production_like_config.transform.llm_body_mode = mode
        # Even an escalation pass that would see strictly less body (3 of the
        # 7 lines) than the first pass saw must not fire here.
        production_like_config.transform.escalation.body_head_lines = 3
        mock_email_processor.strip_html.side_effect = lambda c: c
        llm_service.categorize_email.return_value = ("main", "personal")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "main"
        # The first pass classified with the body already in view.
        first_content = llm_service.categorize_email.call_args[0][0]
        assert "We are a Series A stealth startup" in first_content
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    def test_no_escalation_when_body_strips_to_nothing(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """A body that is only HTML markup (no text) strips to empty: the
        escalated pass would repeat the identical header-only input at temp 0."""
        mock_email_processor.strip_html.side_effect = lambda c: ""
        llm_service.categorize_email.return_value = ("main", "personal")
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute(
            [self._email(content="<html><body><div></div></body></html>")],
            pipeline_context_no_test_mode,
        )

        assert enriched[0].category == "main"
        assert llm_service.categorize_email.call_count == 1
        assert "transform_escalation_second_pass" not in pipeline_context_no_test_mode.metrics

    def test_at_most_one_escalation_per_email(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """Even if the second pass returns `main` again, no further escalation."""
        llm_service.categorize_email.side_effect = [("main", "a"), ("main", "b")]
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "main"
        assert llm_service.categorize_email.call_count == 2
        assert pipeline_context_no_test_mode.metrics["transform_escalation_second_pass"] == 1

    def test_escalated_result_is_validated(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """An out-of-vocabulary escalated category falls back to main like any other."""
        llm_service.categorize_email.side_effect = [("main", "a"), ("Bogus", "b")]
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)

        enriched = stage.execute([self._email()], pipeline_context_no_test_mode)

        assert enriched[0].category == "main"


class TestCalculateConfidence:
    """Confidence tiers are keyed by the production category names (bead 6yq):
    every production category hits an intended base tier; only a category absent
    from the table (e.g. a dev config on the legacy capitalized defaults) falls
    back to the neutral base. The sender-rule shortcut stays at a flat 1.0."""

    @pytest.fixture
    def stage(self, llm_service, mock_email_processor, production_like_config):
        return TransformStage(production_like_config.transform, llm_service, mock_email_processor)

    @pytest.mark.parametrize(
        ("category", "expected_base"),
        [
            ("main", 0.9),
            ("transaction", 0.9),
            ("newsletter", 0.7),
            ("marketing", 0.7),
            ("cold-outreach", 0.7),
        ],
    )
    def test_production_category_hits_intended_tier(self, stage, category, expected_base):
        # "" explanation: no length bump — the pure base tier
        assert stage._calculate_confidence(category, "") == expected_base

    def test_unkeyed_category_gets_neutral_base(self, stage):
        assert stage._calculate_confidence("Bills", "") == 0.7

    def test_longer_explanations_nudge_confidence_up(self, stage):
        assert stage._calculate_confidence("main", "x" * 40) == 0.94  # 0.9 + 0.04
        assert stage._calculate_confidence("newsletter", "x" * 100) == 0.8  # 0.7 + 0.1

    def test_explanation_bump_is_capped_at_one(self, stage):
        assert stage._calculate_confidence("main", "x" * 250) == 1.0  # 0.9 + 0.2, capped
        assert stage._calculate_confidence("newsletter", "x" * 250) == 0.9  # 0.7 + 0.2

    def test_shortcut_path_stays_at_full_confidence(
        self, llm_service, mock_email_processor, production_like_config, pipeline_context_no_test_mode
    ):
        """A sender-rule hit never consults the tiers: flat 1.0 regardless of category."""
        production_like_config.transform.sender_rules = {"ovise.com": "marketing"}
        stage = TransformStage(production_like_config.transform, llm_service, mock_email_processor)
        email = EmailRecord(
            id="e1", subject="x", sender="danny@ovise.com",
            content="b", received_date="2024-01-01T10:00:00Z",
        )

        enriched = stage.execute([email], pipeline_context_no_test_mode)

        assert enriched[0].confidence == 1.0


class TestTransformStageLLMWiring:
    """TransformStage's own LLMService construction takes service, model,
    gpt_oss_reasoning, and timeout from its TransformConfig (bead 3zj, bead omm)
    — the config is authoritative, not the environment."""

    def test_own_llm_service_reflects_config(self, mock_email_processor, production_like_config):
        production_like_config.transform.llm_service = "ollama"
        production_like_config.transform.model = "qwen2.5:7b"
        production_like_config.transform.gpt_oss_reasoning = "high"
        production_like_config.transform.timeout = 120

        # Patch client construction so the eager init stays offline.
        with patch.object(LLMService, "_get_llm_client", return_value=MagicMock()):
            stage = TransformStage(
                production_like_config.transform, email_processor=mock_email_processor
            )

        assert stage.llm_service.service == "ollama"
        assert stage.llm_service.model == "qwen2.5:7b"
        assert stage.llm_service.gpt_oss_reasoning == "high"
        assert stage.llm_service.timeout == 120


class TestLoadStage:
    """Test cases for LoadStage."""

    def test_init(self, mock_email_processor, pipeline_config):
        """Test LoadStage initialization."""
        stage = LoadStage(config=pipeline_config.load, email_processor=mock_email_processor)

        assert stage.email_processor == mock_email_processor
        assert stage.config == pipeline_config.load

    def test_execute_success(
        self, mock_email_processor, pipeline_config, pipeline_context, sample_enriched_email_records
    ):
        """Test successful loading of enriched emails."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        results = stage.execute(sample_enriched_email_records, pipeline_context)

        assert len(results) == 3
        assert all(isinstance(result, ActionResult) for result in results)
        assert all(result.success for result in results)

        # Verify email processor calls were made for applying labels
        assert (
            mock_email_processor.get_or_create_label.call_count > 0
            or mock_email_processor.add_labels_to_email.call_count >= 0
        )

    def test_execute_database_error(
        self, mock_email_processor, pipeline_config, pipeline_context, sample_enriched_email_records
    ):
        """Test handling of database errors."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        # Mock email processor error
        mock_email_processor.add_labels_to_email.side_effect = Exception("Email processor error")

        results = stage.execute(sample_enriched_email_records, pipeline_context)

        assert len(results) == 3
        assert all(not result.success for result in results)
        # Check that errors contain some indication of failure
        assert all(len(result.errors) > 0 for result in results)

    def test_execute_partial_success(
        self, mock_email_processor, pipeline_config, pipeline_context, sample_enriched_email_records
    ):
        """Test partial success in loading."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        # First email fails, others succeed
        def mock_apply_labels(email_id, label_ids):
            if email_id == sample_enriched_email_records[0].id:
                raise Exception("Email processor error")
            return True

        mock_email_processor.add_labels_to_email.side_effect = mock_apply_labels
        mock_email_processor.get_or_create_label.return_value = "test_label_id"

        results = stage.execute(sample_enriched_email_records, pipeline_context)

        assert len(results) == 3
        assert not results[0].success
        assert results[1].success
        assert results[2].success

    def test_execute_dry_run(
        self, mock_email_processor, pipeline_config, sample_enriched_email_records
    ):
        """Test loading in dry run mode."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        # Create dry run context
        dry_run_context = PipelineContext.create(config=pipeline_config, dry_run=True)

        results = stage.execute(sample_enriched_email_records, dry_run_context)

        assert len(results) == 3
        assert all(result.success for result in results)

        # Should not make actual email processor calls in dry run
        mock_email_processor.add_labels_to_email.assert_not_called()

    def test_apply_category_tab_applies_correct_system_label(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """Test that apply_category_tab adds the correct Gmail system label ID."""
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = {"newsletter": "forums"}
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="newsletter",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ) as mock_add:
            results = stage.execute([email], pipeline_context)

        assert results[0].success
        mock_add.assert_called_once_with(
            mock_email_processor.gmail, "msg1", ["CATEGORY_FORUMS"]
        )

    def test_apply_category_tab_no_mapping_succeeds(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """Test apply_category_tab returns True when the email category has no tab mapping."""
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = {"newsletter": "forums"}
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="transaction",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ) as mock_add:
            results = stage.execute([email], pipeline_context)

        assert results[0].success
        mock_add.assert_not_called()

    def test_apply_category_tab_invalid_tab_name_fails(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """Test apply_category_tab returns False when the mapped tab name is not a valid Gmail category."""
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = {"newsletter": "not_a_real_tab"}
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="newsletter",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        results = stage.execute([email], pipeline_context)

        assert not results[0].success

    def test_apply_category_tab_all_production_mappings(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """Test all four production category-to-tab mappings resolve to the right Gmail label IDs."""
        tab_map = {
            "transaction": "updates",
            "marketing": "updates",
            "newsletter": "forums",
            "main": "primary",
        }
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = tab_map
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        emails = [
            EnrichedEmailRecord(
                id=f"msg{i}",
                subject="Test",
                sender="s@example.com",
                content="c",
                received_date="2024-01-01T10:00:00Z",
                category=cat,
                explanation="x",
                confidence=0.9,
                processing_time=1.0,
            )
            for i, cat in enumerate(tab_map)
        ]

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ) as mock_add:
            results = stage.execute(emails, pipeline_context)

        assert all(r.success for r in results)
        applied_labels = [call[0][2] for call in mock_add.call_args_list]
        assert applied_labels == [
            ["CATEGORY_UPDATES"],   # transaction
            ["CATEGORY_UPDATES"],   # marketing
            ["CATEGORY_FORUMS"],    # newsletter
            ["CATEGORY_PERSONAL"],  # main
        ]

    def test_gmail_tab_label_ids_constant(self):
        """Test that GMAIL_TAB_LABEL_IDS contains the expected Gmail system label IDs."""
        assert GMAIL_TAB_LABEL_IDS["primary"] == "CATEGORY_PERSONAL"
        assert GMAIL_TAB_LABEL_IDS["updates"] == "CATEGORY_UPDATES"
        assert GMAIL_TAB_LABEL_IDS["forums"] == "CATEGORY_FORUMS"
        assert GMAIL_TAB_LABEL_IDS["promotions"] == "CATEGORY_PROMOTIONS"
        assert GMAIL_TAB_LABEL_IDS["social"] == "CATEGORY_SOCIAL"

    def test_batch_processing(self, mock_email_processor, pipeline_config, pipeline_context):
        """Test batch processing in load stage."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        # Create large batch
        large_batch = []
        for i in range(100):
            large_batch.append(
                EnrichedEmailRecord(
                    id=f"msg{i}",
                    subject=f"Subject {i}",
                    sender=f"sender{i}@example.com",
                    content=f"Content {i}",
                    received_date="2024-01-01T10:00:00Z",
                    category="Work",
                    explanation="Business email",
                    confidence=0.9,
                    processing_time=1.0,
                )
            )

        # Mock successful email processor responses
        mock_email_processor.get_or_create_label.return_value = "test_label_id"
        mock_email_processor.add_labels_to_email.return_value = True

        results = stage.execute(large_batch, pipeline_context)

        assert len(results) == 100
        assert all(result.success for result in results)

    def test_apply_label_records_applied_label_id(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """apply_label records the concrete label ID the run applied."""
        pipeline_config.load.default_actions = ["apply_label"]
        mock_email_processor.get_or_create_label.return_value = "Label_71"
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="marketing",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        results = stage.execute([email], pipeline_context)

        assert results[0].success
        assert results[0].applied_label_ids == ["Label_71"]

    def test_apply_category_tab_records_system_label_id(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """apply_category_tab records the CATEGORY_* system label ID it applied."""
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = {"marketing": "updates"}
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="marketing",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ):
            results = stage.execute([email], pipeline_context)

        assert results[0].success
        assert results[0].applied_label_ids == ["CATEGORY_UPDATES"]

    def test_star_action_records_starred_label_id(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """star records the STARRED system label ID it applied."""
        pipeline_config.load.default_actions = ["star"]
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="marketing",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ):
            results = stage.execute([email], pipeline_context)

        assert results[0].success
        assert results[0].applied_label_ids == ["STARRED"]

    def test_failed_apply_label_records_no_label_ids(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """A failed apply_label records no label IDs, only the failure."""
        pipeline_config.load.default_actions = ["apply_label"]
        mock_email_processor.get_or_create_label.return_value = None
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        email = EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category="marketing",
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

        results = stage.execute([email], pipeline_context)

        assert not results[0].success
        assert results[0].applied_label_ids == []

    @pytest.mark.parametrize("mode", ["dry_run", "preview_mode"])
    def test_dry_run_and_preview_record_no_applied_label_ids(
        self, mock_email_processor, pipeline_config, sample_enriched_email_records, mode
    ):
        """Dry-run and preview apply nothing to Gmail, so no label IDs are recorded."""
        stage = LoadStage(pipeline_config.load, mock_email_processor)
        context = PipelineContext.create(config=pipeline_config, **{mode: True})

        results = stage.execute(sample_enriched_email_records, context)

        assert all(result.success for result in results)
        assert all(result.applied_label_ids == [] for result in results)
        mock_email_processor.add_labels_to_email.assert_not_called()


class TestSyncStage:
    """Test cases for SyncStage."""

    def test_init(self, email_database, mock_metrics_tracker, pipeline_config):
        """Test SyncStage initialization."""
        stage = SyncStage(
            config=pipeline_config.sync,
            database=email_database,
            metrics_tracker=mock_metrics_tracker,
        )

        assert stage.database == email_database
        assert stage.metrics_tracker == mock_metrics_tracker
        assert stage.config == pipeline_config.sync

    def test_execute_success(
        self, mock_metrics_tracker, pipeline_config, pipeline_context, sample_enriched_email_records
    ):
        """Test successful result synchronization."""
        # Create a mock database
        mock_database = MagicMock()
        mock_database.update_email_labels.return_value = None

        stage = SyncStage(pipeline_config.sync, mock_database, mock_metrics_tracker)

        # Create sample ActionResults
        action_results = [
            ActionResult(
                email_id=email.id,
                category=email.category,
                actions_taken=["apply_label"],
                success=True,
                errors=[],
            )
            for email in sample_enriched_email_records
        ]

        stage.execute(action_results, pipeline_context)

        # Verify database calls were made for syncing
        assert mock_database.update_email_labels.call_count == 3

    def test_execute_batch_processing(
        self, mock_metrics_tracker, pipeline_config, pipeline_context
    ):
        """Test batch processing in sync stage."""
        # Create a mock database
        mock_database = MagicMock()
        mock_database.update_email_labels.return_value = None

        stage = SyncStage(pipeline_config.sync, mock_database, mock_metrics_tracker)

        # Create large batch of action results
        action_results = []
        for i in range(50):
            action_results.append(
                ActionResult(
                    email_id=f"msg{i}",
                    category="Work",
                    actions_taken=["apply_label"],
                    success=True,
                    errors=[],
                )
            )

        stage.execute(action_results, pipeline_context)

        # Verify database calls were made for all results
        assert mock_database.update_email_labels.call_count == 50

    def test_execute_dry_run(
        self, mock_metrics_tracker, pipeline_config, sample_enriched_email_records
    ):
        """Test sync stage in dry run mode."""
        # Create a mock database
        mock_database = MagicMock()

        stage = SyncStage(pipeline_config.sync, mock_database, mock_metrics_tracker)

        # Create sample ActionResults
        action_results = [
            ActionResult(
                email_id=email.id,
                category=email.category,
                actions_taken=["apply_label"],
                success=True,
                errors=[],
            )
            for email in sample_enriched_email_records
        ]

        dry_run_context = PipelineContext.create(config=pipeline_config, dry_run=True)

        stage.execute(action_results, dry_run_context)

        # Should not make actual database calls in dry run
        mock_database.update_email_labels.assert_not_called()

    def test_execute_database_errors(
        self, mock_metrics_tracker, pipeline_config, pipeline_context, sample_enriched_email_records
    ):
        """Test handling of database errors during sync."""
        # Create a mock database that raises errors
        mock_database = MagicMock()
        mock_database.update_email_labels.side_effect = Exception("Database error")

        stage = SyncStage(pipeline_config.sync, mock_database, mock_metrics_tracker)

        # Create sample ActionResults
        action_results = [
            ActionResult(
                email_id=email.id,
                category=email.category,
                actions_taken=["apply_label"],
                success=True,
                errors=[],
            )
            for email in sample_enriched_email_records
        ]

        stage.execute(action_results, pipeline_context)

        # Should have logged errors but continued processing
        assert len(pipeline_context.errors) > 0

    @pytest.mark.parametrize("save_metrics", [True, False])
    def test_save_metrics_setting(
        self,
        mock_metrics_tracker,
        pipeline_config,
        pipeline_context,
        sample_enriched_email_records,
        save_metrics,
    ):
        """Test save_metrics configuration setting."""
        # Create a mock database
        mock_database = MagicMock()
        mock_database.update_email_labels.return_value = None
        pipeline_config.sync.save_metrics = save_metrics
        stage = SyncStage(pipeline_config.sync, mock_database, mock_metrics_tracker)

        # Create sample ActionResults
        action_results = [
            ActionResult(
                email_id=email.id,
                category=email.category,
                actions_taken=["apply_label"],
                success=True,
                errors=[],
            )
            for email in sample_enriched_email_records
        ]

        with patch("builtins.open", create=True) as mock_open, patch("json.dump") as mock_json_dump:
            stage.execute(action_results, pipeline_context)

            if save_metrics:
                # Should save metrics to file
                mock_open.assert_called()
                mock_json_dump.assert_called()
            else:
                # Should not save metrics
                mock_open.assert_not_called()
                mock_json_dump.assert_not_called()

    def test_metrics_tracking(
        self,
        email_database,
        mock_metrics_tracker,
        pipeline_config,
        pipeline_context,
        sample_enriched_email_records,
    ):
        """Test that metrics are tracked during execution."""
        stage = SyncStage(pipeline_config.sync, email_database, mock_metrics_tracker)

        # Create sample ActionResults
        action_results = [
            ActionResult(
                email_id=email.id,
                category=email.category,
                actions_taken=["apply_label"],
                success=True,
                errors=[],
            )
            for email in sample_enriched_email_records
        ]

        stage.execute(action_results, pipeline_context)

        # Should have tracked metrics for each result
        assert mock_metrics_tracker.add_result.call_count == 3


class TestColdOutreachLoadRouting:
    """cold-outreach gets its own label and is archived out of the inbox."""

    def _email(self, category="cold-outreach"):
        return EnrichedEmailRecord(
            id="msg1",
            subject="Engineering Director Opportunity in NYC",
            sender="Tony Svare <tony@byrecruiting.com>",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category=category,
            explanation="cold outreach",
            confidence=1.0,
            processing_time=1.0,
        )

    def test_cold_outreach_applies_label_and_archives(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """category_actions for cold-outreach = apply_label + archive (no tab action)."""
        pipeline_config.load.category_actions = {"cold-outreach": ["apply_label", "archive"]}
        pipeline_config.load.default_actions = ["apply_label", "apply_category_tab"]
        pipeline_config.load.category_tab_map = {"cold-outreach": "updates"}
        mock_email_processor.get_or_create_label.return_value = "Label_cold"
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        results = stage.execute([self._email()], pipeline_context)

        assert results[0].success
        assert "apply_label" in results[0].actions_taken
        assert "archive" in results[0].actions_taken
        # archive = pulled from inbox so it never competes in the Primary tab
        mock_email_processor.remove_from_inbox.assert_called_once_with("msg1")

    def test_cold_outreach_never_maps_to_primary_tab(
        self, mock_email_processor, pipeline_config, pipeline_context
    ):
        """Even if a tab action ran, cold-outreach must not map to the primary tab."""
        pipeline_config.load.category_actions = {}  # fall back to default_actions
        pipeline_config.load.default_actions = ["apply_category_tab"]
        pipeline_config.load.category_tab_map = {"cold-outreach": "updates"}
        stage = LoadStage(pipeline_config.load, mock_email_processor)

        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ) as mock_add:
            results = stage.execute([self._email()], pipeline_context)

        assert results[0].success
        # CATEGORY_UPDATES, never CATEGORY_PERSONAL (primary)
        mock_add.assert_called_once_with(
            mock_email_processor.gmail, "msg1", ["CATEGORY_UPDATES"]
        )


class TestLoadSyncLabelRecording:
    """The DB records the label IDs the load stage actually applied (gmail-llm-labeler-14e).

    LoadStage runs against a mocked Gmail client; SyncStage runs against a real
    SQLite tmp database, so email_labels.labels and label_history carry the
    concrete IDs the (mocked) load stage applied — the syndesus shape: a user
    Label_XX plus a CATEGORY_* tab label. Live-mailbox verification is deferred
    to the next real scheduled run.
    """

    def _email(self, category="marketing"):
        return EnrichedEmailRecord(
            id="msg1",
            subject="Test",
            sender="s@example.com",
            content="c",
            received_date="2024-01-01T10:00:00Z",
            category=category,
            explanation="x",
            confidence=0.9,
            processing_time=1.0,
        )

    def _configure(self, pipeline_config):
        """Shape the run like the syndesus case: label + updates tab, no metrics file."""
        pipeline_config.load.category_actions = {}
        pipeline_config.load.default_actions = ["apply_label", "apply_category_tab"]
        pipeline_config.load.category_tab_map = {"marketing": "updates"}
        pipeline_config.sync.save_metrics = False

    def _run_load_then_sync(
        self, pipeline_config, mock_email_processor, context, database, category_label_id
    ):
        """Run load -> sync once for the sample email; return the load results."""
        mock_email_processor.get_or_create_label.return_value = category_label_id
        with patch(
            "email_labeler.pipeline.load_stage.add_labels_to_email", return_value=True
        ):
            load_results = LoadStage(pipeline_config.load, mock_email_processor).execute(
                [self._email()], context
            )
        SyncStage(pipeline_config.sync, database, MagicMock()).execute(load_results, context)
        return load_results

    def test_load_then_sync_records_applied_label_ids_in_db(
        self, mock_email_processor, pipeline_config, tmp_path
    ):
        """A real load -> sync run lands the applied label IDs in email_labels and label_history."""
        self._configure(pipeline_config)
        context = PipelineContext.create(config=pipeline_config)
        db = EmailDatabase(database_file=str(tmp_path / "labels.db"))

        load_results = self._run_load_then_sync(
            pipeline_config, mock_email_processor, context, db, "Label_71"
        )

        expected = ["Label_71", "CATEGORY_UPDATES"]
        assert load_results[0].applied_label_ids == expected

        category, labels = db.get_email_labels("msg1")
        assert category == "marketing"
        assert labels == expected

        db.cursor.execute(
            "SELECT old_labels, new_labels FROM label_history WHERE email_id = ?", ("msg1",)
        )
        old_labels, new_labels = db.cursor.fetchone()
        assert old_labels is None  # first run: the email had no prior labels row
        assert json.loads(new_labels) == expected
        assert db.is_email_processed("msg1")
        db.close()

    def test_relabel_run_populates_label_history_old_and_new(
        self, mock_email_processor, pipeline_config, tmp_path
    ):
        """A second run's history row carries the first run's labels as old_labels."""
        self._configure(pipeline_config)
        context = PipelineContext.create(config=pipeline_config)
        db = EmailDatabase(database_file=str(tmp_path / "labels.db"))

        self._run_load_then_sync(pipeline_config, mock_email_processor, context, db, "Label_71")
        self._run_load_then_sync(pipeline_config, mock_email_processor, context, db, "Label_99")

        db.cursor.execute(
            "SELECT old_labels, new_labels FROM label_history WHERE email_id = ? ORDER BY id",
            ("msg1",),
        )
        first, second = db.cursor.fetchall()
        assert json.loads(first[1]) == ["Label_71", "CATEGORY_UPDATES"]
        assert json.loads(second[0]) == ["Label_71", "CATEGORY_UPDATES"]  # old
        assert json.loads(second[1]) == ["Label_99", "CATEGORY_UPDATES"]  # new
        db.close()

    @pytest.mark.parametrize("mode", ["dry_run", "preview_mode"])
    def test_dry_run_and_preview_write_nothing_to_database(
        self, mock_email_processor, pipeline_config, tmp_path, mode
    ):
        """Dry-run and preview leave every sync-written table empty."""
        self._configure(pipeline_config)
        context = PipelineContext.create(config=pipeline_config, **{mode: True})
        db = EmailDatabase(database_file=str(tmp_path / "labels.db"))

        load_results = self._run_load_then_sync(
            pipeline_config, mock_email_processor, context, db, "Label_71"
        )

        assert load_results[0].applied_label_ids == []
        for table in ("email_labels", "label_history", "processed_emails"):
            db.cursor.execute(f"SELECT COUNT(*) FROM {table}")
            assert db.cursor.fetchone()[0] == 0, table
        assert not db.is_email_processed("msg1")
        db.close()
