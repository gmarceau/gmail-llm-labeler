"""Tests for gmail_utils functions."""

from unittest.mock import MagicMock, call, patch

from email_labeler.gmail_utils import (
    CLASSIFICATION_METADATA_HEADERS,
    backfill_email_metadata,
    extract_domain,
    format_classification_headers,
)


class TestExtractDomain:
    def test_plain_address(self):
        assert extract_domain("alice@example.com") == "example.com"

    def test_display_name_angle_brackets(self):
        assert extract_domain("Alice Smith <alice@example.com>") == "example.com"

    def test_uppercase_normalized(self):
        assert extract_domain("News@Mail.Substack.COM") == "substack.com"

    def test_subdomain_stripped_to_registered_domain(self):
        assert extract_domain("news@mail.substack.com") == "substack.com"

    def test_no_at_sign(self):
        assert extract_domain("nodomain") == ""

    def test_empty_string(self):
        assert extract_domain("") == ""

    def test_trailing_bracket_without_space(self):
        # Malformed address (trailing > without <) is not valid RFC 2822; parseaddr returns ""
        assert extract_domain("no-reply@updates.example.com>") == ""


class TestFormatClassificationHeaders:
    def test_canonical_casing(self):
        result = format_classification_headers({"list-id": "<bulk.example.com>"})
        assert result == "List-Id: <bulk.example.com>"

    def test_fixed_order_regardless_of_input_order(self):
        headers = {"sender": "a@b.com", "list-unsubscribe": "<url>", "precedence": "bulk"}
        result = format_classification_headers(headers)
        lines = result.splitlines()
        assert lines == [
            "List-Unsubscribe: <url>",
            "Precedence: bulk",
            "Sender: a@b.com",
        ]

    def test_omits_absent_headers(self):
        result = format_classification_headers({"list-id": "<x>"})
        assert result == "List-Id: <x>"
        assert "Reply-To" not in result

    def test_empty_headers_yields_empty_string(self):
        assert format_classification_headers({}) == ""

    def test_omits_headers_with_empty_value(self):
        result = format_classification_headers({"list-id": "<x>", "sender": ""})
        assert result == "List-Id: <x>"


class TestBackfillEmailMetadata:
    def _make_processor(self):
        proc = MagicMock()
        proc.gmail = MagicMock()
        return proc

    def test_saves_fetched_metadata(self):
        proc = self._make_processor()
        mock_data = {
            "subject": "Hello",
            "from": "sender@remote.com",
            "date": "2024-01-01",
            "headers": {"list-unsubscribe": "<url>"},
            "has_unsubscribe": True,
        }
        with patch("email_labeler.gmail_utils.get_email_content", return_value=mock_data):
            mock_db = MagicMock()
            backfill_email_metadata(proc, ["e1"], mock_db)

        mock_db.save_email.assert_called_once_with(
            "e1", "Hello", "sender@remote.com", "2024-01-01", "",
            {"list-unsubscribe": "<url>"},
            True,
        )

    def test_calls_ensure_gmail_client_per_iteration(self):
        proc = self._make_processor()
        with patch("email_labeler.gmail_utils.get_email_content", return_value={}):
            backfill_email_metadata(proc, ["e1", "e2"], MagicMock())

        assert proc._ensure_gmail_client.call_count == 2

    def test_empty_list_is_noop(self):
        proc = self._make_processor()
        with patch("email_labeler.gmail_utils.get_email_content") as mock_get:
            mock_db = MagicMock()
            backfill_email_metadata(proc, [], mock_db)

        proc._ensure_gmail_client.assert_not_called()
        mock_get.assert_not_called()
        mock_db.save_email.assert_not_called()

    def test_failed_email_is_skipped_not_raised(self):
        proc = self._make_processor()
        with patch(
            "email_labeler.gmail_utils.get_email_content",
            side_effect=Exception("API error"),
        ):
            mock_db = MagicMock()
            backfill_email_metadata(proc, ["e1"], mock_db)

        mock_db.save_email.assert_not_called()

    def test_missing_fields_default_to_empty(self):
        proc = self._make_processor()
        with patch("email_labeler.gmail_utils.get_email_content", return_value={}):
            mock_db = MagicMock()
            backfill_email_metadata(proc, ["e1"], mock_db)

        mock_db.save_email.assert_called_once_with("e1", "", "", "", "", {}, False)

    def test_uses_classification_metadata_headers(self):
        proc = self._make_processor()
        with patch(
            "email_labeler.gmail_utils.get_email_content", return_value={}
        ) as mock_get:
            backfill_email_metadata(proc, ["e1"], MagicMock())

        _, kwargs = mock_get.call_args
        assert kwargs.get("metadata_headers") == CLASSIFICATION_METADATA_HEADERS
        assert kwargs.get("format") == "metadata"

    def test_second_email_fetched_with_same_gmail_client(self):
        """All iterations use the same processor.gmail resource."""
        proc = self._make_processor()
        results = [{"from": "a@x.com"}, {"from": "b@y.com"}]
        with patch("email_labeler.gmail_utils.get_email_content", side_effect=results):
            mock_db = MagicMock()
            backfill_email_metadata(proc, ["e1", "e2"], mock_db)

        assert mock_db.save_email.call_count == 2
