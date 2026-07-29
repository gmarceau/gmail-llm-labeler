"""Tests for gmail_utils functions."""

from email_labeler.gmail_utils import (
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

