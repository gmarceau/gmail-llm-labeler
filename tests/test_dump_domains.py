"""Tests for the dump-domains CLI command."""

import json
import tempfile
from unittest.mock import MagicMock, patch

import pytest

from email_labeler.database import EmailDatabase
from email_labeler.pipeline.cli import _extract_domain, dump_domains


class TestExtractDomain:
    def test_plain_address(self):
        assert _extract_domain("alice@example.com") == "example.com"

    def test_display_name_angle_brackets(self):
        assert _extract_domain("Alice Smith <alice@example.com>") == "example.com"

    def test_uppercase_normalized(self):
        assert _extract_domain("News@Mail.Substack.COM") == "substack.com"

    def test_subdomain_stripped_to_registered_domain(self):
        assert _extract_domain("news@mail.substack.com") == "substack.com"

    def test_no_at_sign(self):
        assert _extract_domain("nodomain") == ""

    def test_empty_string(self):
        assert _extract_domain("") == ""

    def test_trailing_bracket_without_space(self):
        # Malformed address (trailing > without <) is not valid RFC 2822; parseaddr returns ""
        assert _extract_domain("no-reply@updates.example.com>") == ""


class TestDumpDomains:
    def _make_db_with_emails(self, tmp_path, emails):
        """Create a real DB populated with labeled emails.

        Each item is (email_id, subject, sender, headers) or
        (email_id, subject, sender, headers, has_unsubscribe).
        """
        db = EmailDatabase(database_file=str(tmp_path / "test.db"))
        for row in emails:
            email_id, subject, sender, headers = row[:4]
            has_unsubscribe = row[4] if len(row) > 4 else False
            db.save_email(email_id, subject, sender, "2024-01-01", "", headers, has_unsubscribe)
            db.update_email_labels(email_id, "unknown", [])
        return db

    def _args(self, config_path=None):
        return type("Args", (), {"config": config_path})()

    def test_groups_by_domain(self, tmp_path, capsys):
        db = self._make_db_with_emails(tmp_path, [
            ("e1", "Deal alert", "deals@amazon.com", {}),
            ("e2", "Order shipped", "shipping@amazon.com", {}),  # same domain as e1
            ("e3", "Weekly digest", "news@mail.substack.com", {"list-unsubscribe": "<url>"}),
        ])

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {}
            mock_pc.return_value.database_file = tmp_path / "test.db"

            result = dump_domains(self._args("fake.yaml"))

        assert result == 0
        output = json.loads(capsys.readouterr().out)
        domains = {e["domain"] for e in output}
        assert "amazon.com" in domains
        assert "substack.com" in domains  # subdomain stripped to registered domain
        # both amazon emails collapse into one domain entry
        amazon = next(e for e in output if e["domain"] == "amazon.com")
        assert amazon["count"] == 2

    def test_excludes_configured_domains(self, tmp_path, capsys):
        db = self._make_db_with_emails(tmp_path, [
            ("e1", "Weekly", "news@substack.com", {"list-unsubscribe": "<url>"}),
            ("e2", "Order", "ship@amazon.com", {}),
        ])

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {"substack.com": "newsletter"}
            mock_pc.return_value.database_file = tmp_path / "test.db"

            dump_domains(self._args("fake.yaml"))

        output = json.loads(capsys.readouterr().out)
        domains = {e["domain"] for e in output}
        assert "substack.com" not in domains
        assert "amazon.com" in domains

    def test_sorted_by_count_descending(self, tmp_path, capsys):
        emails = (
            [("e1", "Sub", "a@rare.com", {})] * 1
            + [("e2", "Sub", "b@common.com", {})] * 3
        )
        # unique ids
        emails = [(f"e{i}", "Sub", addr, h) for i, (_, _, addr, h) in enumerate(emails)]
        db = self._make_db_with_emails(tmp_path, emails)

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {}
            mock_pc.return_value.database_file = tmp_path / "test.db"

            dump_domains(self._args("fake.yaml"))

        output = json.loads(capsys.readouterr().out)
        counts = [e["count"] for e in output]
        assert counts == sorted(counts, reverse=True)

    def test_samples_include_all_items(self, tmp_path, capsys):
        emails = [(f"e{i}", f"Subject {i}", "news@big.com", {}) for i in range(10)]
        db = self._make_db_with_emails(tmp_path, emails)

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {}
            mock_pc.return_value.database_file = tmp_path / "test.db"

            dump_domains(self._args("fake.yaml"))

        output = json.loads(capsys.readouterr().out)
        big_entry = next(e for e in output if e["domain"] == "big.com")
        assert big_entry["count"] == 10
        assert len(big_entry["samples"]) == 10

    def test_headers_included_in_samples(self, tmp_path, capsys):
        headers = {"list-unsubscribe": "<https://unsub.example.com>", "precedence": "bulk"}
        db = self._make_db_with_emails(tmp_path, [
            ("e1", "Weekly", "news@letters.com", headers),
        ])

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {}
            mock_pc.return_value.database_file = tmp_path / "test.db"

            dump_domains(self._args("fake.yaml"))

        output = json.loads(capsys.readouterr().out)
        sample = output[0]["samples"][0]
        assert sample["headers"] == headers
        assert "has_unsubscribe" in sample

    def test_has_unsubscribe_true_when_header_present(self, tmp_path, capsys):
        """has_unsubscribe is True when list-unsubscribe header is stored."""
        db = self._make_db_with_emails(tmp_path, [
            ("e1", "Weekly", "news@letters.com", {"list-unsubscribe": "<url>"}, True),
        ])

        with patch("email_labeler.pipeline.cli.PipelineConfig.from_yaml") as mock_cfg, \
             patch("email_labeler.pipeline.cli.PathConfig") as mock_pc, \
             patch("email_labeler.pipeline.cli.EmailDatabase", return_value=db):
            mock_cfg.return_value.transform.domain_rules = {}
            mock_pc.return_value.database_file = tmp_path / "test.db"
            dump_domains(self._args("fake.yaml"))

        output = json.loads(capsys.readouterr().out)
        assert output[0]["samples"][0]["has_unsubscribe"] is True
