"""Tests for gmail_utils functions."""

from email_labeler.gmail_utils import (
    compute_sender_signals,
    extract_address,
    extract_domain,
    format_classification_headers,
    format_signals_block,
    strip_reply_prefix,
)


class TestExtractAddress:
    def test_plain_address(self):
        assert extract_address("alice@example.com") == "alice@example.com"

    def test_display_name_angle_brackets(self):
        assert extract_address("Alice Smith <alice@example.com>") == "alice@example.com"

    def test_lowercased(self):
        assert extract_address("Recruiter@Gmail.COM") == "recruiter@gmail.com"

    def test_no_at_sign(self):
        assert extract_address("nodomain") == ""

    def test_empty_string(self):
        assert extract_address("") == ""


class TestStripReplyPrefix:
    def test_re_prefix(self):
        assert strip_reply_prefix("Re: Guillaume, Genius AI & Series D") == (
            "Guillaume, Genius AI & Series D"
        )

    def test_fwd_prefix(self):
        assert strip_reply_prefix("Fwd: Twin girls") == "Twin girls"

    def test_fw_prefix(self):
        assert strip_reply_prefix("Fw: notes") == "notes"

    def test_stacked_prefixes(self):
        assert strip_reply_prefix("Re: Fwd: Re: hello") == "hello"

    def test_case_insensitive(self):
        assert strip_reply_prefix("RE: hello") == "hello"
        assert strip_reply_prefix("fwd: hello") == "hello"

    def test_no_prefix_unchanged(self):
        assert strip_reply_prefix("Guillaume, Genius AI & Series D") == (
            "Guillaume, Genius AI & Series D"
        )

    def test_does_not_strip_midword(self):
        # "Repucci" starts with "Re" but is not a "Re:" marker.
        assert strip_reply_prefix("Repucci says hi") == "Repucci says hi"

    def test_empty(self):
        assert strip_reply_prefix("") == ""


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


class TestComputeSenderSignals:
    """Deterministic advisory signals injected for header-poor cold mail."""

    def _signals(self, **overrides):
        defaults = dict(
            sender="Danny Tomkins <danny@ovise.com>",
            subject="An opportunity that made me think of you",
            headers={},
        )
        defaults.update(overrides)
        return compute_sender_signals(**defaults)

    def test_domain_lexicon_hit(self):
        signals = self._signals(sender="A <recruiter@get-rockstar-hiring-ai.com>")
        hit = [s for s in signals if "recruiting/GTM words" in s]
        assert hit and "hiring" in hit[0]

    def test_gtm_domain_lexicon_hit(self):
        signals = self._signals(sender="A <a@unifygtm.com>")
        assert any("recruiting/GTM words" in s and "gtm" in s for s in signals)

    def test_bare_gtm_domain_not_flagged(self):
        # gtm.com alone is a mundane payroll vendor, not a GTM-agency domain; the
        # suffix-only rule keeps it out while unifygtm.com still hits.
        signals = self._signals(sender="A <a@gtm.com>")
        assert not any("recruiting/GTM words" in s for s in signals)

    def test_no_domain_lexicon_hit_for_ordinary_domain(self):
        signals = self._signals(sender="A <a@jupitered.com>")
        assert not any("recruiting/GTM words" in s for s in signals)

    def test_lexicon_hits_are_advisory_facts_not_verdicts(self):
        # 0fa regression guard: a lexicon hit is a fact string, never a category.
        # A legit school/building/utility mail must produce no outreach *verdict*,
        # only structural observations, so the LLM still judges it main.
        for domain in ("jupitered.com", "fsresidential.com", "ps20.org", "coned.com"):
            signals = self._signals(sender=f"A <a@{domain}>", subject="Burns Night")
            assert not any("recruiting/GTM words" in s for s in signals)
            assert not any("outreach phrases" in s for s in signals)

    def test_return_path_plus_tag_verp(self):
        signals = self._signals(
            headers={"return-path": "<chris+bounce@sterlingstrand.com>"}
        )
        hit = [s for s in signals if "plus-tag" in s]
        assert hit and "VERP" in hit[0]

    def test_return_path_srs0_rewrite(self):
        signals = self._signals(
            headers={"return-path": "<SRS0=abc=de=gmarceau.qc.ca=danny@ovise.com>"}
        )
        assert any("SRS0=" in s for s in signals)

    def test_return_path_srs1_rewrite(self):
        # SRS1= is the second-hop rewrite of an already-forwarded message.
        signals = self._signals(
            headers={"return-path": "<SRS1=abc=de==xyz=ovise.com=danny@mailer.example.com>"}
        )
        assert any("SRS1=" in s for s in signals)

    def test_null_return_path_flagged(self):
        # <> is the null bounce sender — a strong bulk tell (automated send that
        # expects bounces) that previously produced no signal at all.
        signals = self._signals(headers={"return-path": "<>"})
        assert any("null bounce sender" in s for s in signals)

    def test_return_path_domain_differs_from_from_domain(self):
        signals = self._signals(
            headers={"return-path": "<bounces@mailer.example.net>"}
        )
        assert any("differs from the From domain" in s for s in signals)

    def test_return_path_same_domain_not_flagged(self):
        signals = self._signals(
            sender="A <danny@ovise.com>",
            headers={"return-path": "<danny@ovise.com>"},
        )
        assert not any("differs from the From domain" in s for s in signals)

    def test_subject_lexicon_opportunity(self):
        assert any("outreach phrases" in s and "opportunit" in s for s in self._signals())

    def test_subject_lexicon_series_round(self):
        signals = self._signals(subject="Join our Series B stealth startup")
        hit = [s for s in signals if "outreach phrases" in s]
        assert hit and "series <funding round>" in hit[0] and "stealth" in hit[0]

    def test_subject_lexicon_angle_brackets(self):
        signals = self._signals(subject="Guillaume <> Arch")
        assert any("outreach phrases" in s and "<>" in s for s in signals)

    def test_reply_prefix_stripped_before_subject_match(self):
        signals = self._signals(subject="Re: Fwd: love your background!")
        assert any("love your background" in s for s in signals)

    def test_all_signals_are_strings(self):
        assert all(isinstance(s, str) for s in self._signals())


class TestFormatSignalsBlock:
    def test_empty_list_yields_empty_string(self):
        assert format_signals_block([]) == ""

    def test_renders_label_and_bullets(self):
        block = format_signals_block(["one fact", "two facts"])
        assert block.startswith("Signals (")
        assert "- one fact" in block
        assert "- two facts" in block

