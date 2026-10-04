"""Tests for PipelineConfig YAML round-tripping and load-time validation."""

import os

import pytest

from email_labeler.pipeline.config import (
    ConfigError,
    EscalationConfig,
    PipelineConfig,
    TransformConfig,
)


class TestTransformConfigRoundTrip:
    def test_sender_rules_round_trips(self, tmp_path):
        # Rule values must be configured categories — from_yaml rejects a value that
        # isn't in `categories` (the default set is capitalized). See TestSenderRuleValues.
        config = PipelineConfig(
            transform=TransformConfig(
                sender_rules={"substack.com": "Newsletters", "recruiter@gmail.com": "Marketing"}
            )
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.sender_rules == {
            "substack.com": "Newsletters",
            "recruiter@gmail.com": "Marketing",
        }

    def test_personal_domains_round_trips(self, tmp_path):
        config = PipelineConfig(
            transform=TransformConfig(personal_domains=["gmail.com", "hotmail.com"])
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.personal_domains == ["gmail.com", "hotmail.com"]

    def test_llm_body_mode_round_trips(self, tmp_path):
        config = PipelineConfig(transform=TransformConfig(llm_body_mode="head"))
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.llm_body_mode == "head"

    def test_llm_body_head_lines_round_trips(self, tmp_path):
        config = PipelineConfig(transform=TransformConfig(llm_body_head_lines=42))
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.llm_body_head_lines == 42

    def test_escalation_round_trips(self, tmp_path):
        config = PipelineConfig(
            transform=TransformConfig(escalation=EscalationConfig(enabled=True, body_head_lines=7))
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.escalation.enabled is True
        assert loaded.transform.escalation.body_head_lines == 7

    def test_escalation_defaults(self):
        assert TransformConfig().escalation.enabled is True
        assert TransformConfig().escalation.body_head_lines == 10

    def test_escalation_unknown_key_rejected(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    escalation:\n"
            "      enabled: true\n"
            "      body_head_line: 5\n"  # typo: missing 's'
        )

        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

    def test_sender_rules_file_is_loaded(self, tmp_path):
        """transform.sender_rules_file (relative to the config) supplies the rules."""
        # Rule values must be configured categories (default set is capitalized).
        (tmp_path / "rules.yaml").write_text(
            "sender_rules:\n"
            "  substack.com: Newsletters\n"
            "  recruiter@gmail.com: Marketing\n"
            "personal_domains:\n"
            "  - gmail.com\n"
        )
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    sender_rules_file: rules.yaml\n"
        )

        loaded = PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

        assert loaded.transform.sender_rules == {
            "substack.com": "Newsletters",
            "recruiter@gmail.com": "Marketing",
        }
        assert loaded.transform.personal_domains == ["gmail.com"]

    def test_to_yaml_writes_pointer_not_inline_rules(self, tmp_path):
        """When sender_rules_file is set, to_yaml emits the pointer, not inline rules."""
        config = PipelineConfig(
            transform=TransformConfig(sender_rules_file="rules.yaml")
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        text = (tmp_path / "config.yaml").read_text()
        assert "sender_rules_file: rules.yaml" in text
        assert "sender_rules:" not in text
        assert "personal_domains:" not in text

    def test_defaults_round_trip(self, tmp_path):
        """A freshly generated config keeps the "none" body-mode default with no sender rules."""
        config = PipelineConfig()
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.sender_rules == {}
        assert loaded.transform.personal_domains == []
        assert loaded.transform.llm_body_mode == "none"
        assert loaded.transform.llm_body_head_lines == 20


class TestConfigValidation:
    """Unknown/typo'd keys and wrong types fail fast at load time."""

    def _write(self, tmp_path, text):
        path = tmp_path / "config.yaml"
        path.write_text(text)
        return str(path)

    def test_typo_in_transform_key_raises(self, tmp_path):
        """A misspelled key (sender_rule vs sender_rules) is rejected, not swallowed."""
        path = self._write(
            tmp_path,
            "pipeline:\n  transform:\n    sender_rule:\n      a.com: marketing\n",
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "sender_rule" in str(exc.value)
        assert "config.yaml" in str(exc.value)

    def test_typo_in_extract_key_raises(self, tmp_path):
        path = self._write(
            tmp_path, "pipeline:\n  extract:\n    gmail_queryy: is:unread\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "gmail_queryy" in str(exc.value)

    def test_unknown_pipeline_key_raises(self, tmp_path):
        path = self._write(tmp_path, "pipeline:\n  extractt:\n    source: gmail\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "extractt" in str(exc.value)

    def test_unknown_top_level_key_raises(self, tmp_path):
        path = self._write(tmp_path, "pipelines:\n  x: 1\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "pipelines" in str(exc.value)

    def test_typo_in_paths_key_raises(self, tmp_path):
        """`paths:` is validated too, so a bad path key fails instead of being ignored."""
        path = self._write(tmp_path, "paths:\n  database_fil: /tmp/x.db\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "database_fil" in str(exc.value)

    def test_wrong_type_raises(self, tmp_path):
        path = self._write(tmp_path, "pipeline:\n  max_retries: not-a-number\n")

        with pytest.raises(ConfigError):
            PipelineConfig.from_yaml(path)

    def test_missing_file_raises_with_path(self, tmp_path):
        missing = str(tmp_path / "nope.yaml")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(missing)

        assert "nope.yaml" in str(exc.value)

    def test_missing_keys_fall_back_to_defaults(self, tmp_path):
        """Only unknown keys are fatal; omitted keys keep their sensible defaults."""
        path = self._write(tmp_path, "pipeline:\n  transform:\n    model: qwen2.5:7b\n")

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.model == "qwen2.5:7b"
        assert loaded.transform.llm_body_mode == "none"
        assert loaded.extract.source == "gmail"
        assert loaded.dry_run is False

    def test_paths_block_is_allowed(self, tmp_path):
        """The legitimate top-level `paths:` block does not trip the unknown-key check."""
        path = self._write(
            tmp_path, "paths:\n  database_file: /tmp/x.db\npipeline:\n  dry_run: true\n"
        )

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.dry_run is True

    def test_unknown_key_in_sender_rules_file_raises(self, tmp_path):
        (tmp_path / "rules.yaml").write_text("sender_rule:\n  a.com: marketing\n")
        path = self._write(
            tmp_path, "pipeline:\n  transform:\n    sender_rules_file: rules.yaml\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "sender_rule" in str(exc.value)

    def test_production_config_loads(self):
        """The shipped production config must keep loading (regression guard)."""
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        path = os.path.join(root, "config_production_7b.yaml")

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.llm_body_mode == "none"
        assert loaded.transform.sender_rules_file == "sender_rules_production_7b.yaml"
        assert len(loaded.transform.sender_rules) > 0


class TestSenderRulesFileTypes:
    """A wrongly-typed rules file fails at from_yaml, not silently at runtime."""

    def _config_with_rules_file(self, tmp_path, rules_text):
        (tmp_path / "rules.yaml").write_text(rules_text)
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n  transform:\n    sender_rules_file: rules.yaml\n"
        )
        return str(tmp_path / "config.yaml")

    def test_sender_rules_as_list_rejected(self, tmp_path):
        path = self._config_with_rules_file(
            tmp_path, "sender_rules:\n  - substack.com: Newsletters\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "rules.yaml" in str(exc.value)
        assert "sender_rules" in str(exc.value)

    def test_sender_rules_as_string_rejected(self, tmp_path):
        path = self._config_with_rules_file(tmp_path, "sender_rules: Newsletters\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "rules.yaml" in str(exc.value)
        assert "sender_rules" in str(exc.value)

    def test_personal_domains_as_dict_rejected(self, tmp_path):
        path = self._config_with_rules_file(tmp_path, "personal_domains:\n  gmail.com: true\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "rules.yaml" in str(exc.value)
        assert "personal_domains" in str(exc.value)

    def test_personal_domains_as_string_rejected(self, tmp_path):
        path = self._config_with_rules_file(tmp_path, "personal_domains: gmail.com\n")

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(path)

        assert "rules.yaml" in str(exc.value)
        assert "personal_domains" in str(exc.value)

    def test_empty_values_are_tolerated(self, tmp_path):
        """Empty/null rules blocks stay coerced to their empty defaults."""
        path = self._config_with_rules_file(tmp_path, "sender_rules: []\npersonal_domains:\n")

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.sender_rules == {}
        assert loaded.transform.personal_domains == []


class TestSenderRuleValues:
    """Every sender-rule value must be a configured category, case-sensitively.

    The runtime sender-rule lookups are case-sensitive, so a value like
    "Marketing" against a "marketing" category silently never matches; such a
    rule is fatal at load time instead.
    """

    def test_invalid_value_inline_raises(self, tmp_path):
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    sender_rules:\n"
            "      a.com: NotACategory\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

        message = str(exc.value)
        assert "a.com" in message
        assert "NotACategory" in message
        assert "config.yaml" in message

    def test_invalid_value_in_rules_file_raises(self, tmp_path):
        """An offender from the rules file names the rules file AND the config."""
        (tmp_path / "rules.yaml").write_text("sender_rules:\n  a.com: NotACategory\n")
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    categories: [marketing, main]\n"
            "    sender_rules_file: rules.yaml\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

        message = str(exc.value)
        assert "a.com" in message
        assert "NotACategory" in message
        assert "rules.yaml" in message
        assert "config.yaml" in message

    def test_case_mismatch_fails(self, tmp_path):
        """"Marketing" is not "marketing": the check is case-sensitive."""
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    categories: [marketing, main]\n"
            "    sender_rules:\n"
            "      a.com: Marketing\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

        assert "a.com: Marketing" in str(exc.value)

    def test_all_offending_rules_listed(self, tmp_path):
        """Every bad rule is listed, not just the first one."""
        (tmp_path / "config.yaml").write_text(
            "pipeline:\n"
            "  transform:\n"
            "    categories: [marketing]\n"
            "    sender_rules:\n"
            "      a.com: newsletter\n"
            "      b.com: NotACategory\n"
        )

        with pytest.raises(ConfigError) as exc:
            PipelineConfig.from_yaml(str(tmp_path / "config.yaml"))

        message = str(exc.value)
        assert "a.com: newsletter" in message
        assert "b.com: NotACategory" in message


class TestColdOutreachConfig:
    """cold-outreach category, prompt, and routing in the production config."""

    @pytest.fixture
    def prod_config(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return PipelineConfig.from_yaml(os.path.join(root, "config_production_7b.yaml"))

    def test_category_present(self, prod_config):
        assert "cold-outreach" in prod_config.transform.categories

    def test_prompt_mentions_all_five_categories(self, prod_config):
        prompt = prod_config.transform.user_prompt
        for cat in prod_config.transform.categories:
            assert cat in prompt
        # The CRITICAL RULES header must advertise five, not four.
        assert "5 categories" in prompt

    def test_prompt_category_definitions_are_consistently_indented(self, prod_config):
        """All `<category>:` definition lines share one indentation level.

        The prompt is a YAML literal block, so indentation is preserved verbatim
        into the model input; inconsistent indentation silently mangles it.
        """
        prompt = prod_config.transform.user_prompt
        cats = prod_config.transform.categories
        indents = {
            len(line) - len(line.lstrip())
            for line in prompt.splitlines()
            if line.lstrip().startswith(tuple(f"{c}:" for c in cats))
        }
        assert len(indents) == 1, f"category definitions indented inconsistently: {indents}"

    def test_prompt_has_no_whitespace_only_lines(self, prod_config):
        """No line should be non-empty whitespace (a sign of a botched block edit)."""
        bad = [i for i, line in enumerate(prod_config.transform.user_prompt.splitlines())
               if line != "" and line.strip() == ""]
        assert bad == [], f"whitespace-only prompt lines at: {bad}"

    def test_prompt_explains_signals_block(self, prod_config):
        """The prompt must describe the deterministic Signals block it may receive."""
        prompt = prod_config.transform.user_prompt
        assert "Signals:" in prompt
        assert "ADVISORY" in prompt

    def test_production_escalation_enabled(self, prod_config):
        """Production runs header-only, so escalation must be on to see cold-mail bodies."""
        assert prod_config.transform.escalation.enabled is True
        assert prod_config.transform.escalation.body_head_lines >= 1

    def test_routing_skips_primary_tab(self, prod_config):
        """cold-outreach gets its own label and is archived (not left in Primary)."""
        assert prod_config.load.category_tab_map.get("cold-outreach") != "primary"
        actions = prod_config.load.category_actions.get("cold-outreach", [])
        assert "apply_label" in actions
        assert "archive" in actions

    def test_extract_query_excludes_cold_outreach_label(self, prod_config):
        assert "-label:cold-outreach" in prod_config.extract.gmail_query

    def test_sender_rules_use_cold_outreach_and_are_valid(self, prod_config):
        """Recruiter domains route to cold-outreach; every rule value is a real category.

        The rules live in `sender_rules_production_7b.yaml`, which is git-ignored (it
        contains real addresses), so skip the value checks when it isn't present.
        """
        rules = prod_config.transform.sender_rules
        if not rules:
            pytest.skip("sender_rules_production_7b.yaml not present (git-ignored, local only)")
        cold = [k for k, v in rules.items() if v == "cold-outreach"]
        assert len(cold) > 0
        invalid = {k: v for k, v in rules.items() if v not in prod_config.transform.categories}
        assert invalid == {}
