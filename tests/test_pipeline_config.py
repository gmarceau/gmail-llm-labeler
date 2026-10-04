"""Tests for PipelineConfig YAML round-tripping and load-time validation."""

import os

import pytest

from email_labeler.pipeline.config import ConfigError, PipelineConfig, TransformConfig


class TestTransformConfigRoundTrip:
    def test_sender_rules_round_trips(self, tmp_path):
        config = PipelineConfig(
            transform=TransformConfig(
                sender_rules={"substack.com": "newsletter", "recruiter@gmail.com": "marketing"}
            )
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.sender_rules == {
            "substack.com": "newsletter",
            "recruiter@gmail.com": "marketing",
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

    def test_sender_rules_file_is_loaded(self, tmp_path):
        """transform.sender_rules_file (relative to the config) supplies the rules."""
        (tmp_path / "rules.yaml").write_text(
            "sender_rules:\n"
            "  substack.com: newsletter\n"
            "  recruiter@gmail.com: marketing\n"
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
            "substack.com": "newsletter",
            "recruiter@gmail.com": "marketing",
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
