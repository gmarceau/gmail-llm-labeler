"""Tests for PipelineConfig YAML round-tripping."""

import tempfile

from email_labeler.pipeline.config import PipelineConfig, TransformConfig


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
