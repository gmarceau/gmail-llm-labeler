"""Tests for PipelineConfig YAML round-tripping."""

import tempfile

from email_labeler.pipeline.config import PipelineConfig, TransformConfig


class TestTransformConfigRoundTrip:
    def test_domain_rules_round_trips(self, tmp_path):
        config = PipelineConfig(
            transform=TransformConfig(domain_rules={"substack.com": "newsletter"})
        )
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.domain_rules == {"substack.com": "newsletter"}

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

    def test_defaults_round_trip(self, tmp_path):
        """A freshly generated config keeps the "none" body-mode default with no domain rules."""
        config = PipelineConfig()
        path = str(tmp_path / "config.yaml")
        config.to_yaml(path)

        loaded = PipelineConfig.from_yaml(path)

        assert loaded.transform.domain_rules == {}
        assert loaded.transform.llm_body_mode == "none"
        assert loaded.transform.llm_body_head_lines == 20
