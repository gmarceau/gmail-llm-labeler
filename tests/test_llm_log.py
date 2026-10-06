"""Tests for the read-only LLM log analysis tool."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from datetime import datetime, timedelta
from pathlib import Path

from email_labeler import llm_log

BASE_TS = datetime(2026, 10, 4, 11, 0, 0).timestamp()


def _interaction(offset_seconds: float, **overrides) -> dict:
    """A log entry shaped like llm_service._log_interaction's output."""
    entry = {
        "request_timestamp": BASE_TS + offset_seconds,
        "response_timestamp": BASE_TS + offset_seconds + 4,
        "duration": 4.0,
        "model": "qwen2.5:7b",
        "service": "ollama",
        "response": '{"category": "main", "explanation": "x"}',
        "processed_email": "Subject: Hello\nFrom: a@b.com\n\nbody",
    }
    entry.update(overrides)
    return entry


def _write_log(path: Path, entries: list) -> None:
    """Mimic the append-only writers: back-to-back JSON objects."""
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, indent=2) + "\n")


class LogToolTestCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.dir = Path(self._tmp.name)
        self.log = self.dir / "llm_interactions.json"
        self.errors_log = self.dir / "categorization_errors.log"

    def run_cli(self, argv: list) -> str:
        out = io.StringIO()
        with redirect_stdout(out):
            code = llm_log.main(argv)
        self.assertEqual(code, 0)
        return out.getvalue()


class TestLoadRecords(LogToolTestCase):
    def test_parses_back_to_back_json_objects(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(60, response="", processed_email="Test email"),
                _interaction(120),
            ],
        )
        records = llm_log.load_records(str(self.log))
        self.assertEqual(len(records), 3)
        self.assertEqual(
            [r.time for r in records],
            [datetime(2026, 10, 4, 11, 0), datetime(2026, 10, 4, 11, 1),
             datetime(2026, 10, 4, 11, 2)],
        )

    def test_missing_file_exits_with_message(self):
        with self.assertRaises(SystemExit) as ctx:
            llm_log.load_records(str(self.dir / "nope.json"))
        self.assertIn("not found", str(ctx.exception))

    def test_malformed_json_reports_byte_offset(self):
        self.log.write_text('{"request_timestamp": 1.0, "res')
        with self.assertRaises(SystemExit) as ctx:
            llm_log.load_records(str(self.log))
        self.assertIn("malformed JSON at byte", str(ctx.exception))

    def test_non_object_record_exits(self):
        self.log.write_text('[1, 2, 3]\n')
        with self.assertRaises(SystemExit) as ctx:
            llm_log.load_records(str(self.log))
        self.assertIn("non-object record", str(ctx.exception))

    def test_record_without_timestamp_exits(self):
        _write_log(self.log, [{"model": "x"}])
        with self.assertRaises(SystemExit) as ctx:
            llm_log.load_records(str(self.log))
        self.assertIn("timestamp", str(ctx.exception))

    def test_iso_timestamp_error_records_load(self):
        self.errors_log.write_text(
            json.dumps(
                {
                    "timestamp": "2026-10-05T00:59:00.760630",
                    "model": "gpt-3.5-turbo",
                    "error": "API Error",
                    "email_preview": "Test email content",
                }
            )
            + "\n"
        )
        records = llm_log.load_records(str(self.errors_log))
        self.assertEqual(records[0].time.replace(microsecond=0), datetime(2026, 10, 5, 0, 59))


class TestRecordClassification(LogToolTestCase):
    def test_production_prompt_starts_with_subject_line(self):
        record = llm_log.Record(_interaction(0), datetime(2026, 10, 4))
        self.assertTrue(record.is_production)

    def test_bare_test_content_is_pollution(self):
        record = llm_log.Record(
            _interaction(0, processed_email="Test email"), datetime(2026, 10, 4)
        )
        self.assertFalse(record.is_production)

    def test_error_records_classified_by_preview(self):
        real = llm_log.Record(
            {
                "timestamp": "2026-10-05T00:00:00",
                "model": "qwen2.5:7b",
                "error": "boom",
                "email_preview": "Subject: Real\nFrom: a@b.com",
            },
            datetime(2026, 10, 5),
        )
        junk = llm_log.Record(
            {
                "timestamp": "2026-10-05T00:00:01",
                "model": "gpt-3.5-turbo",
                "error": "API Error",
                "email_preview": "Test email content",
            },
            datetime(2026, 10, 5),
        )
        self.assertTrue(real.is_production)
        self.assertFalse(junk.is_production)
        self.assertEqual(real.outcome, "boom")

    def test_outcome_variants(self):
        base = datetime(2026, 10, 4)

        def outcome_for(**overrides):
            return llm_log.Record(_interaction(0, **overrides), base).outcome

        self.assertEqual(outcome_for(), "main")
        self.assertEqual(outcome_for(response=""), "EMPTY")
        self.assertEqual(outcome_for(response=None), "EMPTY")
        self.assertEqual(outcome_for(response="not json"), "PARSE-FAIL")
        self.assertEqual(outcome_for(response='"just a string"'), "PARSE-FAIL")
        self.assertEqual(outcome_for(response='{"category": "cold-outreach"}'), "cold-outreach")
        self.assertEqual(outcome_for(response='{}'), "?")

    def test_backend_and_first_line(self):
        record = llm_log.Record(_interaction(0), datetime(2026, 10, 4))
        self.assertEqual(record.backend, "ollama/qwen2.5:7b")
        self.assertEqual(record.first_line, "Subject: Hello")
        error_record = llm_log.Record(
            {"model": "gpt-3.5-turbo"}, datetime(2026, 10, 4)
        )
        self.assertEqual(error_record.backend, "gpt-3.5-turbo")


class TestSplitRuns(unittest.TestCase):
    def _records_at(self, offsets_minutes: list) -> list:
        return [
            llm_log.Record(
                _interaction(m * 60), datetime(2026, 10, 4) + timedelta(minutes=m)
            )
            for m in offsets_minutes
        ]

    def test_splits_on_gap(self):
        runs = llm_log.split_runs(self._records_at([0, 1, 30, 31]), gap_minutes=15)
        self.assertEqual([len(r) for r in runs], [2, 2])

    def test_records_closer_than_gap_stay_together(self):
        runs = llm_log.split_runs(self._records_at([0, 14, 28]), gap_minutes=15)
        self.assertEqual(len(runs), 1)

    def test_empty_input(self):
        self.assertEqual(llm_log.split_runs([]), [])


class TestCli(LogToolTestCase):
    def test_days_counts_real_vs_test_with_outcomes(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
                _interaction(120, response='{"category": "marketing"}'),
                _interaction(180, response=""),
            ],
        )
        out = self.run_cli(["--log", str(self.log), "days"])
        self.assertIn("2026-10-04  real=3  test=1", out)
        self.assertIn("main=1, marketing=1, EMPTY=1", out)

    def test_days_empty_log_says_no_records(self):
        _write_log(self.log, [])
        out = self.run_cli(["--log", str(self.log), "days"])
        self.assertEqual(out, "no records\n")

    def test_days_errors_log(self):
        self.errors_log.write_text(
            json.dumps(
                {
                    "timestamp": "2026-10-05T00:59:00",
                    "model": "qwen2.5:7b",
                    "error": "boom",
                    "email_preview": "Subject: Real\nFrom: a@b.com",
                }
            )
            + "\n"
            + json.dumps(
                {
                    "timestamp": "2026-10-05T00:59:01",
                    "model": "gpt-3.5-turbo",
                    "error": "API Error",
                    "email_preview": "Test email",
                }
            )
            + "\n"
        )
        out = self.run_cli(["--errors-log", str(self.errors_log), "days", "--errors"])
        self.assertIn("2026-10-05  real=1  test=1", out)
        self.assertIn("boom=1", out)

    def test_runs_summarize_production_runs_and_exclude_test_records(self):
        # Two production clusters (a gap of 20 min); test records interleaved.
        _write_log(
            self.log,
            [
                _interaction(0, duration=10.0),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
                _interaction(120, response='{"category": "marketing"}'),
                _interaction(20 * 60 + 180),
            ],
        )
        out = self.run_cli(["--log", str(self.log), "runs"])
        self.assertIn("11:00:00 -> 11:02:00  2 calls", out)
        self.assertIn("11:23:00 -> 11:23:00  1 calls", out)
        self.assertIn("mean 7.0s", out)
        self.assertIn("main=1, marketing=1", out)
        self.assertNotIn("Test email", out)

    def test_runs_no_production_records(self):
        _write_log(
            self.log,
            [_interaction(0, processed_email="Test email", model="gpt-3.5-turbo")],
        )
        out = self.run_cli(["--log", str(self.log), "runs"])
        self.assertEqual(out, "no production records\n")

    def test_show_one_line_per_record(self):
        _write_log(
            self.log,
            [
                _interaction(0, duration=18.6),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
            ],
        )
        out = self.run_cli(["--log", str(self.log), "show"])
        lines = out.splitlines()
        self.assertEqual(len(lines), 2)
        self.assertIn("2026-10-04 11:00:00", lines[0])
        self.assertIn("ollama/qwen2.5:7b", lines[0])
        self.assertIn("18.6s", lines[0])
        self.assertIn("main", lines[0])
        self.assertIn("Subject: Hello", lines[0])
        self.assertIn("Test email", lines[1])

    def test_show_real_and_test_filters(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
            ],
        )
        real_out = self.run_cli(["--log", str(self.log), "show", "--real"])
        self.assertNotIn("Test email", real_out)
        self.assertIn("Subject: Hello", real_out)
        test_out = self.run_cli(["--log", str(self.log), "show", "--test"])
        self.assertIn("Test email", test_out)
        self.assertNotIn("Subject: Hello", test_out)

    def test_show_full_dumps_json_objects(self):
        _write_log(self.log, [_interaction(0)])
        out = self.run_cli(["--log", str(self.log), "show", "--full"])
        self.assertIn('"request_timestamp"', out)

    def test_since_filter(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(2 * 3600),  # two hours later
            ],
        )
        out = self.run_cli(
            ["--log", str(self.log), "show", "--since", "2026-10-04 12:00"]
        )
        self.assertNotIn("11:00:00", out)
        self.assertIn("13:00:00", out)

    def test_show_errors_log(self):
        self.errors_log.write_text(
            json.dumps(
                {
                    "timestamp": "2026-10-05T00:59:00",
                    "model": "qwen2.5:7b",
                    "error": "API Error",
                    "email_preview": "Subject: Real",
                }
            )
            + "\n"
        )
        out = self.run_cli(["--errors-log", str(self.errors_log), "show", "--errors"])
        self.assertIn("API Error", out)
        self.assertIn("Subject: Real", out)
        self.assertIn("qwen2.5:7b", out)  # error records have no service field


class TestDump(LogToolTestCase):
    def _dumped(self, argv: list) -> list:
        out = self.run_cli(argv)
        return [json.loads(line) for line in out.splitlines()]

    def test_bare_invocation_dumps_enriched_jsonl(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
            ],
        )
        records = self._dumped(["--log", str(self.log)])
        self.assertEqual(len(records), 2)
        rec = records[0]
        self.assertEqual(
            rec,
            {
                "time": "2026-10-04T11:00:00",
                "is_production": True,
                "service": "ollama",
                "model": "qwen2.5:7b",
                "duration": 4.0,
                "outcome": "main",
                "explanation": "x",
                "subject": "Hello",
                "sender": "a@b.com",
                "email": "Subject: Hello\nFrom: a@b.com\n\nbody",
            },
        )
        self.assertFalse(records[1]["is_production"])
        self.assertEqual(records[1]["subject"], "Test email")

    def test_dump_subcommand_honors_filters(self):
        _write_log(
            self.log,
            [
                _interaction(0),
                _interaction(60, processed_email="Test email", model="gpt-3.5-turbo"),
                _interaction(2 * 3600),
            ],
        )
        real = self._dumped(["--log", str(self.log), "dump", "--real"])
        self.assertEqual([r["subject"] for r in real], ["Hello", "Hello"])
        test = self._dumped(["--log", str(self.log), "dump", "--test"])
        self.assertEqual([r["subject"] for r in test], ["Test email"])
        since = self._dumped(["--log", str(self.log), "dump", "--since", "2026-10-04 12:00"])
        self.assertEqual(len(since), 1)

    def test_dump_empty_and_unparseable_responses(self):
        _write_log(
            self.log,
            [
                _interaction(0, response=""),
                _interaction(60, response="not json"),
            ],
        )
        records = self._dumped(["--log", str(self.log), "dump"])
        self.assertEqual(records[0]["outcome"], "EMPTY")
        self.assertIsNone(records[0]["explanation"])
        self.assertEqual(records[1]["outcome"], "PARSE-FAIL")
        self.assertIsNone(records[1]["explanation"])


if __name__ == "__main__":
    unittest.main()
