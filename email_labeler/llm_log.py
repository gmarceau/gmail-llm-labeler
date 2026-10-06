"""Read-only analysis of the LLM interaction and categorization error logs.

Both logs are append-only files of back-to-back JSON objects — the
interaction log pretty-printed, the error log compact — so neither is
valid JSON or JSONL as a whole; they need an incremental raw_decode
walk (see llm_service._log_interaction and _log_error, the writers).

Production records are told from test-suite pollution structurally:
pipeline prompts always start with the "Subject: " line built by
TransformStage._build_email_content, while test fixtures pass bare
strings like "Test email".
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import ERROR_LOG_FILE, LLM_LOG_FILE

PRODUCTION_PREFIX = "Subject: "
RUN_GAP_MINUTES = 15  # calls within one pipeline run are at most minutes apart


class Record:
    """One parsed log entry (interaction or error shape)."""

    def __init__(self, raw: Dict[str, Any], time: datetime):
        self.raw = raw
        self.time = time

    @property
    def is_production(self) -> bool:
        email = self.raw.get("processed_email") or self.raw.get("email_preview") or ""
        return email.startswith(PRODUCTION_PREFIX)

    @property
    def backend(self) -> str:
        service = self.raw.get("service")
        model = self.raw.get("model", "?")
        return f"{service}/{model}" if service else str(model)

    @property
    def outcome(self) -> str:
        """Resolved category, or EMPTY / PARSE-FAIL / the error string."""
        if "response" not in self.raw:  # error-log record
            return str(self.raw.get("error", "?"))
        response = self.raw["response"] or ""
        if not response:
            return "EMPTY"
        try:
            parsed = json.loads(response)
        except (TypeError, json.JSONDecodeError):
            return "PARSE-FAIL"
        if isinstance(parsed, dict):
            return str(parsed.get("category", "?"))
        return "PARSE-FAIL"

    @property
    def first_line(self) -> str:
        email = self.raw.get("processed_email") or self.raw.get("email_preview") or ""
        return email.split("\n", 1)[0]


def _record_time(raw: Dict[str, Any]) -> datetime:
    ts = raw.get("request_timestamp")
    if isinstance(ts, (int, float)):
        return datetime.fromtimestamp(ts)
    stamp = raw.get("timestamp")
    if isinstance(stamp, str):
        return datetime.fromisoformat(stamp)
    raise SystemExit(f"error: record without a usable timestamp: {str(raw)[:200]}")


def load_records(path: str) -> List[Record]:
    """Parse a concatenated-JSON log file into records, oldest first."""
    file = Path(path)
    if not file.exists():
        raise SystemExit(f"error: log file not found: {path}")
    text = file.read_text()
    decoder = json.JSONDecoder()
    records: List[Record] = []
    i = 0
    while i < len(text):
        while i < len(text) and text[i].isspace():
            i += 1
        if i >= len(text):
            break
        try:
            raw, i = decoder.raw_decode(text, i)
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: malformed JSON at byte {e.pos} in {path}") from e
        if not isinstance(raw, dict):
            raise SystemExit(f"error: non-object record at byte {i} in {path}")
        records.append(Record(raw, _record_time(raw)))
    return records


def split_runs(records: List[Record], gap_minutes: int = RUN_GAP_MINUTES) -> List[List[Record]]:
    """Split time-ordered records into contiguous runs at gaps > gap_minutes."""
    if not records:
        return []
    runs: List[List[Record]] = [[records[0]]]
    for rec in records[1:]:
        gap = (rec.time - runs[-1][-1].time).total_seconds() / 60
        if gap > gap_minutes:
            runs.append([rec])
        else:
            runs[-1].append(rec)
    return runs


def _outcome_counts(records: List[Record]) -> str:
    counts = Counter(r.outcome for r in records)
    return ", ".join(f"{k}={v}" for k, v in counts.most_common())


def _duration_line(run: List[Record]) -> str:
    durations = [r.raw["duration"] for r in run if "duration" in r.raw]
    if not durations:
        return ""
    return (
        f"mean {statistics.mean(durations):.1f}s  median {statistics.median(durations):.1f}s"
        f"  max {max(durations):.1f}s  total {sum(durations):.0f}s"
    )


def _selected_records(args: argparse.Namespace) -> List[Record]:
    path = args.errors_log if getattr(args, "errors", False) else args.log
    records = load_records(path)
    if args.since:
        records = [r for r in records if r.time >= args.since]
    return records


def cmd_days(args: argparse.Namespace) -> int:
    by_date: Dict[str, List[Record]] = {}
    for rec in _selected_records(args):
        by_date.setdefault(rec.time.strftime("%Y-%m-%d"), []).append(rec)
    if not by_date:
        print("no records")
        return 0
    for date in sorted(by_date):
        day = by_date[date]
        production = [r for r in day if r.is_production]
        print(f"{date}  real={len(production)}  test={len(day) - len(production)}")
        if production:
            print(f"    {_outcome_counts(production)}")
    return 0


def cmd_runs(args: argparse.Namespace) -> int:
    records = [r for r in _selected_records(args) if r.is_production]
    if not records:
        print("no production records")
        return 0
    for run in split_runs(records, args.gap):
        print(
            f"{run[0].time:%Y-%m-%d %H:%M:%S} -> {run[-1].time:%H:%M:%S}"
            f"  {len(run)} calls  {_duration_line(run)}"
        )
        print(f"    {_outcome_counts(run)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    records = _selected_records(args)
    if args.real:
        records = [r for r in records if r.is_production]
    elif args.test:
        records = [r for r in records if not r.is_production]
    for rec in records:
        if args.full:
            print(json.dumps(rec.raw, indent=2))
        else:
            duration = f"{rec.raw['duration']:6.1f}s" if "duration" in rec.raw else "     -"
            print(
                f"{rec.time:%Y-%m-%d %H:%M:%S}  {rec.backend:<24} {duration}"
                f"  {rec.outcome:<14} {rec.first_line}"
            )
    return 0


def _parse_since(value: str) -> datetime:
    try:
        return datetime.fromisoformat(value)
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"expected YYYY-MM-DD[ HH:MM], got {value!r}"
        ) from e


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-log",
        description="Read-only analysis of the LLM interaction and error logs.",
    )
    parser.add_argument(
        "--log", default=LLM_LOG_FILE, help="interaction log file (default: %(default)s)"
    )
    parser.add_argument(
        "--errors-log", default=ERROR_LOG_FILE, help="error log file (default: %(default)s)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def add(name: str, help: str) -> argparse.ArgumentParser:
        sp = sub.add_parser(name, help=help)
        sp.add_argument(
            "--since",
            type=_parse_since,
            metavar="YYYY-MM-DD[ HH:MM]",
            help="only records at/after this time",
        )
        return sp

    days = add("days", help="per-day counts, production vs test pollution")
    days.add_argument(
        "--errors", action="store_true", help="analyze the error log instead"
    )
    days.set_defaults(func=cmd_days)

    runs = add("runs", help="contiguous production runs (split at time gaps)")
    runs.add_argument(
        "--gap",
        type=int,
        default=RUN_GAP_MINUTES,
        metavar="MIN",
        help="gap in minutes that splits runs (default: %(default)s)",
    )
    runs.set_defaults(func=cmd_runs)

    show = add("show", help="one line per record")
    show.add_argument("--real", action="store_true", help="only production records")
    show.add_argument("--test", action="store_true", help="only test-suite pollution")
    show.add_argument("--full", action="store_true", help="print full JSON objects")
    show.add_argument("--errors", action="store_true", help="read the error log instead")
    show.set_defaults(func=cmd_show)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
