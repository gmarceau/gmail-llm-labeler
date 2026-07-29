#!/usr/bin/env python3
"""Command-line interface for the ETL pipeline."""

import argparse
import json
import logging
import sys
from pathlib import Path

import pydash
import yaml

from ..config import PathConfig
from ..database import EmailDatabase
from ..gmail_utils import extract_address, extract_domain
from .config import PipelineConfig
from .orchestrator import EmailPipeline

# Backward-compatible alias; extract_domain now lives in gmail_utils to avoid an
# import cycle (transform_stage needs it too, and it must not import from cli).
_extract_domain = extract_domain


class _ExecuteOnlyInfo(logging.Filter):
    """At default verbosity, suppress INFO messages not from execute()."""
    def filter(self, record):
        if record.levelno != logging.INFO:
            return True
        return record.funcName == "execute"


def setup_logging(verbosity: int):
    """Set up logging based on verbosity level."""
    levels = [logging.INFO, logging.INFO, logging.DEBUG]
    level = levels[min(verbosity, len(levels) - 1)]

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - [%(funcName)s] - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if verbosity == 0:
        _f = _ExecuteOnlyInfo()
        for handler in logging.getLogger().handlers:
            handler.addFilter(_f)

    logging.getLogger("httpx").setLevel(logging.WARNING)


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Email Auto-Labeler ETL Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  %(prog)s run

  # Run with custom config file
  %(prog)s run --config my_config.yaml

  # Run in dry-run mode
  %(prog)s run --dry-run

  # Run in preview mode
  %(prog)s run --preview

  # Run specific stage only
  %(prog)s run-stage extract --config my_config.yaml

  # Generate sample configuration
  %(prog)s generate-config --output my_config.yaml
        """,
    )

    # Add subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the complete pipeline")
    run_parser.add_argument("--config", "-c", type=str, help="Path to configuration YAML file")
    run_parser.add_argument("--dry-run", action="store_true", help="Run without making any changes")
    run_parser.add_argument("--preview", action="store_true", help="Preview what would be done")
    run_parser.add_argument("--test", action="store_true", help="Run in test mode with mock data")
    run_parser.add_argument("--source", choices=["gmail", "database"], help="Override data source")
    run_parser.add_argument("--query", type=str, help="Override Gmail query")
    run_parser.add_argument("--limit", type=int, help="Limit number of emails to process")
    run_parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=0,
        help="Increase verbosity (-v for INFO, -vv for DEBUG)",
    )

    # Run stage command
    stage_parser = subparsers.add_parser("run-stage", help="Run a specific stage")
    stage_parser.add_argument(
        "stage", choices=["extract", "transform", "load", "sync"], help="Stage to run"
    )
    stage_parser.add_argument("--config", "-c", type=str, help="Path to configuration YAML file")
    stage_parser.add_argument("--input", type=str, help="Input data file (JSON) for the stage")
    stage_parser.add_argument("--output", type=str, help="Output file (JSON) to save stage results")
    stage_parser.add_argument(
        "--dry-run", action="store_true", help="Run without making any changes"
    )
    stage_parser.add_argument(
        "-v", "--verbose", action="count", default=0, help="Increase verbosity"
    )

    # Generate config command
    config_parser = subparsers.add_parser(
        "generate-config", help="Generate a sample configuration file"
    )
    config_parser.add_argument(
        "--output",
        "-o",
        type=str,
        default="pipeline_config.yaml",
        help="Output path for configuration file",
    )
    config_parser.add_argument(
        "--full", action="store_true", help="Generate full configuration with all options"
    )

    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate a configuration file")
    validate_parser.add_argument("config", type=str, help="Path to configuration file to validate")

    # Show metrics command
    metrics_parser = subparsers.add_parser("show-metrics", help="Show metrics from the last run")
    metrics_parser.add_argument(
        "--file", type=str, default="pipeline_metrics.json", help="Path to metrics file"
    )

    # Review senders command
    review_parser = subparsers.add_parser(
        "review-senders",
        help="List senders not yet covered by sender_rules, as pasteable rule candidates (YAML)",
    )
    review_parser.add_argument(
        "--config", "-c", type=str, default="config_production_7b.yaml",
        help="Path to configuration YAML file (used for DB path and existing sender_rules)",
    )

    return parser


def run_pipeline(args):
    """Run the complete pipeline."""
    # Load configuration
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
        logging.info(f"Loaded configuration from {args.config}")
    else:
        config = PipelineConfig.from_env()
        logging.info("Using configuration from environment")

    # Override configuration with command-line arguments
    if args.source:
        config.extract.source = args.source
    if args.query:
        config.extract.gmail_query = args.query
    if args.limit:
        config.extract.max_results = args.limit
        config.extract.batch_size = min(args.limit, config.extract.batch_size)

    # Create and run pipeline
    pipeline = EmailPipeline(config)

    result = pipeline.run(dry_run=args.dry_run, preview_mode=args.preview, test_mode=args.test)

    logging.info("\n" + "=" * 60)
    logging.info("PIPELINE RUN COMPLETE")
    logging.info("=" * 60)
    logging.info(f"Run ID: {result.run_id}")
    logging.info(f"Duration: {(result.end_time - result.start_time).total_seconds():.2f} seconds")
    logging.info(f"Stages completed: {', '.join(result.stages_completed)}")
    logging.info(f"Emails processed: {result.emails_processed}")
    logging.info(f"Successful: {result.successful}")
    logging.info(f"Failed: {result.failed}")

    if result.errors:
        logging.error(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            logging.error(f"  - {error}")
        if len(result.errors) > 5:
            logging.error(f"  ... and {len(result.errors) - 5} more")

    logging.info("=" * 60)

    return 0 if result.failed == 0 else 1


def run_stage(args):
    """Run a specific stage."""
    import json

    # Load configuration
    if args.config:
        config = PipelineConfig.from_yaml(args.config)
    else:
        config = PipelineConfig.from_env()

    # Load input data if provided
    input_data = None
    if args.input:
        with open(args.input) as f:
            input_data = json.load(f)
        logging.info(f"Loaded input data from {args.input}")

    # Create pipeline and run stage
    pipeline = EmailPipeline(config)

    try:
        result = pipeline.run_stage(args.stage, input_data, dry_run=args.dry_run)

        # Save output if requested
        if args.output and result is not None:
            # Convert result to JSON-serializable format
            if isinstance(result, list):
                output_data = [
                    item.__dict__ if hasattr(item, "__dict__") else item for item in result
                ]
            else:
                output_data = result

            with open(args.output, "w") as f:
                json.dump(output_data, f, indent=2, default=str)
            print(f"Saved stage output to {args.output}")

        print(f"Stage '{args.stage}' completed successfully")
        return 0

    except Exception as e:
        print(f"Stage '{args.stage}' failed: {e}", file=sys.stderr)
        return 1


def generate_config(args):
    """Generate a sample configuration file."""
    config = PipelineConfig()

    if args.full:
        # Add example customizations for full config
        config.load.category_actions.update(
            {
                "Custom Category": ["apply_label", "star"],
            }
        )

    # Save to file
    config.to_yaml(args.output)
    print(f"Generated configuration file: {args.output}")

    # Print usage instructions
    print("\nEdit the configuration file to customize:")
    print("  - Data source (Gmail or database)")
    print("  - Gmail query for fetching emails")
    print("  - LLM service (OpenAI or Ollama)")
    print("  - Email categories")
    print("  - Actions per category")
    print("  - Database and metrics settings")

    return 0


def validate_config(args):
    """Validate a configuration file."""
    try:
        config = PipelineConfig.from_yaml(args.config)
        print(f"✓ Configuration file is valid: {args.config}")

        # Print summary
        print("\nConfiguration summary:")
        print(f"  Extract source: {config.extract.source}")
        print(f"  Transform LLM: {config.transform.llm_service}")
        print(f"  Categories: {len(config.transform.categories)}")
        print(f"  Database: {config.sync.database_path}")
        print(f"  Dry run: {config.dry_run}")

        return 0

    except Exception as e:
        print(f"✗ Configuration file is invalid: {e}", file=sys.stderr)
        return 1


def show_metrics(args):
    """Show metrics from the last run."""
    import json

    metrics_path = Path(args.file)

    if not metrics_path.exists():
        print(f"Metrics file not found: {args.file}", file=sys.stderr)
        return 1

    try:
        with open(metrics_path) as f:
            metrics = json.load(f)

        print("=" * 60)
        print("PIPELINE METRICS")
        print("=" * 60)
        print(f"Run ID: {metrics.get('run_id', 'N/A')}")
        print(f"Start: {metrics.get('start_time', 'N/A')}")
        print(f"End: {metrics.get('end_time', 'N/A')}")

        if "summary" in metrics:
            summary = metrics["summary"]
            print("\nSummary:")
            print(f"  Total processed: {summary.get('total_processed', 0)}")
            print(f"  Successful: {summary.get('successful', 0)}")
            print(f"  Failed: {summary.get('failed', 0)}")

            if "categories" in summary:
                print("\nCategories:")
                for category, count in summary["categories"].items():
                    print(f"  {category}: {count}")

            if "actions" in summary:
                print("\nActions:")
                for action, count in summary["actions"].items():
                    print(f"  {action}: {count}")

        if "pipeline_metrics" in metrics:
            print("\nPipeline Metrics:")
            for key, value in metrics["pipeline_metrics"].items():
                if isinstance(value, float):
                    print(f"  {key}: {value:.2f}")
                else:
                    print(f"  {key}: {value}")

        print("=" * 60)
        return 0

    except Exception as e:
        print(f"Failed to read metrics: {e}", file=sys.stderr)
        return 1


def review_senders(args):
    """List senders not yet covered by sender_rules, as pasteable rule candidates (YAML).

    Groups cached email metadata by a `rule` key: an individual address for senders on a
    personal domain, otherwise the registered domain. Senders already covered by a
    sender_rules entry (by address or domain) are omitted.
    """
    config = PipelineConfig.from_yaml(args.config) if args.config else PipelineConfig.from_env()
    # sender_rules keys mix full addresses (recruiter@gmail.com) and bare domains
    # (substack.com); a sender is already covered if either its address or domain has a rule.
    rule_keys = set(getattr(config.transform, "sender_rules", {}).keys())
    known_addresses = {k for k in rule_keys if "@" in k}
    known_domains = rule_keys - known_addresses
    # On personal/freemail domains every sender is a different person, so grouping by domain
    # is useless; break those out per individual address (the pasteable sender_rule key).
    personal_domains = set(getattr(config.transform, "personal_domains", []))

    path_config = PathConfig(config_file=args.config)
    db = EmailDatabase(database_file=str(path_config.database_file))

    rows = db.get_all_email_metadata()
    flat = [
        {
            "domain": _extract_domain(sender),
            "address": extract_address(sender),
            "subject": subject or "",
            "headers": json.loads(headers_json) if headers_json else {},
            "has_unsubscribe": bool(has_unsubscribe),
        }
        for _, subject, sender, headers_json, has_unsubscribe in rows
        if sender
    ]
    filtered = [
        r for r in flat
        if r["domain"]
        and r["domain"] not in known_domains
        and r["address"] not in known_addresses
    ]

    # The rule key is exactly what you'd paste into sender_rules: an address for personal
    # domains (when parseable), otherwise the domain.
    for r in filtered:
        r["rule"] = r["address"] if (r["domain"] in personal_domains and r["address"]) else r["domain"]

    grouped = pydash.group_by(filtered, "rule")
    # Full per-email list per group (no cap): recruiters and the like are told apart by their
    # distinct subjects, so the whole list is the signal.
    result = [
        {
            "rule": rule,
            "count": len(items),
            "samples": [
                {
                    "subject": i["subject"],
                    "headers": i["headers"],
                    "has_unsubscribe": i["has_unsubscribe"],
                }
                for i in items
            ],
        }
        for rule, items in sorted(grouped.items(), key=lambda x: -len(x[1]))
    ]

    print(yaml.dump(result, default_flow_style=False, sort_keys=False, allow_unicode=True))
    db.close()
    return 0


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Set up logging
    if hasattr(args, "verbose"):
        setup_logging(args.verbose)

    # Execute command
    if args.command == "run":
        return run_pipeline(args)
    elif args.command == "run-stage":
        return run_stage(args)
    elif args.command == "generate-config":
        return generate_config(args)
    elif args.command == "validate-config":
        return validate_config(args)
    elif args.command == "show-metrics":
        return show_metrics(args)
    elif args.command == "review-senders":
        return review_senders(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
