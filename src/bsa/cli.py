"""Command-line interface."""

import argparse
import sys
from pathlib import Path
from typing import Optional

from .common.config import load_config
from .common.logging import setup_logging


def main() -> None:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="Belief-Sensitive Assistance Research CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Generate command
    gen_parser = subparsers.add_parser("generate", help="Generate episodes")
    gen_parser.add_argument("--config", type=Path, required=True, help="Generator config file")
    gen_parser.add_argument("--output", type=Path, help="Output directory")
    gen_parser.add_argument("--seed", type=int, help="Random seed")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run experiments")
    run_parser.add_argument("--config", type=Path, required=True, help="Experiment config file")
    run_parser.add_argument("--output", type=Path, help="Output directory")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze results")
    analyze_parser.add_argument("--config", type=Path, required=True, help="Analysis config file")
    analyze_parser.add_argument("--input", type=Path, help="Input results directory")

    # Reproduce command
    repro_parser = subparsers.add_parser("reproduce", help="Full reproduction")
    repro_parser.add_argument("--small", action="store_true", help="Small dataset for CI")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Setup logging
    setup_logging()

    # Route to command handler
    if args.command == "generate":
        from .experiments.run_experiment import generate_episodes
        config = load_config(args.config)
        generate_episodes(config, output_dir=args.output, seed=args.seed)
    elif args.command == "run":
        from .experiments.run_experiment import run_experiments
        config = load_config(args.config)
        run_experiments(config, output_dir=args.output)
    elif args.command == "analyze":
        from .analysis.aggregate import analyze_results
        config = load_config(args.config)
        analyze_results(config, input_dir=args.input)
    elif args.command == "reproduce":
        from .experiments.run_experiment import reproduce
        reproduce(small=args.small)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
