#!/usr/bin/env python3
"""
Test runner script for the Alligator project.

This script provides convenient ways to run the test suite with different configurations.
"""

import argparse
import subprocess
import sys


def run_tests(
    test_path: str = "tests/",
    coverage: bool = False,
    verbose: bool = False,
    markers: str | None = None,
    parallel: bool = False,
    fail_fast: bool = False,
    show_capture: bool = False,
):
    """Run the test suite with specified options."""

    cmd = ["python", "-m", "pytest"]

    # Add test path
    cmd.append(test_path)

    # Add verbosity
    if verbose:
        cmd.append("-v")
    else:
        cmd.append("-q")

    # Add coverage
    if coverage:
        cmd.extend(
            [
                "--cov=alligator",
                "--cov-report=html",
                "--cov-report=term-missing",
                "--cov-fail-under=80",
            ]
        )

    # Add markers
    if markers:
        cmd.extend(["-m", markers])

    # Add parallel execution
    if parallel:
        cmd.extend(["-n", "auto"])

    # Add fail fast
    if fail_fast:
        cmd.append("-x")

    # Show capture
    if show_capture:
        cmd.append("-s")

    print(f"Running command: {' '.join(cmd)}")
    return subprocess.run(cmd)


def main():
    """Main entry point for the test runner."""
    parser = argparse.ArgumentParser(description="Run Alligator tests")

    parser.add_argument(
        "test_path", nargs="?", default="tests/", help="Path to tests (default: tests/)"
    )

    parser.add_argument("--coverage", action="store_true", help="Run with coverage reporting")

    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")

    parser.add_argument(
        "--markers",
        "-m",
        help="Run only tests matching given mark expression (e.g., 'unit', 'not slow')",
    )

    parser.add_argument(
        "--parallel",
        "-n",
        action="store_true",
        help="Run tests in parallel (requires pytest-xdist)",
    )

    parser.add_argument("--fail-fast", "-x", action="store_true", help="Stop on first failure")

    parser.add_argument(
        "--capture", "-s", action="store_true", help="Show print statements and logging"
    )

    # Predefined test suites
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")

    parser.add_argument("--integration", action="store_true", help="Run only integration tests")

    parser.add_argument("--slow", action="store_true", help="Run only slow tests")

    parser.add_argument(
        "--fast", action="store_true", help="Run fast tests only (exclude slow tests)"
    )

    args = parser.parse_args()

    # Set markers based on predefined suites
    markers = args.markers
    if args.unit:
        markers = "unit"
    elif args.integration:
        markers = "integration"
    elif args.slow:
        markers = "slow"
    elif args.fast:
        markers = "not slow"

    # Run the tests
    result = run_tests(
        test_path=args.test_path,
        coverage=args.coverage,
        verbose=args.verbose,
        markers=markers,
        parallel=args.parallel,
        fail_fast=args.fail_fast,
        show_capture=args.capture,
    )

    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
