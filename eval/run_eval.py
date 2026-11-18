"""Run the evaluation (including ablations) and save the results to eval/results/."""

from pathlib import Path


def run_evaluation(results_dir: Path) -> None:
    """Evaluate retrieval, extraction and the agent against the ground truth."""
    raise NotImplementedError


def main() -> None:
    """Command-line entry point."""
    raise NotImplementedError
