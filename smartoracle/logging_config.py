"""Logging configuration for the triage agent."""

import logging
import os
from pathlib import Path


def _make_formatter(debug: bool) -> logging.Formatter:
    time_format = "%Y-%m-%dT%H:%M:%S"
    if debug:
        fmt = (
            "%(asctime)s.%(msecs)03d %(levelname)s %(name)s:%(lineno)d "
            "[%(process)d] %(message)s"
        )
    else:
        fmt = "%(asctime)s %(levelname)s %(name)s: %(message)s"
    return logging.Formatter(fmt=fmt, datefmt=time_format)


def setup_logging(logs_dir: Path | None = None, debug: bool = False) -> None:
    """Configure root logger with console and optional file output.

    Parameters
    ----------
    logs_dir: Path | None
        If provided, write logs to logs_dir/agent.log as well.
    debug: bool
        Enables DEBUG level and more verbose formatter when True.
    """
    level = logging.DEBUG if debug or os.environ.get("DEBUG") else logging.INFO
    root = logging.getLogger()
    root.setLevel(level)

    # Clear existing handlers to avoid duplication when reconfiguring
    for h in list(root.handlers):
        root.removeHandler(h)

    formatter = _make_formatter(debug)

    console = logging.StreamHandler()
    console.setLevel(level)
    console.setFormatter(formatter)
    root.addHandler(console)

    if logs_dir:
        logs_dir.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(logs_dir / "agent.log", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
