"""Run bookkeeping: TensorBoard + CSV scalar logging and provenance capture."""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Any

_LOGGER_NAME = "rsma_aoi"


def get_logger(name: str = _LOGGER_NAME, level: int = logging.INFO) -> logging.Logger:
    """Return a configured stdout logger (idempotent)."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s", "%H:%M:%S"))
        logger.addHandler(handler)
        logger.propagate = False
    logger.setLevel(level)
    return logger


def git_commit_hash(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit, or ``None`` outside a repository."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(cwd) if cwd else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


class RunLogger:
    """Writes scalars to TensorBoard (if available) and to a CSV file."""

    def __init__(self, run_dir: str | Path, use_tensorboard: bool = True) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.csv_path = self.run_dir / "scalars.csv"
        self._csv_keys: list[str] = []
        self._csv_rows: list[dict[str, float]] = []
        self._writer = None
        if use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter

                self._writer = SummaryWriter(log_dir=str(self.run_dir / "tb"))
            except ImportError:  # pragma: no cover - tensorboard is optional
                self._writer = None
        self.logger = get_logger()

    def log_scalars(self, step: int, scalars: dict[str, float]) -> None:
        """Record ``scalars`` at global ``step``."""
        clean = {k: float(v) for k, v in scalars.items() if v is not None}
        if self._writer is not None:
            for key, value in clean.items():
                self._writer.add_scalar(key, value, step)
        self._append_csv(step, clean)

    def _append_csv(self, step: int, scalars: dict[str, float]) -> None:
        """Append a row, rewriting the file if new metric columns appeared.

        Metrics can show up late (e.g. ``episode/*`` only after the first
        episode finishes), so the column set is the union over all rows.
        """
        self._csv_rows.append({"step": float(step), **scalars})
        new_keys = [k for k in sorted(scalars) if k not in self._csv_keys]
        if new_keys:
            self._csv_keys.extend(new_keys)
            self._rewrite_csv()
            return
        with self.csv_path.open("a", encoding="utf-8") as fh:
            fh.write(self._format_row(self._csv_rows[-1]) + "\n")

    def _format_row(self, row: dict[str, float]) -> str:
        values = [f"{int(row['step'])}"]
        values += [f"{row.get(k, float('nan')):.8g}" for k in self._csv_keys]
        return ",".join(values)

    def _rewrite_csv(self) -> None:
        with self.csv_path.open("w", encoding="utf-8") as fh:
            fh.write("step," + ",".join(self._csv_keys) + "\n")
            for row in self._csv_rows:
                fh.write(self._format_row(row) + "\n")

    def log_meta(self, meta: dict[str, Any]) -> None:
        """Persist run provenance (seed, git commit, argv, ...)."""
        with (self.run_dir / "run_meta.json").open("w", encoding="utf-8") as fh:
            json.dump(meta, fh, indent=2, default=str)

    def close(self) -> None:
        if self._writer is not None:
            self._writer.flush()
            self._writer.close()
