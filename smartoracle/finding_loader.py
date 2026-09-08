"""Finding loader for differential testing case directories."""

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class FindingLoader:
    """Load a finding from a differential testing case directory.

    Expected layout (flexible):
    - case_summary.txt: optional human-readable summary
    - finding_data/: may contain *.js repros, metadata.json, etc.
    - engine subdirs (v8/, javascriptcore/, spidermonkey/, graaljs/): optional
    """

    def __init__(self, case_dir: Path) -> None:
        self.case_dir = case_dir

    def load(self) -> Dict[str, Any]:
        title = self._read_title()
        description = self._read_description()
        repro = self._find_repro_js()
        finding: Dict[str, Any] = {
            "title": title or self.case_dir.name,
            "description": description,
            "minimal_repro": repro or "",
            "context": {"case_dir": str(self.case_dir)},
        }
        logger.info("Loaded finding from %s", self.case_dir)
        return finding

    def _read_title(self) -> Optional[str]:
        summary = self.case_dir / "case_summary.txt"
        if summary.exists():
            try:
                first = summary.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                return first[0].strip() if first else None
            except Exception as exc:
                logger.debug("Failed reading title from %s: %s", summary, exc)
        return None

    def _read_description(self) -> str:
        summary = self.case_dir / "case_summary.txt"
        if summary.exists():
            try:
                return summary.read_text(encoding="utf-8", errors="ignore").strip()
            except Exception as exc:
                logger.debug("Failed reading description from %s: %s", summary, exc)
        return ""

    def _find_repro_js(self) -> Optional[str]:
        # Prefer files under finding_data/, fallback to any .js in case dir
        candidates = []
        data_dir = self.case_dir / "finding_data"
        if data_dir.exists():
            candidates.extend(sorted(data_dir.rglob("*.js")))
        candidates.extend(sorted(self.case_dir.glob("*.js")))
        for p in candidates:
            try:
                content = p.read_text(encoding="utf-8", errors="ignore").strip()
                if content:
                    logger.debug("Using repro from %s", p)
                    return content
            except Exception as exc:
                logger.debug("Failed reading repro from %s: %s", p, exc)
        return None
