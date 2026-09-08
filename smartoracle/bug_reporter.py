"""Bug reporter for generating JSON and Markdown reports."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _slugify(text: str, max_len: int = 60) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    if len(text) > max_len:
        text = text[:max_len].rstrip("-")
    return text or "issue"


class BugReporter:
    def __init__(self, reports_dir: Path) -> None:
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def _load_index(self) -> Dict[str, Any]:
        index_path = self.reports_dir / "index.json"
        if index_path.exists():
            try:
                return json.loads(index_path.read_text(encoding="utf-8"))
            except Exception as exc:
                logger.exception("Failed reading index.json: %s", exc)
        return {"issues": []}

    def _write_index(self, index: Dict[str, Any]) -> None:
        index_path = self.reports_dir / "index.json"
        index_path.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")

    def report(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        title = payload.get("title", "untitled")
        slug = _slugify(title)
        base_name = f"{now}_{slug}"
        json_path = self.reports_dir / f"{base_name}.json"
        md_path = self.reports_dir / f"{base_name}.md"

        # Write JSON
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

        # Write Markdown summary
        md = [
            f"# {title}",
            "",
            f"Timestamp (UTC): {now}",
            "",
            "## Summary",
            payload.get("summary", "(no summary)"),
            "",
            "## Details",
            "```json",
            json.dumps(payload, ensure_ascii=False, indent=2),
            "```",
        ]
        md_path.write_text("\n".join(md), encoding="utf-8")

        # Update index
        index = self._load_index()
        new_item = {
            "id": base_name,
            "title": title,
            "description": payload.get("description", ""),
            "path": str(json_path),
            "timestamp": now,
        }
        index.setdefault("issues", []).append(new_item)
        self._write_index(index)

        logger.info("Wrote report %s", json_path)
        return {"id": base_name, "json_path": str(json_path), "md_path": str(md_path)}
