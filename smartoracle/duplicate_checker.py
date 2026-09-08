"""Duplicate checker for finding similar prior reports."""

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[A-Za-z0-9_]+", (text or "").lower()))


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


@dataclass
class PriorIssue:
    id: str
    title: str
    description: str
    path: str
    timestamp: str


class DuplicateChecker:
    def __init__(self, reports_dir: Path, top_k: int = 10, dup_threshold: float = 0.6) -> None:
        self.reports_dir = reports_dir
        self.top_k = top_k
        self.dup_threshold = dup_threshold

    def _load_index(self) -> List[PriorIssue]:
        index_path = self.reports_dir / "index.json"
        if not index_path.exists():
            logger.info("Reports index not found: %s", index_path)
            return []
        try:
            data = json.loads(index_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.exception("Failed to read index %s: %s", index_path, exc)
            return []
        issues: List[PriorIssue] = []
        for item in data.get("issues", [])[-self.top_k:]:
            issues.append(
                PriorIssue(
                    id=item.get("id", ""),
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    path=item.get("path", ""),
                    timestamp=item.get("timestamp", ""),
                )
            )
        return list(reversed(issues))  # newest first

    def find_duplicates(self, title: str, description: str) -> List[dict]:
        candidates = self._load_index()
        q_tokens = _tokenize((title or "") + "\n" + (description or ""))
        matches: List[dict] = []
        for issue in candidates:
            i_tokens = _tokenize(issue.title + "\n" + issue.description)
            score = _jaccard(q_tokens, i_tokens)
            if score >= self.dup_threshold:
                matches.append({
                    "id": issue.id,
                    "title": issue.title,
                    "path": issue.path,
                    "timestamp": issue.timestamp,
                    "similarity": round(score, 3),
                })
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        logger.debug("Duplicate check found %d potential matches", len(matches))
        return matches
