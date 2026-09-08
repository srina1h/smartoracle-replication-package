import os, json, time, sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from baseline_runner import run_and_maybe_report
from dotenv import load_dotenv

# Load .env from this folder so GOOGLE_API_KEY is available
_BASELINE_DIR = Path(__file__).parent
load_dotenv(_BASELINE_DIR / ".env")

IN_ROOT = Path(os.getenv("BASELINE_INPUT_ROOT", "data/baseline_inputs"))
OUT_ROOT = Path(os.getenv("BASELINE_OUTPUT_ROOT", "results/baseline_outputs"))


def _iter_case_files(root: Path) -> List[Path]:
    exts = {".json"}
    files: List[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return files


def _rel_out_path(in_path: Path, in_root: Path, out_root: Path) -> Path:
    rel = in_path.relative_to(in_root)
    # mirror structure, change filename to triage_result.json
    # Many inputs already named; we preserve base name but suffix with _baseline.json
    stem = in_path.stem
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{stem}_baseline.json"


def process_all(in_root: Path = IN_ROOT, out_root: Path = OUT_ROOT) -> Dict[str, Any]:
    t0 = time.time()
    cases = _iter_case_files(in_root)
    
    # Ensure output root exists
    out_root.mkdir(parents=True, exist_ok=True)
    
    summary: Dict[str, Any] = {
        "input_root": str(in_root),
        "output_root": str(out_root),
        "count": len(cases),
        "processed": [],
        "errors": [],
    }

    for idx, case_path in enumerate(sorted(cases)):
        print(f"Processing case {idx+1}/{len(cases)}: {case_path.name}")
        try:
            case = json.loads(case_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  ERROR loading case: {e}")
            summary["errors"].append({"path": str(case_path), "error": str(e)})
            continue

        try:
            result = run_and_maybe_report(case)
            out_path = _rel_out_path(case_path, in_root, out_root)
            out_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

            # Collect per-case token/cost if present
            metrics = result.get("result", {}).get("metrics") if result.get("action") else result.get("metrics")
            decision = (result.get("result") or result).get("decision")
            tokens = (metrics or {}).get("summary", {}).get("total_tokens")
            elapsed = (metrics or {}).get("summary", {}).get("total_elapsed_s")
            print(f"  COMPLETED: {decision} (tokens: {tokens}, elapsed: {elapsed}s)")
            
            summary["processed"].append({
                "path": str(case_path),
                "out": str(out_path),
                "decision": decision,
                "tokens": tokens,
                "cost": (metrics or {}).get("summary", {}).get("total_cost_usd"),
                "elapsed_s": elapsed,
            })
        except Exception as e:
            print(f"  ERROR processing case: {e}")
            summary["errors"].append({"path": str(case_path), "error": str(e)})

    summary["elapsed_s"] = time.time() - t0
    # Write a batch summary
    (out_root / "batch_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    s = process_all()
    print(json.dumps(s, ensure_ascii=False, indent=2))

