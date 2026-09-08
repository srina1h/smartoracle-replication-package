#!/usr/bin/env python3
"""
Convert raw test files from generated_diff_findings_preprocessed to JSON format
expected by the baseline runner.
"""

import os, json, sys
from pathlib import Path
from typing import Dict, Any, List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

def convert_case_directory(case_dir: Path) -> Dict[str, Any]:
    """Convert a case directory to JSON format."""
    
    # Read input.js
    input_js_path = case_dir / "input.js"
    if not input_js_path.exists():
        raise ValueError(f"input.js not found in {case_dir}")
    
    input_code = input_js_path.read_text(encoding="utf-8")
    
    # Read engine results
    engines = ["graaljs", "quickjs", "v8", "moddablexs"]
    engine_results = []
    divergence_groups = {}
    
    for engine in engines:
        stdout_path = case_dir / f"{engine}.txt"
        stderr_path = case_dir / f"{engine}_stderr.txt"
        
        stdout = ""
        stderr = ""
        
        if stdout_path.exists():
            stdout = stdout_path.read_text(encoding="utf-8")
        if stderr_path.exists():
            stderr = stderr_path.read_text(encoding="utf-8")
        
        # Determine exit code based on stderr content
        exit_code = 0 if not stderr.strip() else 1
        
        engine_results.append({
            "engine": engine,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr
        })
        
        # Group by signature for divergence analysis
        signature = stderr.strip() if stderr.strip() else stdout.strip()
        if signature not in divergence_groups:
            divergence_groups[signature] = []
        divergence_groups[signature].append(engine)
    
    # Create divergence analysis
    groups = []
    for signature, engines_list in divergence_groups.items():
        groups.append({
            "engines": engines_list,
            "signature": signature
        })
    
    has_divergence = len(groups) > 1
    
    # Create the JSON structure
    case_data = {
        "test_id": str(case_dir.relative_to(case_dir.parent.parent)),
        "exit_code_pattern": "unknown",
        "cluster": "JEST",
        "input_code": input_code,
        "engine_results": engine_results,
        "divergence": {
            "has_divergence": has_divergence,
            "groups": groups
        }
    }
    
    return case_data

def convert_all_cases(input_root: Path, output_root: Path) -> Dict[str, Any]:
    """Convert all case directories to JSON files."""
    
    output_root.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "input_root": str(input_root),
        "output_root": str(output_root),
        "converted": [],
        "errors": []
    }
    
    # Find all case directories (directories containing input.js)
    for case_dir in input_root.rglob("*"):
        if case_dir.is_dir() and (case_dir / "input.js").exists():
            try:
                case_data = convert_case_directory(case_dir)
                
                # Create output path mirroring the input structure
                rel_path = case_dir.relative_to(input_root)
                output_dir = output_root / rel_path
                output_dir.mkdir(parents=True, exist_ok=True)
                output_file = output_dir / "case_data.json"
                
                # Write JSON file
                output_file.write_text(json.dumps(case_data, ensure_ascii=False, indent=2), encoding="utf-8")
                
                summary["converted"].append({
                    "input": str(case_dir),
                    "output": str(output_file),
                    "test_id": case_data["test_id"],
                    "has_divergence": case_data["divergence"]["has_divergence"]
                })
                
            except Exception as e:
                summary["errors"].append({
                    "path": str(case_dir),
                    "error": str(e)
                })
    
    # Write summary
    summary_file = output_root / "conversion_summary.json"
    summary_file.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    
    return summary

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Convert raw test files to JSON format")
    parser.add_argument("--input", type=Path, required=True, help="Input directory with raw test files")
    parser.add_argument("--output", type=Path, required=True, help="Output directory for JSON files")
    
    args = parser.parse_args()
    
    summary = convert_all_cases(args.input, args.output)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
