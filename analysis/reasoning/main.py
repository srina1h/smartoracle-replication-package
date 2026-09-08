#!/usr/bin/env python3
"""
Main orchestrator script to run all reasoning log analyses.
"""

import sys
import subprocess
from pathlib import Path


def run_analysis_script(script_name: str):
    """Run an analysis script and handle errors."""
    script_path = Path(__file__).parent / script_name
    
    if not script_path.exists():
        print(f"Error: Script not found: {script_path}")
        return False
    
    print(f"\n{'='*60}")
    print(f"Running: {script_name}")
    print(f"{'='*60}")
    
    # Try to use venv Python if available
    venv_python = Path(__file__).parent / "venv" / "bin" / "python3"
    python_cmd = str(venv_python) if venv_python.exists() else sys.executable
    
    try:
        result = subprocess.run(
            [python_cmd, str(script_path)],
            cwd=Path(__file__).parent,
            check=True,
            capture_output=False
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running {script_name}: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error running {script_name}: {e}")
        return False


def main():
    """Run all analysis scripts in sequence."""
    print("Reasoning Log Analysis")
    print("=" * 60)
    
    # Check if logs directory exists (try current dir, then parent)
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir = Path(__file__).parent.parent / "logs"
    if not logs_dir.exists():
        print(f"Error: Logs directory not found. Tried: logs and {logs_dir.absolute()}")
        print("Please ensure the logs directory exists with sessions/ subdirectory.")
        return 1
    
    sessions_dir = logs_dir / "sessions"
    if not sessions_dir.exists():
        print(f"Error: Sessions directory not found: {sessions_dir.absolute()}")
        return 1
    
    # Run all analysis scripts
    scripts = [
        "analyze_basic_stats.py",
        "analyze_step_types.py",
        "analyze_tool_usage.py",
        "analyze_agent_usage.py",
        "analyze_branching_patterns.py",
    ]
    
    results = []
    for script in scripts:
        success = run_analysis_script(script)
        results.append((script, success))
    
    # Generate comprehensive report
    print(f"\n{'='*60}")
    print("Generating comprehensive report")
    print(f"{'='*60}")
    report_success = run_analysis_script("generate_report.py")
    
    # Generate visualizations
    print(f"\n{'='*60}")
    print("Generating visualizations")
    print(f"{'='*60}")
    viz_success = run_analysis_script("generate_visualizations.py")
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    
    failed = []
    for script, success in results:
        status = "✓" if success else "✗"
        print(f"{status} {script}")
        if not success:
            failed.append(script)
    
    if report_success:
        print(f"✓ generate_report.py")
    else:
        print(f"✗ generate_report.py")
        failed.append("generate_report.py")
    
    if viz_success:
        print(f"✓ generate_visualizations.py")
    else:
        print(f"✗ generate_visualizations.py")
        failed.append("generate_visualizations.py")
    
    if failed:
        print(f"\n{len(failed)} script(s) failed:")
        for script in failed:
            print(f"  - {script}")
        return 1
    else:
        print("\nAll analyses completed successfully!")
        print(f"\nResults saved to: {Path(__file__).parent / 'results'}")
        print(f"Comprehensive report: {Path(__file__).parent / 'results' / 'comprehensive_report.md'}")
        return 0


if __name__ == "__main__":
    sys.exit(main())

