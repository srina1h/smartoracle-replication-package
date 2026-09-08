"""
Analyze step type distribution and frequency.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, extract_steps, compute_statistics


def analyze_step_types(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze step type distribution and frequency.
    
    Returns:
        Dictionary with step type statistics.
    """
    all_steps = extract_steps(sessions)
    
    # Count step types
    step_types = [step.get("step_type", "unknown") for step in all_steps]
    step_type_counts = dict(Counter(step_types))
    
    # Count step types per session
    step_type_counts_per_session = {step_type: [] for step_type in set(step_types)}
    
    for session in sessions:
        steps = session.get("steps", [])
        session_step_types = Counter(step.get("step_type", "unknown") for step in steps)
        
        for step_type in step_type_counts_per_session.keys():
            count = session_step_types.get(step_type, 0)
            step_type_counts_per_session[step_type].append(count)
    
    # Compute statistics for each step type
    step_type_stats = {}
    for step_type, counts in step_type_counts_per_session.items():
        step_type_stats[step_type] = compute_statistics(counts)
    
    # Overall statistics
    total_steps = len(all_steps)
    step_type_percentages = {
        step_type: (count / total_steps * 100) if total_steps > 0 else 0
        for step_type, count in step_type_counts.items()
    }
    
    return {
        "step_type_counts": step_type_counts,
        "step_type_percentages": step_type_percentages,
        "step_type_stats_per_session": step_type_stats,
        "total_steps": total_steps,
        "total_sessions": len(sessions),
    }


def main():
    """Main function to run step type analysis."""
    # Try current directory first, then parent directory
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir = Path(__file__).parent.parent / "logs"
    if not logs_dir.exists():
        print(f"Error: Logs directory not found. Tried: logs and {logs_dir}")
        return
    
    # Load and filter logs
    valid_sessions, failed_sessions = load_and_filter_logs(logs_dir)
    
    if not valid_sessions:
        print("No valid sessions found to analyze.")
        return
    
    # Analyze
    stats = analyze_step_types(valid_sessions)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "step_types.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Step type analysis saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total steps: {stats['total_steps']}")
    print(f"  Step type distribution:")
    for step_type, count in sorted(stats['step_type_counts'].items(), key=lambda x: x[1], reverse=True):
        percentage = stats['step_type_percentages'][step_type]
        print(f"    {step_type}: {count} ({percentage:.1f}%)")


if __name__ == "__main__":
    main()

