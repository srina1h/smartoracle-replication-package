"""
Analyze basic statistics from reasoning logs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, compute_statistics


def analyze_basic_stats(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute basic statistics about sessions.
    
    Returns:
        Dictionary with statistics on:
        - Step counts
        - Session durations
        - Decision distribution
        - Confidence scores
    """
    step_counts = []
    durations = []
    decisions = []
    confidences = []
    
    for session in sessions:
        # Step count
        steps = session.get("steps", [])
        step_counts.append(len(steps))
        
        # Duration
        duration_ms = session.get("total_duration_ms")
        if duration_ms is not None:
            durations.append(duration_ms)
        
        # Decision
        decision = session.get("final_decision")
        if decision:
            decisions.append(decision)
        
        # Confidence
        confidence = session.get("final_confidence")
        if confidence is not None:
            confidences.append(confidence)
    
    # Compute statistics
    step_stats = compute_statistics(step_counts)
    duration_stats = compute_statistics(durations)
    
    # Decision distribution
    decision_dist = dict(Counter(decisions))
    
    # Confidence statistics
    confidence_stats = compute_statistics(confidences) if confidences else {}
    
    return {
        "step_count": step_stats,
        "duration_ms": duration_stats,
        "decision_distribution": decision_dist,
        "confidence": confidence_stats,
        "total_sessions": len(sessions),
        "sessions_with_duration": len(durations),
        "sessions_with_decision": len(decisions),
        "sessions_with_confidence": len(confidences),
    }


def main():
    """Main function to run basic statistics analysis."""
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
    stats = analyze_basic_stats(valid_sessions)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "basic_stats.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Basic statistics saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total valid sessions: {stats['total_sessions']}")
    print(f"  Average steps per session: {stats['step_count']['mean']:.2f}")
    print(f"  Average duration: {stats['duration_ms']['mean']:.2f} ms")
    print(f"  Decision distribution: {stats['decision_distribution']}")


if __name__ == "__main__":
    main()

