"""
Analyze tool usage patterns and sequences.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Set, Tuple
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, extract_steps, get_tool_sequence
EXCLUDED_TOOLS = {"transfer_to_agent", "test_case_minimizer"}


def analyze_tool_usage(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze tool usage patterns.
    
    Returns:
        Dictionary with tool usage statistics.
    """
    all_steps = extract_steps(sessions)
    
    # Filter tool calls
    tool_calls = [
        step for step in all_steps
        if step.get("step_type") == "tool_call"
        and step.get("tool_name") not in EXCLUDED_TOOLS
    ]
    
    # Count tool usage
    tool_names = [step.get("tool_name") for step in tool_calls if step.get("tool_name")]
    tool_counts = dict(Counter(tool_names))
    
    # Tool usage per session
    tool_usage_per_session = defaultdict(list)
    for session in sessions:
        steps = session.get("steps", [])
        session_tools = [
            step.get("tool_name") for step in steps 
            if step.get("step_type") == "tool_call"
            and step.get("tool_name")
            and step.get("tool_name") not in EXCLUDED_TOOLS
        ]
        unique_tools = set(session_tools)
        for tool in unique_tools:
            tool_usage_per_session[tool].append(session_tools.count(tool))
    
    # Tool usage statistics per session
    tool_stats_per_session = {}
    for tool, counts in tool_usage_per_session.items():
        tool_stats_per_session[tool] = {
            "total_sessions_using": len(counts),
            "avg_per_session": sum(counts) / len(counts) if counts else 0,
            "max_per_session": max(counts) if counts else 0,
        }
    
    # Tool sequences
    tool_sequences = []
    for session in sessions:
        sequence = get_tool_sequence(session)
        if sequence:
            tool_sequences.append(sequence)
    
    # Most common tool sequences (bigrams and trigrams)
    bigrams = []
    trigrams = []
    
    for sequence in tool_sequences:
        # Bigrams
        for i in range(len(sequence) - 1):
            if sequence[i] and sequence[i + 1]:
                bigrams.append((sequence[i], sequence[i + 1]))
        # Trigrams
        for i in range(len(sequence) - 2):
            if sequence[i] and sequence[i + 1] and sequence[i + 2]:
                trigrams.append((sequence[i], sequence[i + 1], sequence[i + 2]))
    
    bigram_counts = dict(Counter(bigrams))
    trigram_counts = dict(Counter(trigrams))
    
    # Tool usage by decision type
    tool_usage_by_decision = defaultdict(lambda: defaultdict(int))
    for session in sessions:
        decision = session.get("final_decision", "unknown")
        steps = session.get("steps", [])
        session_tools = [
            step.get("tool_name") for step in steps 
            if step.get("step_type") == "tool_call"
            and step.get("tool_name")
            and step.get("tool_name") not in EXCLUDED_TOOLS
        ]
        for tool in session_tools:
            tool_usage_by_decision[decision][tool] += 1
    
    return {
        "tool_counts": tool_counts,
        "tool_stats_per_session": tool_stats_per_session,
        "total_tool_calls": len(tool_calls),
        "unique_tools": len(tool_counts),
        "tool_sequences": {
            "total_sequences": len(tool_sequences),
            "avg_sequence_length": sum(len(s) for s in tool_sequences) / len(tool_sequences) if tool_sequences else 0,
        },
        "common_bigrams": {f"{k[0]} -> {k[1]}": v for k, v in sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:20]},
        "common_trigrams": {f"{k[0]} -> {k[1]} -> {k[2]}": v for k, v in sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)[:20]},
        "tool_usage_by_decision": {k: dict(v) for k, v in tool_usage_by_decision.items()},
    }


def main():
    """Main function to run tool usage analysis."""
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
    stats = analyze_tool_usage(valid_sessions)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "tool_usage.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Tool usage analysis saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total tool calls: {stats['total_tool_calls']}")
    print(f"  Unique tools: {stats['unique_tools']}")
    print(f"  Most used tools:")
    for tool, count in sorted(stats['tool_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {tool}: {count} times")


if __name__ == "__main__":
    main()

