"""
Analyze agent usage patterns and sequences.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, extract_steps, get_agent_sequence


def analyze_agent_usage(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze agent usage patterns.
    
    Returns:
        Dictionary with agent usage statistics.
    """
    all_steps = extract_steps(sessions)
    
    # Filter agent calls
    agent_calls = [step for step in all_steps if step.get("step_type") == "agent_call"]
    
    # Count agent usage
    agent_names = [step.get("agent_name") for step in agent_calls if step.get("agent_name")]
    agent_counts = dict(Counter(agent_names))
    
    # Agent usage per session
    agent_usage_per_session = defaultdict(list)
    for session in sessions:
        steps = session.get("steps", [])
        session_agents = [step.get("agent_name") for step in steps 
                         if step.get("step_type") == "agent_call" and step.get("agent_name")]
        unique_agents = set(session_agents)
        for agent in unique_agents:
            agent_usage_per_session[agent].append(session_agents.count(agent))
    
    # Agent usage statistics per session
    agent_stats_per_session = {}
    for agent, counts in agent_usage_per_session.items():
        agent_stats_per_session[agent] = {
            "total_sessions_using": len(counts),
            "avg_per_session": sum(counts) / len(counts) if counts else 0,
            "max_per_session": max(counts) if counts else 0,
        }
    
    # Agent sequences
    agent_sequences = []
    for session in sessions:
        sequence = get_agent_sequence(session)
        if sequence:
            agent_sequences.append(sequence)
    
    # Most common agent sequences (bigrams)
    bigrams = []
    for sequence in agent_sequences:
        for i in range(len(sequence) - 1):
            if sequence[i] and sequence[i + 1]:
                bigrams.append((sequence[i], sequence[i + 1]))
    
    bigram_counts = dict(Counter(bigrams))
    
    # Agent usage by decision type
    agent_usage_by_decision = defaultdict(lambda: defaultdict(int))
    for session in sessions:
        decision = session.get("final_decision", "unknown")
        steps = session.get("steps", [])
        session_agents = [step.get("agent_name") for step in steps 
                         if step.get("step_type") == "agent_call" and step.get("agent_name")]
        for agent in session_agents:
            agent_usage_by_decision[decision][agent] += 1
    
    return {
        "agent_counts": agent_counts,
        "agent_stats_per_session": agent_stats_per_session,
        "total_agent_calls": len(agent_calls),
        "unique_agents": len(agent_counts),
        "agent_sequences": {
            "total_sequences": len(agent_sequences),
            "avg_sequence_length": sum(len(s) for s in agent_sequences) / len(agent_sequences) if agent_sequences else 0,
        },
        "common_bigrams": {f"{k[0]} -> {k[1]}": v for k, v in sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:20]},
        "agent_usage_by_decision": {k: dict(v) for k, v in agent_usage_by_decision.items()},
    }


def main():
    """Main function to run agent usage analysis."""
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
    stats = analyze_agent_usage(valid_sessions)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "agent_usage.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Agent usage analysis saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total agent calls: {stats['total_agent_calls']}")
    print(f"  Unique agents: {stats['unique_agents']}")
    print(f"  Most used agents:")
    for agent, count in sorted(stats['agent_counts'].items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {agent}: {count} times")


if __name__ == "__main__":
    main()

