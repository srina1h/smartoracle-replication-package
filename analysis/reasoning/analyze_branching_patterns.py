"""
Analyze step type sequences and common reasoning paths.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, get_step_type_sequence


def extract_ngrams(sequences: List[List[str]], n: int) -> List[Tuple[str, ...]]:
    """Extract n-grams from sequences."""
    ngrams = []
    for sequence in sequences:
        for i in range(len(sequence) - n + 1):
            ngram = tuple(sequence[i:i+n])
            ngrams.append(ngram)
    return ngrams


def analyze_branching_patterns(sessions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Analyze branching patterns in reasoning paths.
    
    Returns:
        Dictionary with branching pattern statistics.
    """
    # Extract step type sequences
    step_sequences = []
    for session in sessions:
        sequence = get_step_type_sequence(session)
        if sequence:
            step_sequences.append(sequence)
    
    # Sequence length statistics
    sequence_lengths = [len(seq) for seq in step_sequences]
    
    # Extract n-grams (unigrams, bigrams, trigrams, 4-grams)
    unigrams = []
    bigrams = []
    trigrams = []
    fourgrams = []
    
    for sequence in step_sequences:
        unigrams.extend(sequence)
        
        for i in range(len(sequence) - 1):
            bigrams.append((sequence[i], sequence[i + 1]))
        
        for i in range(len(sequence) - 2):
            trigrams.append((sequence[i], sequence[i + 1], sequence[i + 2]))
        
        for i in range(len(sequence) - 3):
            fourgrams.append((sequence[i], sequence[i + 1], sequence[i + 2], sequence[i + 3]))
    
    # Count n-grams
    unigram_counts = dict(Counter(unigrams))
    bigram_counts = dict(Counter(bigrams))
    trigram_counts = dict(Counter(trigrams))
    fourgram_counts = dict(Counter(fourgrams))
    
    # Common paths (top n-grams) - convert tuples to strings for JSON
    common_bigrams = {f"{k[0]} -> {k[1]}": v for k, v in sorted(bigram_counts.items(), key=lambda x: x[1], reverse=True)[:30]}
    common_trigrams = {f"{k[0]} -> {k[1]} -> {k[2]}": v for k, v in sorted(trigram_counts.items(), key=lambda x: x[1], reverse=True)[:30]}
    common_fourgrams = {f"{k[0]} -> {k[1]} -> {k[2]} -> {k[3]}": v for k, v in sorted(fourgram_counts.items(), key=lambda x: x[1], reverse=True)[:20]}
    
    # Analyze paths by decision type
    paths_by_decision = defaultdict(list)
    for session in sessions:
        decision = session.get("final_decision", "unknown")
        sequence = get_step_type_sequence(session)
        if sequence:
            paths_by_decision[decision].append(sequence)
    
    # Common paths per decision
    common_paths_by_decision = {}
    for decision, paths in paths_by_decision.items():
        # Get most common bigrams for this decision
        decision_bigrams = []
        for path in paths:
            for i in range(len(path) - 1):
                decision_bigrams.append((path[i], path[i + 1]))
        bigram_counts_decision = dict(Counter(decision_bigrams))
        common_paths_by_decision[decision] = {f"{k[0]} -> {k[1]}": v for k, v in sorted(bigram_counts_decision.items(), key=lambda x: x[1], reverse=True)[:10]}
    
    # Path patterns (common prefixes and suffixes)
    path_prefixes = defaultdict(int)
    path_suffixes = defaultdict(int)
    
    for sequence in step_sequences:
        if len(sequence) >= 2:
            prefix = tuple(sequence[:2])
            path_prefixes[prefix] += 1
        if len(sequence) >= 2:
            suffix = tuple(sequence[-2:])
            path_suffixes[suffix] += 1
    
    # Decision points: steps that lead to decisions
    decision_points = []
    for session in sessions:
        steps = session.get("steps", [])
        for i, step in enumerate(steps):
            if step.get("step_type") == "decision":
                if i > 0:
                    decision_points.append(steps[i - 1].get("step_type", "unknown"))
    
    decision_point_counts = dict(Counter(decision_points))
    
    return {
        "sequence_lengths": {
            "count": len(sequence_lengths),
            "min": min(sequence_lengths) if sequence_lengths else 0,
            "max": max(sequence_lengths) if sequence_lengths else 0,
            "mean": sum(sequence_lengths) / len(sequence_lengths) if sequence_lengths else 0,
            "median": sorted(sequence_lengths)[len(sequence_lengths) // 2] if sequence_lengths else 0,
        },
        "step_type_distribution": unigram_counts,
        "common_bigrams": common_bigrams,
        "common_trigrams": common_trigrams,
        "common_fourgrams": common_fourgrams,
        "common_paths_by_decision": common_paths_by_decision,
        "common_prefixes": {f"{k[0]} -> {k[1]}": v for k, v in sorted(path_prefixes.items(), key=lambda x: x[1], reverse=True)[:10]},
        "common_suffixes": {f"{k[0]} -> {k[1]}": v for k, v in sorted(path_suffixes.items(), key=lambda x: x[1], reverse=True)[:10]},
        "decision_points": decision_point_counts,
        "total_sequences": len(step_sequences),
    }


def main():
    """Main function to run branching pattern analysis."""
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
    stats = analyze_branching_patterns(valid_sessions)
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = results_dir / "branching_patterns.json"
    with open(output_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"Branching pattern analysis saved to {output_file}")
    print(f"\nSummary:")
    print(f"  Total sequences: {stats['total_sequences']}")
    print(f"  Average sequence length: {stats['sequence_lengths']['mean']:.2f}")
    print(f"  Most common bigrams:")
    for bigram, count in list(stats['common_bigrams'].items())[:5]:
        print(f"    {bigram}: {count} times")


if __name__ == "__main__":
    main()

