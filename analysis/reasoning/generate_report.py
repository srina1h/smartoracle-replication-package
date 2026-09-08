"""
Generate comprehensive markdown report from all analysis results.
"""

import json
from pathlib import Path
from typing import Dict, Any
from datetime import datetime


def load_analysis_results(results_dir: Path) -> Dict[str, Any]:
    """Load all analysis result JSON files."""
    results = {}
    
    files = {
        "basic_stats": "basic_stats.json",
        "step_types": "step_types.json",
        "tool_usage": "tool_usage.json",
        "agent_usage": "agent_usage.json",
        "branching_patterns": "branching_patterns.json",
    }
    
    for key, filename in files.items():
        filepath = results_dir / filename
        if filepath.exists():
            with open(filepath, 'r') as f:
                results[key] = json.load(f)
        else:
            print(f"Warning: {filename} not found")
    
    return results


def format_number(num: float, decimals: int = 2) -> str:
    """Format a number for display."""
    return f"{num:.{decimals}f}"


def generate_report(results: Dict[str, Any], output_file: Path):
    """Generate comprehensive markdown report."""
    
    with open(output_file, 'w') as f:
        f.write("# Reasoning Log Analysis Report\n\n")
        f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n")
        
        # Basic Statistics
        if "basic_stats" in results:
            bs = results["basic_stats"]
            f.write("## Basic Statistics\n\n")
            f.write(f"**Total Sessions Analyzed:** {bs.get('total_sessions', 0)}\n\n")
            
            f.write("### Step Count Statistics\n\n")
            sc = bs.get("step_count", {})
            f.write(f"- **Average steps per session:** {format_number(sc.get('mean', 0))}\n")
            f.write(f"- **Min steps:** {sc.get('min', 0)}\n")
            f.write(f"- **Max steps:** {sc.get('max', 0)}\n")
            f.write(f"- **Median steps:** {format_number(sc.get('median', 0))}\n")
            f.write(f"- **25th percentile:** {format_number(sc.get('p25', 0))}\n")
            f.write(f"- **75th percentile:** {format_number(sc.get('p75', 0))}\n")
            f.write(f"- **95th percentile:** {format_number(sc.get('p95', 0))}\n\n")
            
            f.write("### Duration Statistics\n\n")
            dur = bs.get("duration_ms", {})
            f.write(f"- **Average duration:** {format_number(dur.get('mean', 0))} ms\n")
            f.write(f"- **Min duration:** {format_number(dur.get('min', 0))} ms\n")
            f.write(f"- **Max duration:** {format_number(dur.get('max', 0))} ms\n")
            f.write(f"- **Median duration:** {format_number(dur.get('median', 0))} ms\n\n")
            
            f.write("### Decision Distribution\n\n")
            dd = bs.get("decision_distribution", {})
            total_decisions = sum(dd.values())
            for decision, count in sorted(dd.items(), key=lambda x: x[1], reverse=True):
                percentage = (count / total_decisions * 100) if total_decisions > 0 else 0
                f.write(f"- **{decision}:** {count} ({format_number(percentage, 1)}%)\n")
            f.write("\n")
            
            f.write("### Confidence Statistics\n\n")
            conf = bs.get("confidence", {})
            if conf:
                f.write(f"- **Average confidence:** {format_number(conf.get('mean', 0))}\n")
                f.write(f"- **Min confidence:** {format_number(conf.get('min', 0))}\n")
                f.write(f"- **Max confidence:** {format_number(conf.get('max', 0))}\n")
                f.write(f"- **Median confidence:** {format_number(conf.get('median', 0))}\n\n")
            else:
                f.write("- No confidence data available\n\n")
        
        # Step Type Analysis
        if "step_types" in results:
            st = results["step_types"]
            f.write("## Step Type Analysis\n\n")
            f.write(f"**Total Steps:** {st.get('total_steps', 0)}\n\n")
            
            f.write("### Step Type Distribution\n\n")
            counts = st.get("step_type_counts", {})
            percentages = st.get("step_type_percentages", {})
            
            f.write("| Step Type | Count | Percentage |\n")
            f.write("|-----------|-------|------------|\n")
            for step_type in sorted(counts.keys(), key=lambda x: counts[x], reverse=True):
                count = counts[step_type]
                pct = percentages.get(step_type, 0)
                f.write(f"| {step_type} | {count} | {format_number(pct, 1)}% |\n")
            f.write("\n")
            
            f.write("### Average Steps per Type per Session\n\n")
            stats = st.get("step_type_stats_per_session", {})
            for step_type in sorted(stats.keys()):
                type_stats = stats[step_type]
                f.write(f"- **{step_type}:**\n")
                f.write(f"  - Average per session: {format_number(type_stats.get('mean', 0))}\n")
                f.write(f"  - Max per session: {type_stats.get('max', 0)}\n\n")
        
        # Tool Usage Analysis
        if "tool_usage" in results:
            tu = results["tool_usage"]
            f.write("## Tool Usage Analysis\n\n")
            f.write(f"**Total Tool Calls:** {tu.get('total_tool_calls', 0)}\n")
            f.write(f"**Unique Tools:** {tu.get('unique_tools', 0)}\n\n")
            
            f.write("### Most Frequently Used Tools\n\n")
            tool_counts = tu.get("tool_counts", {})
            f.write("| Tool | Count |\n")
            f.write("|------|-------|\n")
            for tool, count in sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                f.write(f"| {tool} | {count} |\n")
            f.write("\n")
            
            f.write("### Common Tool Sequences (Bigrams)\n\n")
            bigrams = tu.get("common_bigrams", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(bigrams.items())[:10]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
            
            f.write("### Tool Usage by Decision Type\n\n")
            by_decision = tu.get("tool_usage_by_decision", {})
            for decision, tools in sorted(by_decision.items()):
                f.write(f"#### {decision}\n\n")
                f.write("| Tool | Count |\n")
                f.write("|------|-------|\n")
                for tool, count in sorted(tools.items(), key=lambda x: x[1], reverse=True)[:5]:
                    f.write(f"| {tool} | {count} |\n")
                f.write("\n")
        
        # Agent Usage Analysis
        if "agent_usage" in results:
            au = results["agent_usage"]
            f.write("## Agent Usage Analysis\n\n")
            f.write(f"**Total Agent Calls:** {au.get('total_agent_calls', 0)}\n")
            f.write(f"**Unique Agents:** {au.get('unique_agents', 0)}\n\n")
            
            f.write("### Most Frequently Used Agents\n\n")
            agent_counts = au.get("agent_counts", {})
            f.write("| Agent | Count |\n")
            f.write("|-------|-------|\n")
            for agent, count in sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
                f.write(f"| {agent} | {count} |\n")
            f.write("\n")
            
            f.write("### Common Agent Sequences (Bigrams)\n\n")
            bigrams = au.get("common_bigrams", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(bigrams.items())[:10]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
        
        # Branching Pattern Analysis
        if "branching_patterns" in results:
            bp = results["branching_patterns"]
            f.write("## Branching Pattern Analysis\n\n")
            f.write(f"**Total Sequences Analyzed:** {bp.get('total_sequences', 0)}\n\n")
            
            sl = bp.get("sequence_lengths", {})
            f.write("### Sequence Length Statistics\n\n")
            f.write(f"- **Average length:** {format_number(sl.get('mean', 0))}\n")
            f.write(f"- **Min length:** {sl.get('min', 0)}\n")
            f.write(f"- **Max length:** {sl.get('max', 0)}\n")
            f.write(f"- **Median length:** {format_number(sl.get('median', 0))}\n\n")
            
            f.write("### Most Common Step Type Sequences (Bigrams)\n\n")
            bigrams = bp.get("common_bigrams", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(bigrams.items())[:15]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
            
            f.write("### Most Common Step Type Sequences (Trigrams)\n\n")
            trigrams = bp.get("common_trigrams", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(trigrams.items())[:10]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
            
            f.write("### Common Path Prefixes\n\n")
            prefixes = bp.get("common_prefixes", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(prefixes.items())[:10]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
            
            f.write("### Common Path Suffixes\n\n")
            suffixes = bp.get("common_suffixes", {})
            f.write("| Sequence | Count |\n")
            f.write("|----------|-------|\n")
            for seq, count in list(suffixes.items())[:10]:
                f.write(f"| {seq} | {count} |\n")
            f.write("\n")
            
            f.write("### Decision Points\n\n")
            dp = bp.get("decision_points", {})
            f.write("| Step Type | Count |\n")
            f.write("|-----------|-------|\n")
            for step_type, count in sorted(dp.items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {step_type} | {count} |\n")
            f.write("\n")
            
            f.write("### Common Paths by Decision Type\n\n")
            paths_by_decision = bp.get("common_paths_by_decision", {})
            for decision, paths in sorted(paths_by_decision.items()):
                f.write(f"#### {decision}\n\n")
                f.write("| Sequence | Count |\n")
                f.write("|----------|-------|\n")
                for seq, count in list(paths.items())[:5]:
                    f.write(f"| {seq} | {count} |\n")
                f.write("\n")


def main():
    """Main function to generate report."""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the analysis scripts first.")
        return
    
    results = load_analysis_results(results_dir)
    
    if not results:
        print("No analysis results found. Please run the analysis scripts first.")
        return
    
    output_file = results_dir / "comprehensive_report.md"
    generate_report(results, output_file)
    
    print(f"Comprehensive report generated: {output_file}")


if __name__ == "__main__":
    main()

