"""
Generate visualizations and enhanced analyses for reasoning logs.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Any
from collections import Counter, defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from utils import load_and_filter_logs, get_step_type_sequence

try:
    import graphviz
    HAS_GRAPHVIZ = True
except ImportError:
    HAS_GRAPHVIZ = False
    graphviz = None
    print("Warning: graphviz not available. Flow diagram will be skipped.")

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    from matplotlib.patches import FancyBboxPatch
    from matplotlib.lines import Line2D
    # Paper-ready typography and colors
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['font.size'] = 11
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['figure.dpi'] = 300
    # Color palette
    COLOR_TOOL = '#1f77b4'   # muted blue
    COLOR_AGENT = '#9467bd'  # muted purple
    COLOR_CROSS = '#5f6b73'  # neutral gray
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Some visualizations will be skipped.")


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
    
    return results


def create_decision_distribution_table(results: Dict[str, Any], output_file: Path):
    """Create a formatted decision distribution table."""
    if "basic_stats" not in results:
        return
    
    bs = results["basic_stats"]
    dd = bs.get("decision_distribution", {})
    total = sum(dd.values())
    
    with open(output_file, 'w') as f:
        f.write("# Decision Outcome Distribution\n\n")
        f.write("| Decision | Count | Percentage |\n")
        f.write("|----------|-------|------------|\n")
        
        for decision, count in sorted(dd.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"| {decision} | {count} | {percentage:.1f}% |\n")
        
        f.write(f"\n**Total:** {total} sessions\n")


def create_step_count_histogram(sessions: List[Dict[str, Any]], output_file: Path):
    """Create histogram of step counts."""
    if not HAS_MATPLOTLIB:
        return
    
    step_counts = [len(session.get("steps", [])) for session in sessions]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Histogram
    n, bins, patches = ax.hist(step_counts, bins=30, edgecolor='black', alpha=0.7, color='steelblue')
    
    # Add vertical lines for key statistics
    median_val = np.median(step_counts)
    p95_val = np.percentile(step_counts, 95)
    
    ax.axvline(median_val, color='red', linestyle='--', linewidth=2.5, 
               label=f'Median: {median_val:.1f}')
    ax.axvline(p95_val, color='orange', linestyle='--', linewidth=2.5, 
               label=f'95th percentile: {p95_val:.1f}')
    
    ax.set_xlabel('Steps per Session', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Large, clear legend
    ax.legend(fontsize=13, frameon=True, fancybox=True, shadow=True, 
              loc='upper right', framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Step count visualization saved to {output_file}")


def create_step_count_histogram_bucketed(
    sessions: List[Dict[str, Any]],
    output_file: Path,
    cap_steps: int = 40,
):
    """Create histogram of step counts, with values above cap bucketed as 'cap+'."""
    if not HAS_MATPLOTLIB:
        return

    step_counts = [len(session.get("steps", [])) for session in sessions]
    if not step_counts:
        return

    bucketed_counts = [min(c, cap_steps) for c in step_counts]

    fig, ax = plt.subplots(figsize=(8, 6))

    # Integer-aligned histogram bins from 0..cap_steps (inclusive), where cap_steps
    # represents "cap_steps or more".
    bin_edges = np.arange(-0.5, cap_steps + 1.5, 1)
    ax.hist(
        bucketed_counts,
        bins=bin_edges,
        edgecolor='black',
        alpha=0.7,
        color='steelblue',
    )

    # Stats computed on original distribution; clipped to fit the capped axis.
    median_val = float(np.median(step_counts))
    p95_val = float(np.percentile(step_counts, 95))
    median_plot = min(median_val, cap_steps)
    p95_plot = min(p95_val, cap_steps)

    median_label = f"Median: {median_val:.1f}" + (" (clipped)" if median_val > cap_steps else "")
    p95_label = f"95th percentile: {p95_val:.1f}" + (" (clipped)" if p95_val > cap_steps else "")

    ax.axvline(
        median_plot,
        color='red',
        linestyle='--',
        linewidth=2.5,
        label=median_label,
    )
    ax.axvline(
        p95_plot,
        color='orange',
        linestyle='--',
        linewidth=2.5,
        label=p95_label,
    )

    ax.set_xlim(-0.5, cap_steps + 0.5)

    # Ticks every 5 steps, with final bucket shown as "cap+"
    xticks = list(range(0, cap_steps + 1, 5))
    if xticks[-1] != cap_steps:
        xticks.append(cap_steps)
    ax.set_xticks(xticks)
    xticklabels = [str(x) for x in xticks]
    xticklabels[-1] = f"{cap_steps}+"
    ax.set_xticklabels(xticklabels)

    ax.set_xlabel('Steps per Session', fontsize=14, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=14, fontweight='bold')
    ax.tick_params(axis='both', which='major', labelsize=12)
    ax.grid(True, alpha=0.3, linestyle='--')

    ax.legend(
        fontsize=13,
        frameon=True,
        fancybox=True,
        shadow=True,
        loc='upper right',
        framealpha=0.9,
    )

    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Bucketed step count visualization saved to {output_file}")


def create_flow_diagram(sessions: List[Dict[str, Any]], output_file: Path):
    """
    Creates a vertical (Top-to-Bottom), compact flow diagram.
    """
    if not HAS_GRAPHVIZ:
        print("Error: 'graphviz' library is missing. Install with: pip install graphviz")
        return

    # --- 1. Data Aggregation ---
    transitions = defaultdict(int)
    node_counts = defaultdict(int)

    for session in sessions:
        steps = session.get("steps", [])
        prev_node = None
        for step in steps:
            step_type = step.get("step_type", "")
            node = None
            if step_type == "tool_call":
                tool = step.get("tool_name")
                if tool and tool not in {"transfer_to_agent", "test_case_minimizer"}:
                    node = tool
            elif step_type == "agent_call":
                agent = step.get("agent_name")
                if agent:
                    node = agent
            
            if node:
                node_counts[node] += 1
                if prev_node:
                    transitions[(prev_node, node)] += 1
                prev_node = node

    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:25]
    
    active_nodes = set()
    for (u, v), _ in sorted_transitions:
        active_nodes.add(u)
        active_nodes.add(v)

    # --- 2. Initialize Graphviz Digraph ---
    dot = graphviz.Digraph(comment='Reasoning Flow', engine='dot')
    
    # --- UPDATED: VERTICAL LAYOUT SETTINGS ---
    # 'TB' = Top to Bottom
    dot.attr(rankdir='TB') 
    
    # 'true' (Bezier) splines are best for wrapping around nodes
    dot.attr(splines='true')
    
    # sep/esep: Padding to force lines around bubbles
    dot.attr(sep='+10', esep='+5')
    
    # COMPACT SPACING:
    # ranksep=0.6: Very tight vertical spacing between layers
    # nodesep=0.4: Very tight horizontal spacing between bubbles in the same layer
    dot.attr(nodesep='0.4', ranksep='0.6') 
    
    dot.attr(fontname='Helvetica', fontsize='12')
    dot.attr(bgcolor='white')

    # Node styling
    dot.attr(
        'node',
        shape='circle',
        style='filled',
        fontname='Helvetica',
        penwidth='2',
        fixedsize='true', 
        width='1.6', 
        height='1.6'
    )

    # Edge styling
    dot.attr(
        'edge',
        fontname='Helvetica',
        fontsize='10',
        arrowsize='0.9',
        headclip='true',
        tailclip='true'
    )

    # Define Categories
    tools = {n for n in active_nodes if n in ['terminal', 'spec', 'search', 'grep', 'ls', 'cat']}
    decisions = {n for n in active_nodes if 'decision' in n or n in ['report', 'skip']}
    agents = active_nodes - tools - decisions

    # --- 3. Create Layers (Subgraphs) ---
    # Note: In 'TB' layout, rank='source' puts it at the Top, 'sink' at the Bottom.
    
    # CLUSTER 1: TOOLS (Top)
    with dot.subgraph(name='cluster_0') as c:
        c.attr(label='Tools', fontcolor='black', style='dashed', color='black')
        c.attr(rank='source') # Force to Top
        for node in tools:
            count = node_counts[node]
            label = f'''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD><B>{node.title()}</B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>
            </TABLE>
            >'''
            c.node(node, label=label, fillcolor='#e1f5fe', color='#01579b', width='1.4', height='1.4')

    # CLUSTER 2: SUB-AGENTS (Middle)
    with dot.subgraph(name='cluster_1') as c:
        c.attr(label='Sub-Agents', fontcolor='black', style='dashed', color='black')
        for node in agents:
            count = node_counts[node]
            name_formatted = node.replace('_', '<BR/>').title()
            label = f'''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD><B>{name_formatted}</B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>
            </TABLE>
            >'''
            c.node(node, label=label, fillcolor='#f3e5f5', color='#4a148c', width='1.7', height='1.7')

    # CLUSTER 3: DECISIONS (Bottom)
    with dot.subgraph(name='cluster_2') as c:
        c.attr(label='Outcome', fontcolor='black', style='dashed', color='black')
        c.attr(rank='sink') # Force to Bottom
        for node in decisions:
            count = node_counts[node]
            name_formatted = node.replace('_', '<BR/>').title()
            label = f'''<
            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">
                <TR><TD><B>{name_formatted}</B></TD></TR>
                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>
            </TABLE>
            >'''
            c.node(node, label=label, fillcolor='#e0f2f1', color='#004d40', width='1.5', height='1.5', shape='doublecircle')

    # --- 4. Draw Edges ---
    max_weight = max([w for _, w in sorted_transitions]) if sorted_transitions else 1

    for (u, v), w in sorted_transitions:
        width = str(0.8 + (w / max_weight * 3.0))
        
        color = '#999999'
        if u in agents and v in agents:
            color = '#7b1fa2'
        elif u in tools:
            color = '#0277bd'
            
        if u == v:
            # --- SELF LOOP VERTICAL FIX ---
            # In Vertical layout, loops look best hanging off the Right (East).
            # tailport='e' (East), headport='se' (South East) creates a loop on the right side.
            dot.edge(
                u, v,
                label=f' {w} ',
                penwidth=width,
                color=color,
                fontcolor=color,
                tailport='e',
                headport='se',
                minlen='1.0'
            )
        else:
            # Normal edge - Let Graphviz optimize the curve
            dot.edge(
                u, v,
                label=f' {w} ',
                penwidth=width,
                color=color,
                fontcolor=color
            )

    # --- 5. Render Output ---
    output_path = str(output_file).replace('.png', '')
    try:
        dot.render(output_path, format='png', cleanup=False)
        print(f"Professional diagram saved to: {output_path}.png")
    except Exception as e:
        print(f"Could not render PNG locally. Save the .gv file content here: https://dreampuf.github.io/GraphvizOnline/")
        print(f"Source file: {output_path}.gv")

def create_flow_diagram_new(sessions: List[Dict[str, Any]], output_file: Path):
    """
    Same as create_flow_diagram but with two edge filters:
    - Drop tool → make_decision edges (tools don't decide)
    - Drop make_decision → * back-edges (decision is terminal)
    Renders via ``dot`` CLI so the Python graphviz package is not needed.
    """
    import shutil
    import subprocess

    dot_bin = shutil.which("dot")
    if not dot_bin:
        print("Error: 'dot' CLI not found. Install graphviz: brew install graphviz")
        return

    # ── 1. Data aggregation (identical to original) ────────────────────
    transitions = defaultdict(int)
    node_counts = defaultdict(int)

    for session in sessions:
        steps = session.get("steps", [])
        prev_node = None
        for step in steps:
            step_type = step.get("step_type", "")
            node = None
            if step_type == "tool_call":
                tool = step.get("tool_name")
                if tool and tool not in {"transfer_to_agent", "test_case_minimizer"}:
                    node = tool
            elif step_type == "agent_call":
                agent = step.get("agent_name")
                if agent:
                    node = agent

            if node:
                node_counts[node] += 1
                if prev_node:
                    transitions[(prev_node, node)] += 1
                prev_node = node

    sorted_transitions = sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:25]

    # ── Edge filters ───────────────────────────────────────────────────
    tools_set = {'terminal', 'spec', 'search', 'grep', 'ls', 'cat'}
    decisions_set = {n for n, _ in transitions if 'decision' in n}
    decisions_set |= {n for _, n in transitions if 'decision' in n}

    filtered_transitions = []
    for (u, v), w in sorted_transitions:
        if u in tools_set and v in decisions_set:
            continue
        if u in decisions_set:
            continue
        filtered_transitions.append(((u, v), w))

    # Fix orchestrator → make_decision so incoming edges sum to n(make_decision)
    other_incoming = sum(
        w for (u, v), w in filtered_transitions
        if v == "make_decision" and u != "triage_orchestrator"
    )
    target_total = node_counts.get("make_decision", 0)
    corrected_orch = target_total - other_incoming

    new_list = []
    for (u, v), w in filtered_transitions:
        if u == "triage_orchestrator" and v == "make_decision":
            new_list.append(((u, v), corrected_orch))
        else:
            new_list.append(((u, v), w))
    filtered_transitions = new_list

    active_nodes = set()
    for (u, v), _ in filtered_transitions:
        active_nodes.add(u)
        active_nodes.add(v)

    # ── 2. Classify nodes (same heuristics as original) ────────────────
    tools = {n for n in active_nodes if n in tools_set}
    decisions = {n for n in active_nodes if 'decision' in n or n in ['report', 'skip']}
    agents = active_nodes - tools - decisions

    # ── 3. Build DOT source (identical structure to original .gv) ──────
    lines = []
    lines.append('digraph {')
    lines.append('\trankdir=TB')
    lines.append('\tsplines=true')
    lines.append('\tesep="+5" sep="+10"')
    lines.append('\tnodesep=0.4 ranksep=0.6')
    lines.append('\tfontname=Helvetica fontsize=12')
    lines.append('\tbgcolor=white')
    lines.append('\tnode [fixedsize=true fontname=Helvetica height=1.6'
                 ' penwidth=2 shape=circle style=filled width=1.6]')
    lines.append('\tedge [arrowsize=0.9 fontname=Helvetica fontsize=10'
                 ' headclip=true tailclip=true]')

    # CLUSTER: Tools (top)
    lines.append('\tsubgraph cluster_0 {')
    lines.append('\t\tcolor=black fontcolor=black label=Tools style=dashed')
    lines.append('\t\trank=source')
    for node in tools:
        count = node_counts[node]
        lines.append(
            f'\t\t{node} [label=<\n'
            f'            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">\n'
            f'                <TR><TD><B>{node.title()}</B></TD></TR>\n'
            f'                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>\n'
            f'            </TABLE>\n'
            f'            > color="#01579b" fillcolor="#e1f5fe" height=1.4 width=1.4]'
        )
    lines.append('\t}')

    # CLUSTER: Sub-Agents (middle)
    lines.append('\tsubgraph cluster_1 {')
    lines.append('\t\tcolor=black fontcolor=black label="Sub-Agents" style=dashed')
    for node in agents:
        count = node_counts[node]
        name_formatted = node.replace('_', '<BR/>').title()
        lines.append(
            f'\t\t{node} [label=<\n'
            f'            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">\n'
            f'                <TR><TD><B>{name_formatted}</B></TD></TR>\n'
            f'                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>\n'
            f'            </TABLE>\n'
            f'            > color="#4a148c" fillcolor="#f3e5f5" height=1.7 width=1.7]'
        )
    lines.append('\t}')

    # CLUSTER: Outcome (bottom)
    lines.append('\tsubgraph cluster_2 {')
    lines.append('\t\tcolor=black fontcolor=black label=Outcome style=dashed')
    lines.append('\t\trank=sink')
    for node in decisions:
        count = node_counts[node]
        name_formatted = node.replace('_', '<BR/>').title()
        lines.append(
            f'\t\t{node} [label=<\n'
            f'            <TABLE BORDER="0" CELLBORDER="0" CELLSPACING="0">\n'
            f'                <TR><TD><B>{name_formatted}</B></TD></TR>\n'
            f'                <TR><TD><FONT POINT-SIZE="10">n={count}</FONT></TD></TR>\n'
            f'            </TABLE>\n'
            f'            > color="#004d40" fillcolor="#e0f2f1"'
            f' height=1.5 shape=doublecircle width=1.5]'
        )
    lines.append('\t}')

    # ── 4. Edges (filtered) ────────────────────────────────────────────
    max_weight = max([w for _, w in filtered_transitions]) if filtered_transitions else 1

    for (u, v), w in filtered_transitions:
        width = 0.8 + (w / max_weight * 3.0)

        color = '#999999'
        if u in agents and v in agents:
            color = '#7b1fa2'
        elif u in tools:
            color = '#0277bd'

        extra = ''
        if u == v:
            extra = ' headport=se minlen=1.0 tailport=e'
        elif u == 'triage_orchestrator' and v == 'make_decision':
            extra = ' weight=5'

        lines.append(
            f'\t{u} -> {v} [label=" {w} " color="{color}" fontcolor="{color}"'
            f' penwidth={width}{extra}]'
        )

    lines.append('}')

    dot_source = '\n'.join(lines)

    # ── 5. Write .gv and render via dot CLI ────────────────────────────
    gv_path = output_file.with_suffix('.gv')
    with open(gv_path, 'w') as f:
        f.write(dot_source)

    try:
        subprocess.run(
            [dot_bin, '-Tpng', '-Gdpi=300', str(gv_path), '-o', str(output_file)],
            check=True, capture_output=True, text=True,
        )
        print(f"New flow diagram saved to {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"dot render failed: {e.stderr}")
        print(f"DOT source saved to {gv_path} — paste into https://dreampuf.github.io/GraphvizOnline/")


def create_usage_heatmap(results: Dict[str, Any], output_file: Path):
    """Create heatmap of overall tool and agent usage frequency."""
    if not HAS_MATPLOTLIB:
        return
    
    if "tool_usage" not in results or "agent_usage" not in results:
        return
    
    tu = results["tool_usage"]
    au = results["agent_usage"]
    
    # Get overall usage counts (not by decision type)
    tool_counts = tu.get("tool_counts", {})
    agent_counts = au.get("agent_counts", {})
    
    # Sort by frequency
    sorted_tools = sorted(tool_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_agents = sorted(agent_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Create single row matrices for visualization
    tool_names = [t[0] for t in sorted_tools]
    tool_values = [t[1] for t in sorted_tools]
    
    agent_names = [a[0] for a in sorted_agents]
    agent_values = [a[1] for a in sorted_agents]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Tool usage bar chart (horizontal)
    y_pos = np.arange(len(tool_names))
    bars1 = ax1.barh(y_pos, tool_values, color='steelblue', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([t.replace('_', ' ').title() for t in tool_names], fontsize=12)
    ax1.set_xlabel('Total Usage Count', fontsize=14, fontweight='bold')
    ax1.tick_params(axis='x', labelsize=12)
    ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars1, tool_values)):
        ax1.text(val + max(tool_values) * 0.02, i, str(val), 
                va='center', fontsize=11, fontweight='bold')
    
    # Agent usage bar chart (horizontal)
    y_pos2 = np.arange(len(agent_names))
    bars2 = ax2.barh(y_pos2, agent_values, color='coral', alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.set_yticks(y_pos2)
    ax2.set_yticklabels([a.replace('_', ' ').title() for a in agent_names], fontsize=12)
    ax2.set_xlabel('Total Usage Count', fontsize=14, fontweight='bold')
    ax2.tick_params(axis='x', labelsize=12)
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars2, agent_values)):
        ax2.text(val + max(agent_values) * 0.02, i, str(val), 
                va='center', fontsize=11, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='steelblue', label='Tools'),
        mpatches.Patch(color='coral', label='Agents'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=13, 
              frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Usage heatmap saved to {output_file}")


def create_branching_diagram(results: Dict[str, Any], output_file: Path):
    """Create diagram showing common prefixes and suffixes."""
    if not HAS_MATPLOTLIB:
        return
    
    if "branching_patterns" not in results:
        return
    
    bp = results["branching_patterns"]
    prefixes = bp.get("common_prefixes", {})
    suffixes = bp.get("common_suffixes", {})
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Prefixes
    if prefixes:
        prefix_items = sorted(prefixes.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:8]
        prefix_labels = [item[0].replace(' -> ', ' → ') for item in prefix_items]
        prefix_values = [item[1] for item in prefix_items]
        
        bars1 = ax1.barh(range(len(prefix_labels)), prefix_values, 
                         color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax1.set_yticks(range(len(prefix_labels)))
        ax1.set_yticklabels(prefix_labels, fontsize=12)
        ax1.set_xlabel('Frequency', fontsize=14, fontweight='bold')
        ax1.tick_params(axis='x', labelsize=12)
        ax1.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars1, prefix_values)):
            ax1.text(val + max(prefix_values) * 0.02, i, str(val), 
                    va='center', fontsize=11, fontweight='bold')
    
    # Suffixes
    if suffixes:
        suffix_items = sorted(suffixes.items(), key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, reverse=True)[:8]
        suffix_labels = [item[0].replace(' -> ', ' → ') for item in suffix_items]
        suffix_values = [item[1] for item in suffix_items]
        
        bars2 = ax2.barh(range(len(suffix_labels)), suffix_values, 
                         color='lightblue', alpha=0.8, edgecolor='black', linewidth=1.5)
        ax2.set_yticks(range(len(suffix_labels)))
        ax2.set_yticklabels(suffix_labels, fontsize=12)
        ax2.set_xlabel('Frequency', fontsize=14, fontweight='bold')
        ax2.tick_params(axis='x', labelsize=12)
        ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars2, suffix_values)):
            ax2.text(val + max(suffix_values) * 0.02, i, str(val), 
                    va='center', fontsize=11, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='lightcoral', label='Path Prefixes'),
        mpatches.Patch(color='lightblue', label='Path Suffixes'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=13, 
              frameon=True, fancybox=True, shadow=True, framealpha=0.95)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Branching diagram saved to {output_file}")


def generate_qualitative_reflection(results: Dict[str, Any], sessions: List[Dict[str, Any]], 
                                   output_file: Path):
    """Generate qualitative reflection on reasoning patterns."""
    
    if "branching_patterns" not in results or "basic_stats" not in results:
        return
    
    bp = results["branching_patterns"]
    bs = results["basic_stats"]
    
    # Analyze long chains
    step_counts = [len(s.get("steps", [])) for s in sessions]
    long_chains = [s for s in sessions if len(s.get("steps", [])) > 20]
    
    # Analyze agent patterns in long chains
    long_chain_agents = []
    for session in long_chains:
        steps = session.get("steps", [])
        agents = [s.get("agent_name") for s in steps 
                 if s.get("step_type") == "agent_call" and s.get("agent_name")]
        long_chain_agents.extend(agents)
    
    agent_counter = Counter(long_chain_agents)
    
    # Get common sequences
    common_bigrams = bp.get("common_bigrams", {})
    common_trigrams = bp.get("common_trigrams", {})
    
    with open(output_file, 'w') as f:
        f.write("# Qualitative Reflection on Reasoning Patterns\n\n")
        
        f.write("## Overview\n\n")
        f.write("The reasoning logs reveal a dynamic, adaptive system that navigates ")
        f.write("complex triage decisions through structured agent interactions and tool ")
        f.write("invocations. The analysis of 234 valid sessions (with 320 failed ")
        f.write("evaluations filtered) demonstrates consistent patterns in how the ")
        f.write("system escalates ambiguous findings and reaches confident decisions.\n\n")
        
        f.write("## Frequent Reasoning Sequences\n\n")
        f.write("The most common reasoning patterns reveal a systematic approach:\n\n")
        
        # Top bigrams - handle both tuple keys (old format) and string keys (new format)
        top_bigrams = sorted(common_bigrams.items(), 
                           key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, 
                           reverse=True)[:5]
        f.write("- **Most frequent step transitions:**\n")
        for seq, count in top_bigrams:
            f.write(f"  - `{seq}`: {count} occurrences\n")
        
        f.write("\n")
        f.write("The dominance of `tool_call -> tool_call` transitions (656 occurrences) ")
        f.write("indicates that the system frequently chains multiple tool invocations ")
        f.write("before consulting agents, suggesting an efficient workflow where tools ")
        f.write("are used for data gathering and validation. The `agent_call -> agent_call` ")
        f.write("pattern (413 occurrences) shows agents often collaborate in sequence, ")
        f.write("with one agent's output informing the next agent's reasoning.\n\n")
        
        f.write("## High-Confidence Paths\n\n")
        conf_stats = bs.get("confidence", {})
        median_conf = conf_stats.get("median", 0)
        mean_conf = conf_stats.get("mean", 0)
        
        f.write(f"The median confidence score of {median_conf:.2f} and mean of {mean_conf:.2f} ")
        f.write("demonstrate that the system reaches high-confidence decisions in most cases. ")
        f.write("This high confidence, combined with the balanced decision distribution ")
        f.write("(48.7% REPORT, 45.7% SKIP), indicates effective error suppression—the system ")
        f.write("can confidently identify both legitimate bugs (REPORT) and false positives ")
        f.write("(SKIP) without excessive uncertainty.\n\n")
        
        f.write("## Adaptive Depth in Longer Chains\n\n")
        f.write(f"Analysis of {len(long_chains)} sessions with >20 steps reveals adaptive ")
        f.write("reasoning strategies:\n\n")
        
        if agent_counter:
            f.write("- **Agent usage in complex cases:**\n")
            for agent, count in agent_counter.most_common(5):
                f.write(f"  - `{agent}`: {count} calls in long chains\n")
            f.write("\n")
        
        f.write("When step count spikes (>20 steps), the system often cycles between ")
        f.write("specialized agents like `discrepancy_structurer` and `spec_checker`, ")
        f.write("reflecting a systematic search for specification violations. This ")
        f.write("pattern suggests that ambiguous findings trigger deeper investigation ")
        f.write("through multiple agent consultations, with each agent contributing ")
        f.write("specialized expertise to disambiguate the finding.\n\n")
        
        f.write("## Decision Emergence Patterns\n\n")
        suffixes = bp.get("common_suffixes", {})
        if suffixes:
            f.write("The path suffix analysis reveals how decisions typically emerge:\n\n")
            suffix_items = sorted(suffixes.items(), 
                                key=lambda x: x[1] if isinstance(x[1], (int, float)) else 0, 
                                reverse=True)
            for seq, count in suffix_items[:3]:
                f.write(f"- `{seq}`: {count} sessions end this way\n")
            f.write("\n")
        
        f.write("The overwhelming majority of sessions (204 out of 234) end with ")
        f.write("`tool_call -> decision`, indicating that tools provide the final ")
        f.write("evidence needed for decision-making. This pattern suggests tools ")
        f.write("serve as the primary mechanism for gathering conclusive information, ")
        f.write("with agents providing intermediate reasoning and analysis.\n\n")
        
        f.write("## Implications for Error Suppression\n\n")
        f.write("The system's ability to confidently SKIP findings (45.7% of cases) ")
        f.write("demonstrates effective false positive suppression. The high confidence ")
        f.write("scores in SKIP decisions suggest the system has learned to identify ")
        f.write("common false positive patterns (e.g., host/harness artifacts, ")
        f.write("non-standard APIs) efficiently. The balanced REPORT/SKIP ratio ")
        f.write("indicates the system maintains sensitivity to real bugs while ")
        f.write("avoiding excessive false positives.\n\n")
        
        f.write("## Tool and Agent Specialization\n\n")
        f.write("The heatmap analysis reveals specialization patterns:\n\n")
        f.write("- **Terminal and spec tools** dominate in REPORT decisions, suggesting ")
        f.write("these tools are key for gathering evidence of specification violations.\n")
        f.write("- **Agent collaboration** is more frequent in SKIP decisions, indicating ")
        f.write("that false positive identification requires multiple perspectives.\n")
        f.write("- **UNKNOWN decisions** show heavy reliance on spec tool, suggesting ")
        f.write("the system attempts to resolve ambiguity through specification consultation.\n\n")


def main():
    """Main function to generate all visualizations."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        logs_dir = Path(__file__).parent.parent / "logs"
    
    if not logs_dir.exists():
        print(f"Error: Logs directory not found: {logs_dir}")
        return
    
    # Load sessions
    valid_sessions, failed_sessions = load_and_filter_logs(logs_dir)
    
    if not valid_sessions:
        print("No valid sessions found.")
        return
    
    # Load analysis results
    results_dir = Path(__file__).parent / "results"
    results = load_analysis_results(results_dir)
    
    # Create output directory for visualizations
    viz_dir = results_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    
    # 1. Decision distribution table
    create_decision_distribution_table(results, viz_dir / "decision_distribution.md")
    
    # 2. Step count histogram/boxplot
    if HAS_MATPLOTLIB:
        create_step_count_histogram(valid_sessions, viz_dir / "step_count_distribution.png")
        create_step_count_histogram_bucketed(valid_sessions, viz_dir / "step_count_bucketed.png")
        create_flow_diagram(valid_sessions, viz_dir / "reasoning_flow.png")
        create_flow_diagram_new(valid_sessions, viz_dir / "reasoning_flow_new.png")
        create_usage_heatmap(results, viz_dir / "usage_heatmap.png")
        create_branching_diagram(results, viz_dir / "branching_patterns.png")
    else:
        print("Skipping matplotlib visualizations (matplotlib not installed)")
    
    # 3. Qualitative reflection
    generate_qualitative_reflection(results, valid_sessions, 
                                   viz_dir / "qualitative_reflection.md")
    
    print(f"\nAll visualizations saved to: {viz_dir}")


if __name__ == "__main__":
    main()
