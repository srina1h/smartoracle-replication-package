"""
Shared utility functions for reasoning log analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict

EXCLUDED_TOOLS = {"transfer_to_agent", "test_case_minimizer"}


def is_failed_evaluation(session_data: Dict[str, Any]) -> bool:
    """
    Check if a session represents a failed evaluation.
    
    Returns True if:
    - final_decision is "ERROR"
    - Any step has a non-null error field
    - Missing critical fields (final_decision, steps)
    """
    # Check for ERROR decision
    final_decision = session_data.get("final_decision")
    if final_decision == "ERROR":
        return True
    
    # Check for missing critical fields
    if final_decision is None:
        return True
    
    # Check if steps exist and are valid
    steps = session_data.get("steps", [])
    if not isinstance(steps, list):
        return True
    
    # Check for errors in steps
    for step in steps:
        if step.get("error") is not None and step.get("error") != "":
            return True
    
    return False


def load_session_file(session_path: Path) -> Optional[Dict[str, Any]]:
    """Load and parse a session JSON file."""
    try:
        with open(session_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError, OSError) as e:
        print(f"Warning: Failed to load {session_path}: {e}")
        return None


def load_and_filter_logs(logs_dir: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Load all session JSON files from logs/sessions/ and filter out failed evaluations.
    
    Returns:
        Tuple of (valid_sessions, failed_sessions)
    """
    sessions_dir = logs_dir / "sessions"
    
    if not sessions_dir.exists():
        print(f"Warning: Sessions directory not found: {sessions_dir}")
        return [], []
    
    valid_sessions = []
    failed_sessions = []
    
    for session_file in sessions_dir.glob("*.json"):
        session_data = load_session_file(session_file)
        if session_data is None:
            failed_sessions.append({"path": str(session_file), "reason": "parse_error"})
            continue
        
        if is_failed_evaluation(session_data):
            failed_sessions.append({
                "path": str(session_file),
                "session_id": session_data.get("session_id"),
                "reason": "failed_evaluation"
            })
        else:
            valid_sessions.append(session_data)
    
    print(f"Loaded {len(valid_sessions)} valid sessions, filtered {len(failed_sessions)} failed evaluations")
    return valid_sessions, failed_sessions


def extract_steps(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Extract all steps from all sessions.
    
    Returns a flat list of all steps with session_id attached.
    """
    all_steps = []
    for session in sessions:
        session_id = session.get("session_id", "unknown")
        steps = session.get("steps", [])
        for step in steps:
            step_copy = step.copy()
            step_copy["session_id"] = session_id
            all_steps.append(step_copy)
    return all_steps


def get_step_type_sequence(session: Dict[str, Any]) -> List[str]:
    """Extract the sequence of step types from a session."""
    steps = session.get("steps", [])
    return [step.get("step_type", "unknown") for step in steps]


def get_tool_sequence(session: Dict[str, Any]) -> List[str]:
    """Extract the sequence of tool names from a session (skips excluded tools)."""
    steps = session.get("steps", [])
    return [
        step.get("tool_name")
        for step in steps
        if step.get("step_type") == "tool_call"
        and step.get("tool_name")
        and step.get("tool_name") not in EXCLUDED_TOOLS
    ]


def get_agent_sequence(session: Dict[str, Any]) -> List[str]:
    """Extract the sequence of agent names from a session."""
    steps = session.get("steps", [])
    return [
        step.get("agent_name") 
        for step in steps 
        if step.get("step_type") == "agent_call" and step.get("agent_name")
    ]


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """Compute basic statistics for a list of numeric values."""
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "median": 0.0
        }
    
    sorted_values = sorted(values)
    n = len(values)
    
    return {
        "count": n,
        "mean": sum(values) / n,
        "min": min(values),
        "max": max(values),
        "median": sorted_values[n // 2] if n > 0 else 0.0,
        "p25": sorted_values[n // 4] if n >= 4 else sorted_values[0],
        "p75": sorted_values[3 * n // 4] if n >= 4 else sorted_values[-1],
        "p90": sorted_values[9 * n // 10] if n >= 10 else sorted_values[-1],
        "p95": sorted_values[19 * n // 20] if n >= 20 else sorted_values[-1],
    }

