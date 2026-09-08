"""
Enhanced reasoning logger for capturing orchestrator decision paths.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ReasoningStep:
    """Represents a single step in the orchestrator's reasoning process."""
    timestamp: str
    step_type: str  # "agent_call", "tool_call", "decision", "error"
    agent_name: Optional[str] = None
    tool_name: Optional[str] = None
    input_data: Optional[Dict[str, Any]] = None
    output_data: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None
    confidence: Optional[float] = None
    duration_ms: Optional[float] = None
    error: Optional[str] = None


@dataclass
class TriageSession:
    """Represents a complete triage session with all reasoning steps."""
    session_id: str
    test_id: str
    start_time: str
    end_time: Optional[str] = None
    final_decision: Optional[str] = None
    final_confidence: Optional[float] = None
    final_rationale: Optional[str] = None
    steps: List[ReasoningStep] = None
    total_duration_ms: Optional[float] = None
    
    def __post_init__(self):
        if self.steps is None:
            self.steps = []


class ReasoningLogger:
    """Enhanced logger for capturing orchestrator reasoning paths."""
    
    def __init__(self, logs_dir: Path):
        self.logs_dir = Path(logs_dir)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories for different log types
        self.reasoning_dir = self.logs_dir / "reasoning"
        self.timeline_dir = self.logs_dir / "timeline"
        self.sessions_dir = self.logs_dir / "sessions"
        
        for dir_path in [self.reasoning_dir, self.timeline_dir, self.sessions_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Setup file handlers
        self._setup_file_handlers()
    
    def _setup_file_handlers(self):
        """Setup file handlers for different log types."""
        # Reasoning trace handler
        reasoning_handler = logging.FileHandler(
            self.reasoning_dir / "reasoning_trace.log",
            mode='a'
        )
        reasoning_handler.setLevel(logging.INFO)
        reasoning_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        reasoning_handler.setFormatter(reasoning_formatter)
        
        # Add to root logger
        root_logger = logging.getLogger()
        root_logger.addHandler(reasoning_handler)
    
    def start_session(self, session_id: str, test_id: str) -> TriageSession:
        """Start a new triage session."""
        session = TriageSession(
            session_id=session_id,
            test_id=test_id,
            start_time=datetime.now().isoformat()
        )
        
        logger.info(f"🚀 Started triage session {session_id} for test {test_id}")
        return session
    
    def log_agent_call(self, session: TriageSession, agent_name: str, 
                      input_data: Dict[str, Any], output_data: Dict[str, Any],
                      duration_ms: float, reasoning: Optional[str] = None):
        """Log an agent call."""
        step = ReasoningStep(
            timestamp=datetime.now().isoformat(),
            step_type="agent_call",
            agent_name=agent_name,
            input_data=input_data,
            output_data=output_data,
            reasoning=reasoning,
            duration_ms=duration_ms
        )
        
        session.steps.append(step)
        
        logger.info(f"🤖 Agent call: {agent_name} (took {duration_ms:.1f}ms)")
        if reasoning:
            logger.info(f"   Reasoning: {reasoning}")
    
    def log_tool_call(self, session: TriageSession, tool_name: str,
                     input_data: Dict[str, Any], output_data: Dict[str, Any],
                     duration_ms: float):
        """Log a tool call."""
        step = ReasoningStep(
            timestamp=datetime.now().isoformat(),
            step_type="tool_call",
            tool_name=tool_name,
            input_data=input_data,
            output_data=output_data,
            duration_ms=duration_ms
        )
        
        session.steps.append(step)
        
        logger.info(f"🔧 Tool call: {tool_name} (took {duration_ms:.1f}ms)")
    
    def log_decision(self, session: TriageSession, decision: str, 
                    confidence: float, rationale: str):
        """Log a decision step."""
        step = ReasoningStep(
            timestamp=datetime.now().isoformat(),
            step_type="decision",
            reasoning=rationale,
            confidence=confidence
        )
        
        session.steps.append(step)
        
        session.final_decision = decision
        session.final_confidence = confidence
        session.final_rationale = rationale
        
        logger.info(f"🎯 Decision: {decision} (confidence: {confidence:.2f})")
        logger.info(f"   Rationale: {rationale}")
    
    def log_error(self, session: TriageSession, error: str, context: Optional[str] = None):
        """Log an error."""
        step = ReasoningStep(
            timestamp=datetime.now().isoformat(),
            step_type="error",
            error=error,
            reasoning=context
        )
        
        session.steps.append(step)
        
        logger.error(f"❌ Error: {error}")
        if context:
            logger.error(f"   Context: {context}")
    
    def end_session(self, session: TriageSession):
        """End a triage session and save logs."""
        session.end_time = datetime.now().isoformat()
        
        # Calculate total duration
        start_time = datetime.fromisoformat(session.start_time)
        end_time = datetime.fromisoformat(session.end_time)
        session.total_duration_ms = (end_time - start_time).total_seconds() * 1000
        
        # Save session data
        self._save_session(session)
        
        # Save timeline
        self._save_timeline(session)
        
        # Save reasoning trace
        self._save_reasoning_trace(session)
        
        logger.info(f"✅ Completed triage session {session.session_id} "
                   f"(took {session.total_duration_ms:.1f}ms)")
    
    def _save_session(self, session: TriageSession):
        """Save complete session data as JSON."""
        session_file = self.sessions_dir / f"{session.session_id}.json"
        
        with open(session_file, 'w') as f:
            json.dump(asdict(session), f, indent=2, default=str)
    
    def _save_timeline(self, session: TriageSession):
        """Save timeline as JSONL for easy processing."""
        timeline_file = self.timeline_dir / f"{session.session_id}_timeline.jsonl"
        
        with open(timeline_file, 'w') as f:
            for step in session.steps:
                timeline_entry = {
                    "timestamp": step.timestamp,
                    "step_type": step.step_type,
                    "agent_name": step.agent_name,
                    "tool_name": step.tool_name,
                    "duration_ms": step.duration_ms,
                    "reasoning": step.reasoning,
                    "confidence": step.confidence,
                    "error": step.error
                }
                f.write(json.dumps(timeline_entry) + '\n')
    
    def _save_reasoning_trace(self, session: TriageSession):
        """Save detailed reasoning trace as markdown."""
        trace_file = self.reasoning_dir / f"{session.session_id}_reasoning.md"
        
        with open(trace_file, 'w') as f:
            f.write(f"# Triage Reasoning Trace\n\n")
            f.write(f"**Session ID:** {session.session_id}\n")
            f.write(f"**Test ID:** {session.test_id}\n")
            f.write(f"**Start Time:** {session.start_time}\n")
            f.write(f"**End Time:** {session.end_time}\n")
            f.write(f"**Total Duration:** {session.total_duration_ms:.1f}ms\n\n")
            
            if session.final_decision:
                f.write(f"## Final Decision\n\n")
                f.write(f"**Decision:** {session.final_decision}\n")
                f.write(f"**Confidence:** {session.final_confidence:.2f}\n")
                f.write(f"**Rationale:** {session.final_rationale}\n\n")
            
            f.write(f"## Reasoning Steps\n\n")
            
            for i, step in enumerate(session.steps, 1):
                f.write(f"### Step {i}: {step.step_type.title()}\n\n")
                f.write(f"**Timestamp:** {step.timestamp}\n")
                
                if step.agent_name:
                    f.write(f"**Agent:** {step.agent_name}\n")
                if step.tool_name:
                    f.write(f"**Tool:** {step.tool_name}\n")
                if step.duration_ms:
                    f.write(f"**Duration:** {step.duration_ms:.1f}ms\n")
                if step.confidence:
                    f.write(f"**Confidence:** {step.confidence:.2f}\n")
                
                if step.reasoning:
                    f.write(f"**Reasoning:**\n{step.reasoning}\n")
                
                if step.error:
                    f.write(f"**Error:** {step.error}\n")
                
                if step.input_data:
                    f.write(f"**Input:**\n```json\n{json.dumps(step.input_data, indent=2)}\n```\n")
                
                if step.output_data:
                    f.write(f"**Output:**\n```json\n{json.dumps(step.output_data, indent=2)}\n```\n")
                
                f.write("\n---\n\n")
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get a summary of a session."""
        session_file = self.sessions_dir / f"{session_id}.json"
        
        if not session_file.exists():
            return None
        
        with open(session_file, 'r') as f:
            session_data = json.load(f)
        
        return {
            "session_id": session_data["session_id"],
            "test_id": session_data["test_id"],
            "duration_ms": session_data["total_duration_ms"],
            "final_decision": session_data["final_decision"],
            "final_confidence": session_data["final_confidence"],
            "step_count": len(session_data["steps"]),
            "agent_calls": len([s for s in session_data["steps"] if s["step_type"] == "agent_call"]),
            "tool_calls": len([s for s in session_data["steps"] if s["step_type"] == "tool_call"]),
            "errors": len([s for s in session_data["steps"] if s["step_type"] == "error"])
        }
