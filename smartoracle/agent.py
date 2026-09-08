"""
ADK Triage Agent - Main entry point for the ADK agent system.
This follows the ADK directory structure requirements.
"""

import os
import sys
import json
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import ADK components
from google.adk.runners import LlmAgent, InvocationContext, RunConfig
from google.adk.sessions import InMemorySessionService
from google.adk.events import Event
import google.genai.types as genai_types

# Import our components
from .agents import (
    create_discrepancy_agent,
    create_spec_checker_agent,
    create_duplicate_analyzer_agent,
    create_false_positive_agent,
    create_orchestrator_agent,
    run_agent_async
)
from .tools import EngineRunner, SpecRetriever, make_terminal_tool, make_spec_tool, make_engines_tool, make_decision_tool
from .finding_loader import FindingLoader
from .bug_reporter import BugReporter
from .duplicate_checker import DuplicateChecker
from .logging_config import setup_logging

# Load environment variables
load_dotenv()

class TriageAgent:
    """Main ADK agent class that handles triage requests."""
    
    def __init__(self):
        # Setup logging
        logs_dir = Path(os.getenv("LOG_DIR", "logs"))
        setup_logging(logs_dir)
        
        # Initialize components
        self.reports_dir = Path(os.getenv("REPORTS_DIR", "reports"))
        self.bug_reporter = BugReporter(self.reports_dir)
        self.duplicate_checker = DuplicateChecker(self.reports_dir)
        
        # Create the main agent
        self.agent = self._create_agent()
    
    def _create_agent(self) -> LlmAgent:
        """Create the main ADK agent with sub-agents and tools."""
        # Get model configuration from environment
        orchestrator_model = os.getenv("TRIAGE_ORCHESTRATOR_MODEL", "gemini-2.0-flash")
        discrepancy_model = os.getenv("DISCREPANCY_AGENT_MODEL", "gemini-2.0-flash")
        spec_model = os.getenv("SPEC_CHECKER_AGENT_MODEL", "gemini-2.0-flash")
        duplicate_model = os.getenv("DUPLICATE_AGENT_MODEL", "gemini-2.0-flash")
        false_positive_model = os.getenv("FALSE_POSITIVE_AGENT_MODEL", "gemini-2.0-flash")
        
        # Get available engines from environment or default
        available_engines = os.getenv("AVAILABLE_ENGINES", "v8,spidermonkey,javascriptcore,graaljs").split(",")
        
        # Create tools
        engine_runner = EngineRunner()
        spec_cache_dir = Path(os.getenv("SPEC_CACHE_DIR", "spec_cache"))
        spec_retriever = SpecRetriever(spec_cache_dir)
        
        tools = [
            make_terminal_tool(engine_runner),
            make_spec_tool(spec_retriever),
            make_engines_tool(available_engines),
            make_decision_tool()
        ]
        
        # Create sub-agents
        sub_agents = [
            create_discrepancy_agent(discrepancy_model, tools, available_engines),
            create_spec_checker_agent(spec_model, tools, available_engines),
            create_duplicate_analyzer_agent(duplicate_model, tools, available_engines),
            create_false_positive_agent(false_positive_model, tools, available_engines),
        ]
        
        # Create orchestrator with sub-agents and tools
        agent = create_orchestrator_agent(orchestrator_model, sub_agents, tools)
        
        return agent
    
    def load_finding_from_case(self, case_dir: str) -> Dict[str, Any]:
        """Load a finding from a case directory."""
        case_path = Path(case_dir)
        if not case_path.exists():
            raise ValueError(f"Case directory not found: {case_dir}")
        
        loader = FindingLoader(case_path)
        finding = loader.load()
        
        # Check for duplicates
        duplicates = self.duplicate_checker.find_duplicates(
            finding.get("title", ""),
            finding.get("description", "")
        )
        finding["duplicates"] = duplicates
        
        return finding
    
    async def process_finding(self, finding_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process a single finding through the triage system.
        
        Args:
            finding_data: Dictionary containing finding information
            
        Returns:
            Dictionary with triage results
        """
        try:
            # Convert finding data to a user message
            user_message = f"Please analyze this JavaScript engine differential testing result:\n\n{json.dumps(finding_data, indent=2)}"
            
            # Use ADK to process the finding
            result = await run_agent_async(self.agent, user_message)
            
            # Try to parse the response as JSON
            try:
                return json.loads(result.get("text", "{}"))
            except json.JSONDecodeError:
                return {
                    "decision": "SKIP",
                    "confidence": 0.0,
                    "rationale": f"Could not parse agent response: {result.get('text', '')}"
                }
        except Exception as e:
            return {
                "error": str(e),
                "decision": "SKIP",
                "confidence": 0.0,
                "rationale": f"Processing failed: {e}"
            }
    
    def report_bug(self, finding_data: Dict[str, Any], triage_result: Dict[str, Any]) -> Dict[str, Any]:
        """Report a bug if the triage result indicates it should be reported."""
        if triage_result.get("decision") != "REPORT":
            return {"action": "skipped", "reason": "Not marked for reporting"}
        
        # Create report payload
        report_payload = {
            "title": finding_data.get("title", "Untitled Finding"),
            "description": finding_data.get("description", ""),
            "minimal_repro": finding_data.get("minimal_repro", ""),
            "engine_results": finding_data.get("engine_results", []),
            "divergence": finding_data.get("divergence", {}),
            "triage_result": triage_result,
            "duplicates": finding_data.get("duplicates", []),
            "summary": f"Confidence: {triage_result.get('confidence', 0.0)}, Rationale: {triage_result.get('rationale', '')}"
        }
        
        # Generate report
        report_result = self.bug_reporter.report(report_payload)
        return {"action": "reported", "report": report_result}

# Create the main agent instance
agent = TriageAgent()

# Export the agent for ADK - ADK expects 'root_agent'
root_agent = agent.agent

# # Export both for compatibility
__all__ = ['agent', 'root_agent']
