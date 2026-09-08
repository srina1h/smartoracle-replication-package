"""ADK agents for triage system."""

import json
import logging
import os
import uuid
from typing import Any, Dict, List, Optional
from pathlib import Path

from google.adk.runners import LlmAgent
from google.adk.tools.agent_tool import AgentTool
from google.adk.sessions import InMemorySessionService
from google.adk.runners import InvocationContext, RunConfig
from google.adk.events import Event
from google.adk.models.lite_llm import LiteLlm
import google.genai.types as genai_types

from .prompts import (
    ORCHESTRATOR_SYSTEM,
    make_discrepancy_system,
    make_spec_checker_system,
    make_confidence_system,
    make_duplicate_system,
    make_false_positive_system,
    make_test_case_minimizer_system,
)

logger = logging.getLogger(__name__)


def _create_model(model_name: str):
    """Create the appropriate model object for the agent."""
    # Check if it's an OpenRouter model
    if model_name.startswith("openrouter/"):
        return LiteLlm(model=model_name)
    # Check if it's a Gemini model (default)
    elif model_name.startswith("gemini-"):
        return model_name
    # For other models, try LiteLLM
    else:
        return LiteLlm(model=model_name)


def _build_generate_config() -> genai_types.GenerateContentConfig:
    """Build Google GenAI GenerateContentConfig from environment defaults.

    Env vars supported:
    - GENAI_MAX_OUTPUT_TOKENS (int, default: 32768)
    - GENAI_TEMPERATURE (float, default: 0.2)
    - GENAI_AFC_DISABLE (bool as string, default: false)
    - GENAI_AFC_MAX_REMOTE_CALLS (int, optional)
    """
    max_tokens_str = os.getenv("GENAI_MAX_OUTPUT_TOKENS", "32768")  # Much higher default
    temperature_str = os.getenv("GENAI_TEMPERATURE", "0.2")
    afc_disable_str = os.getenv("GENAI_AFC_DISABLE", "false").lower()
    afc_max_calls_str = os.getenv("GENAI_AFC_MAX_REMOTE_CALLS")

    try:
        max_tokens = int(max_tokens_str)
    except Exception:
        max_tokens = 32768  # Much higher default

    try:
        temperature = float(temperature_str)
    except Exception:
        temperature = 0.2

    afc_disable = afc_disable_str in {"1", "true", "yes", "y"}
    automatic_function_calling: dict = {"disable": afc_disable}
    if afc_max_calls_str:
        try:
            automatic_function_calling["maximum_remote_calls"] = int(afc_max_calls_str)
        except Exception:
            pass
    else:
        # Default to 20 remote calls to accommodate sub-agent calls
        automatic_function_calling["maximum_remote_calls"] = 20

    return genai_types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_tokens,
        automatic_function_calling=automatic_function_calling,
    )


def create_discrepancy_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create discrepancy structurer agent."""
    return LlmAgent(
        name="discrepancy_structurer",
        model=_create_model(model),
        description="Extract and normalize behavioral differences across engines",
        instruction=make_discrepancy_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_spec_checker_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create spec checker agent."""
    return LlmAgent(
        name="spec_checker",
        model=_create_model(model),
        description="Map discrepancies to ECMA-262 normative steps",
        instruction=make_spec_checker_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_confidence_analyzer_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create confidence analyzer agent."""
    return LlmAgent(
        name="confidence_analyzer",
        model=_create_model(model),
        description="Evaluate reasoning quality and provide confidence scores",
        instruction=make_confidence_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_duplicate_analyzer_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create duplicate analyzer agent."""
    return LlmAgent(
        name="duplicate_analyzer",
        model=_create_model(model),
        description="Assess if finding duplicates prior reports",
        instruction=make_duplicate_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_false_positive_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create false positive critic agent."""
    return LlmAgent(
        name="false_positive_critic",
        model=_create_model(model),
        description="Critique potential false positives",
        instruction=make_false_positive_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_test_case_minimizer_agent(model: str, tools: List[Any], available_engines: List[str]) -> LlmAgent:
    """Create test case minimizer agent."""
    return LlmAgent(
        name="test_case_minimizer",
        model=_create_model(model),
        description="Create minimal reproducible examples from bug-reproducing code",
        instruction=make_test_case_minimizer_system(available_engines or []),
        tools=tools or [],
        generate_content_config=_build_generate_config(),
    )


def create_orchestrator_agent(model: str, sub_agents: List[LlmAgent], tools: List[Any]) -> LlmAgent:
    """Create orchestrator agent with sub-agents and tools."""
    # Wrap sub-agents as tools
    agent_tools = [AgentTool(agent=agent) for agent in sub_agents]
    
    return LlmAgent(
        name="triage_orchestrator",
        model=_create_model(model),
        description="Autonomous orchestrator for JavaScript engine triage",
        instruction=ORCHESTRATOR_SYSTEM,
        sub_agents=sub_agents,
        tools=tools + agent_tools,
        generate_content_config=_build_generate_config(),
    )


async def run_agent_async(agent: LlmAgent, user_text: str) -> Dict[str, Any]:
    """Run an ADK agent asynchronously and return result."""
    service = InMemorySessionService()
    session = await service.create_session(app_name="triage_agent", user_id="triager")

    ctx = InvocationContext(
        session_service=service,
        invocation_id=str(uuid.uuid4()),
        agent=agent,
        session=session,
        run_config=RunConfig(response_modalities=["text"]),
        user_content=genai_types.Content(role="user", parts=[{"text": user_text}]),
    )

    last_text = ""
    events = []
    async for event in agent.run_async(ctx):
        # Commit the event back to the session service (Runner's role)
        await service.append_event(session, event)
        
        # Capture all events for debugging
        events.append(event)

        try:
            content = getattr(event, "content", None)
            if content and hasattr(content, "parts"):
                for part in content.parts:
                    if isinstance(part, dict) and "text" in part:
                        last_text = part.get("text") or last_text
                    else:
                        last_text = str(part)
        except Exception:
            pass

    return {"text": last_text, "events": events}
