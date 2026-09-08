"""Simple tools for the ADK agent.

Enhancements:
- Load eligible engines from a local `paths.txt` (alongside this module)
- Only expose these engines for execution
- Provide a tool to list available engine names to the agent
"""

import logging
import shutil
import subprocess
import tempfile
import re
import signal
import os
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


@dataclass
class EngineResult:
    engine: str
    exit_code: int  # raw subprocess returncode (negative if terminated by signal)
    raw_return_code: int  # subprocess returncode (may be negative if signal)
    stdout: str
    stderr: str
    signal_num: Optional[int] = None
    signal_name: Optional[str] = None


@dataclass
class SpecSection:
    path: Path
    snippet: str
    score: float


class EngineRunner:
    """Simple engine runner for JavaScript execution with configurable engines."""
    
    def __init__(self, timeout_ms: int = 5000, paths_file: Optional[Path] = None, prefix_cmd: Optional[List[str]] = None):
        self.timeout_ms = timeout_ms
        self.paths_file = paths_file or (Path(__file__).parent / "paths.txt")
        self._engines: Dict[str, List[str]] = self._load_engines(self.paths_file)
        # Optional command prefix (e.g., GNU timeout wrapper)
        # If not provided, try env ENGINE_TIMEOUT_WRAPPER or default to GNU timeout used in test infra
        if prefix_cmd is not None:
            self._prefix_cmd = prefix_cmd
        else:
            env_prefix = os.getenv("ENGINE_TIMEOUT_WRAPPER", "timeout -s 9 -k 1 30").strip()
            tokens = shlex.split(env_prefix) if env_prefix else []
            if tokens and shutil.which(tokens[0]):
                self._prefix_cmd = tokens
            else:
                # If timeout is not available (e.g., macOS without coreutils), skip prefix
                if tokens:
                    logger.warning(f"Command prefix not found or unavailable: {tokens[0]}; running engines without wrapper")
                self._prefix_cmd = []

    def _load_engines(self, paths_file: Path) -> Dict[str, List[str]]:
        """Load engine commands from paths.txt if present.

        Supported keys in paths.txt (case sensitive to match prior datasets):
        - QJS=<path to quickjs binary>
        - XS=<path to Moddable XS binary>
        - V8=<path to d8/v8 binary>
        - graalJS=<path to GraalJS binary>
        - JSC=<path to JavaScriptCore binary>
        - CHAKRA=<path to Chakra/ChakraCore binary> (alias: CH)
        Returns mapping of canonical engine names to command argv lists.
        """
        mapping: Dict[str, List[str]] = {}
        try:
            if paths_file.exists():
                content = paths_file.read_text(encoding="utf-8").splitlines()
                for line in content:
                    line = line.strip()
                    if not line or "=" not in line:
                        continue
                    key, val = line.split("=", 1)
                    val = val.strip()
                    if not val:
                        continue
                    key_lower = key.strip()
                    if key_lower == "QJS":
                        mapping["quickjs"] = [val]
                    elif key_lower == "XS":
                        mapping["moddablexs"] = [val]
                    elif key_lower == "V8":
                        mapping["v8"] = [val]
                    elif key_lower == "graalJS":
                        mapping["graaljs"] = [val]
                    elif key_lower == "JSC":
                        mapping["javascriptcore"] = [val]
                    elif key_lower in ("CHAKRA", "CH"):
                        mapping["chakracore"] = [val]
                    elif key_lower == "SM":
                        mapping["spidermonkey"] = [val]
        except Exception as e:
            logger.warning(f"Failed to read engines from {paths_file}: {e}")

        # No fallbacks: if paths.txt is absent or empty, no engines are allowed
        return mapping

    def available_engines(self) -> List[str]:
        """Return the list of canonical engine names available for execution."""
        return sorted(self._engines.keys())

    def _resolve_command(self, engine: str) -> Optional[List[str]]:
        """Resolve engine name to command using configured engines only."""
        engine = engine.lower()
        return self._engines.get(engine)
    
    def _interpret_exit_code(self, returncode: int) -> int:
        """Interpret subprocess return code to match fuzzing tool conventions.
        
        This method converts subprocess return codes to match the exit codes
        that fuzzing tools typically report, ensuring consistency with
        existing fuzzing data.
        """
        if returncode >= 0:
            # Positive exit codes are passed through as-is
            return returncode
        
        # Handle negative exit codes (signal-based exits)
        signal_num = -returncode
        
        # Map common signals to fuzzing tool conventions
        signal_mapping = {
            signal.SIGSEGV: 133,    # Segmentation fault
            signal.SIGKILL: 137,    # Kill signal (often used for timeouts)
            signal.SIGABRT: 134,    # Abort signal
            signal.SIGTERM: 143,    # Termination signal
            signal.SIGINT: 130,     # Interrupt signal
            signal.SIGQUIT: 131,    # Quit signal
            signal.SIGILL: 132,     # Illegal instruction
            signal.SIGFPE: 136,     # Floating point exception
            signal.SIGBUS: 135,     # Bus error
        }
        
        if signal_num in signal_mapping:
            return signal_mapping[signal_num]
        else:
            # For unknown signals, use standard Unix convention: 128 + signal
            return 128 + signal_num
    
    def run_engine(self, engine: str, code: str) -> EngineResult:
        """Run JavaScript code on specified engine."""
        cmd = self._resolve_command(engine)
        if not cmd:
            return EngineResult(engine=engine, exit_code=-127, raw_return_code=-127, stdout="", stderr=f"{engine} not found")
        
        with tempfile.TemporaryDirectory(prefix=f"{engine}_run_") as tmp:
            script_path = Path(tmp) / "repro.js"
            script_path.write_text(code, encoding="utf-8")
            try:
                full_cmd = (self._prefix_cmd or []) + cmd + [str(script_path)]
                # When using external timeout wrapper, avoid short internal timeouts to preserve raw exit code semantics
                run_timeout = None if self._prefix_cmd else (self.timeout_ms / 1000.0)
                proc = subprocess.run(
                    full_cmd,
                    capture_output=True,
                    text=True,
                    timeout=run_timeout,
                )
                raw = proc.returncode
                sig_num: Optional[int] = None
                sig_name: Optional[str] = None
                if raw < 0:
                    sig_num = -raw
                    try:
                        sig_name = signal.Signals(sig_num).name
                    except Exception:
                        sig_name = None
                return EngineResult(
                    engine=engine,
                    exit_code=raw,
                    raw_return_code=raw,
                    stdout=proc.stdout,
                    stderr=proc.stderr,
                    signal_num=sig_num,
                    signal_name=sig_name,
                )
            except subprocess.TimeoutExpired:
                # Timeout typically results in SIGKILL (137)
                return EngineResult(
                    engine=engine,
                    exit_code=-signal.SIGKILL,
                    raw_return_code=-signal.SIGKILL,
                    stdout="",
                    stderr="timeout",
                    signal_num=signal.SIGKILL,
                    signal_name="SIGKILL",
                )


class SpecRetriever:
    """Simple ECMA spec retriever."""
    
    def __init__(self, spec_cache_dir: Path):
        self.spec_cache_dir = spec_cache_dir
    
    def _iter_spec_files(self):
        """Iterate over spec files."""
        if not self.spec_cache_dir.exists():
            return []
        exts = {".md", ".txt"}
        skip_patterns = {"vendor.txt", "requirements.txt", "package.json", "node_modules", "__pycache__", ".git"}
        for p in self.spec_cache_dir.rglob("*"):
            if (p.suffix.lower() in exts and p.is_file() and 
                not any(skip in str(p) for skip in skip_patterns)):
                yield p
    
    def _tokenize(self, text: str) -> set[str]:
        """Simple tokenization."""
        tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
        return set(tokens)
    
    def search(self, query: str, max_results: int = 5) -> List[SpecSection]:
        """Search spec cache for query."""
        q_tokens = self._tokenize(query)
        if not q_tokens:
            return []
        
        results: List[SpecSection] = []
        for path in self._iter_spec_files():
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            
            doc_tokens = self._tokenize(text)
            if not doc_tokens:
                continue
            
            overlap = q_tokens & doc_tokens
            if not overlap:
                continue
            
            score = len(overlap) / (len(q_tokens) ** 0.5 * len(doc_tokens) ** 0.5)
            first_token = next(iter(overlap))
            m = re.search(rf"(.{{0,200}}\b{re.escape(first_token)}\b.{{0,200}})", text, re.IGNORECASE | re.DOTALL)
            snippet = m.group(1) if m else text[:400]
            results.append(SpecSection(path=path, snippet=snippet, score=score))
        
        results.sort(key=lambda s: s.score, reverse=True)
        return results[:max_results]


# Tool functions for ADK
def make_terminal_tool(engine_runner: EngineRunner):
    """Create terminal tool for engine execution."""
    def terminal(engine_name: str, code: str) -> Dict[str, Any]:
        """Run JavaScript code on specified engine."""
        # Enforce eligible engines only
        if engine_name.lower() not in engine_runner.available_engines():
            return {
                "engine": engine_name,
                "exit_code": -127,
                "stdout": "",
                "stderr": f"engine '{engine_name}' not allowed. Available: {', '.join(engine_runner.available_engines())}"
            }
        result = engine_runner.run_engine(engine_name, code)
        return {
            "engine": result.engine,
            # For backward compatibility, keep exit_code as normalized
            "exit_code": result.exit_code,
            # Provide raw return code for exactness
            "raw_return_code": result.raw_return_code,
            # Provide signal metadata when applicable
            "signal_num": result.signal_num,
            "signal_name": result.signal_name,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    
    terminal.__name__ = "terminal"
    return terminal


def make_spec_tool(spec_retriever: SpecRetriever):
    """Create spec search tool."""
    def spec(query: str, top_k: int) -> Dict[str, Any]:
        """Search ECMA spec for query."""
        results = spec_retriever.search(query, top_k)
        return {
            "hits": [
                {
                    "path": str(r.path),
                    "snippet": r.snippet,
                    "score": r.score
                }
                for r in results
            ]
        }
    
    spec.__name__ = "spec"
    return spec


def make_engines_tool(engine_runner: EngineRunner):
    """Create a tool to list available engines for the agent."""
    def engines() -> Dict[str, Any]:
        return {"engines": engine_runner.available_engines()}

    engines.__name__ = "engines"
    return engines


def make_decision_tool():
    """Create a tool for making the final triage decision."""
    def make_decision(decision: str, confidence: float, rationale: str) -> Dict[str, Any]:
        """Make the final triage decision.
        
        Args:
            decision: Either "REPORT" or "SKIP"
            confidence: Confidence score between 0.0 and 1.0
            rationale: Clear explanation of the decision
            
        Returns:
            Dictionary with the decision details
        """
        if decision not in ["REPORT", "SKIP"]:
            return {"error": "Decision must be either 'REPORT' or 'SKIP'"}
        
        if not 0.0 <= confidence <= 1.0:
            return {"error": "Confidence must be between 0.0 and 1.0"}
        
        return {
            "decision": decision,
            "confidence": confidence,
            "rationale": rationale
        }
    
    make_decision.__name__ = "make_decision"
    return make_decision
