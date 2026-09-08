import os, json, time, sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from triage_agent_adk.tools import EngineRunner, SpecRetriever
from triage_agent_adk.duplicate_checker import DuplicateChecker
from triage_agent_adk.bug_reporter import BugReporter

import google.genai as genai
from google.genai import types as genai_types
from dotenv import load_dotenv

# Load .env from the baseline folder explicitly
_BASELINE_DIR = Path(__file__).parent
load_dotenv(_BASELINE_DIR / ".env")

# Fixed model for deterministic baseline
MODEL = "gemini-2.5-pro"
SPEC_CACHE_DIR = Path(os.getenv("SPEC_CACHE_DIR", "spec_cache"))
REPORTS_DIR = Path(os.getenv("REPORTS_DIR", "reports"))
TIMEOUT_MS = int(os.getenv("ENGINE_TIMEOUT_MS", "5000"))
INPUT_COST_PER_1K = float(os.getenv("INPUT_COST_PER_1K", "0"))
OUTPUT_COST_PER_1K = float(os.getenv("OUTPUT_COST_PER_1K", "0"))


def _genai_client():
    client = genai.Client()
    generate_cfg = genai_types.GenerateContentConfig(
        temperature=0.0,
    )
    return client, generate_cfg


def llm_json(client, generate_cfg, system: str, user: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    t0 = time.time()
    # Combine system and user into a single user message for Gemini
    combined_message = f"{system}\n\n{user}"
    resp = client.models.generate_content(
        model=MODEL,
        config=generate_cfg,
        contents=[
            {"role": "user", "parts": [{"text": combined_message}]},
        ],
    )
    t1 = time.time()
    text_parts = []
    try:
        if hasattr(resp, 'candidates') and resp.candidates and len(resp.candidates) > 0:
            candidate = resp.candidates[0]
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts:
                    for p in candidate.content.parts:
                        if hasattr(p, "text") and p.text:
                            text_parts.append(p.text)
    except Exception as e:
        print(f"DEBUG: Exception in parsing: {e}")
        pass
    text = "".join(text_parts)
    try:
        # Remove markdown code blocks if present
        if text.strip().startswith("```json"):
            # Find the JSON content between ```json and ```
            start = text.find("```json") + 7
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        elif text.strip().startswith("```"):
            # Handle generic code blocks
            start = text.find("```") + 3
            end = text.rfind("```")
            if end > start:
                text = text[start:end].strip()
        
        parsed = json.loads(text)
    except Exception as e:
        print(f"DEBUG: JSON parsing failed: {e}")
        print(f"DEBUG: Text was: {text[:200]}...")
        parsed = {"_raw": text}
    usage = getattr(resp, "usage_metadata", None)
    # Normalize token counts if present
    if usage:
        in_tok = getattr(usage, "input_tokens", 0) or getattr(usage, "input_token_count", 0) or getattr(usage, "prompt_token_count", 0) or 0
        out_tok = getattr(usage, "output_tokens", 0) or getattr(usage, "output_token_count", 0) or getattr(usage, "candidates_token_count", 0) or 0
    else:
        in_tok = out_tok = 0
    
    usage_norm = {
        "input_tokens": int(in_tok or 0),
        "output_tokens": int(out_tok or 0),
        "total_tokens": int((in_tok or 0) + (out_tok or 0)),
    }
    return parsed, {"latency_s": t1 - t0, "usage": usage_norm, "raw_text": text}


def identify_discrepancy(client, generate_cfg, finding: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are an expert in JavaScript engine behavior analysis. "
        "Extract and normalize the behavioral discrepancies observed across engines in the provided finding. "
        "Respond ONLY with strict JSON containing these keys: "
        "title (brief summary), description (detailed explanation), minimal_repro (minimal reproducible code), divergence (summary of divergent behaviors)."
    )
    user = json.dumps(finding, ensure_ascii=False, indent=2)
    return llm_json(client, generate_cfg, system, user)


def draft_spec_queries(client, generate_cfg, discrepancy: Dict[str, Any], original_finding: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are an ECMA-262 specification search assistant. "
        "Given the described JavaScript engine discrepancy and the original finding context, propose up to 3 precise and focused search queries to retrieve relevant specification sections. "
        "Respond ONLY with a JSON object containing key 'queries' which is a list of strings."
    )
    user = json.dumps({"discrepancy": discrepancy, "original_finding": original_finding}, ensure_ascii=False)
    return llm_json(client, generate_cfg, system, user)


def assess_need_for_clarifying_code(client, generate_cfg, discrepancy: Dict[str, Any], spec_snippets: List[Dict[str, Any]], engine_results: List[Dict[str, Any]], original_finding: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are a reasoning assistant evaluating whether additional clarifying code tests are needed to resolve uncertainty in a JavaScript engine discrepancy. "
        "Given the discrepancy, related ECMA-262 spec snippets, engine test results, and original finding context, "
        "decide if generating new targeted code snippets will improve decision confidence. "
        "Respond ONLY with JSON: {needs_clarifying_code: boolean, reasoning: string}."
    )
    user = json.dumps({"discrepancy": discrepancy, "spec_snippets": spec_snippets, "engine_results": engine_results, "original_finding": original_finding}, ensure_ascii=False)
    return llm_json(client, generate_cfg, system, user)


def generate_clarifying_code(client, generate_cfg, discrepancy: Dict[str, Any], spec_snippets: List[Dict[str, Any]], engine_results: List[Dict[str, Any]], original_finding: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are a JavaScript engineer expert tasked with generating 1-3 small, focused clarifying code snippets "
        "to interrogate the root cause of a discrepancy based on the provided information, ECMA-262 spec snippets, and engine test results. "
        "Each test should have a descriptive name, JS source code, and a brief purpose. "
        "Respond ONLY with JSON: {clarifying_tests: [{name: string, code: string, purpose: string}]}."
    )
    user = json.dumps({"discrepancy": discrepancy, "spec_snippets": spec_snippets, "engine_results": engine_results, "original_finding": original_finding}, ensure_ascii=False)
    return llm_json(client, generate_cfg, system, user)


def finalize_decision(client, generate_cfg, bundle: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    system = (
        "You are a bug triage expert. Given the complete context including the original finding, extracted discrepancy, "
        "retrieved spec snippets, original engine test results, and any clarifying test results, "
        "carefully decide whether to REPORT this finding as a valid bug or SKIP it. "
        "Include a numeric confidence score in [0,1] and a concise, well-reasoned rationale referencing all available evidence. "
        "Respond ONLY with JSON: {decision: 'REPORT'|'SKIP', confidence: float, rationale: string}."
    )
    user = json.dumps(bundle, ensure_ascii=False)
    return llm_json(client, generate_cfg, system, user)


def run_spec_search(spec: SpecRetriever, queries: List[str], top_k=5) -> List[Dict[str, Any]]:
    hits = []
    for q in (queries or [])[:3]:
        for h in spec.search(q, max_results=top_k):
            hits.append({"path": str(h.path), "snippet": h.snippet, "score": h.score, "query": q})
    return sorted(hits, key=lambda x: x["score"], reverse=True)[:10]


def run_engines(engine_runner: EngineRunner, code: str, engines: List[str]) -> List[Dict[str, Any]]:
    results = []
    for e in engines:
        r = engine_runner.run_engine(e, code)
        results.append({"engine": r.engine, "exit_code": r.exit_code, "stdout": r.stdout, "stderr": r.stderr})
    return results


def triage_case(case: Dict[str, Any]) -> Dict[str, Any]:
    client, generate_cfg = _genai_client()
    metrics = {"steps": []}
    start_time = time.time()

    disc, m1 = identify_discrepancy(client, generate_cfg, case)
    metrics["steps"].append({"step": "identify_discrepancy", **m1})

    spec_q, m2 = draft_spec_queries(client, generate_cfg, disc, case)
    metrics["steps"].append({"step": "draft_spec_queries", **m2})

    spec_hits = run_spec_search(SpecRetriever(SPEC_CACHE_DIR), spec_q.get("queries", []))

    runner = EngineRunner(timeout_ms=TIMEOUT_MS)
    engines = runner.available_engines()
    engine_results = []
    if engines and isinstance(disc.get("minimal_repro"), str) and disc["minimal_repro"].strip():
        t0 = time.time()
        engine_results = run_engines(runner, disc["minimal_repro"], engines)
        metrics["steps"].append({"step": "engine_exec", "latency_s": time.time() - t0, "engines": engines})
    else:
        metrics["steps"].append({"step": "engine_exec", "latency_s": 0, "engines": [], "note": "no engines or repro"})

    # Assess if clarifying code is needed
    need_assessment, m3 = assess_need_for_clarifying_code(client, generate_cfg, disc, spec_hits, engine_results, case)
    metrics["steps"].append({"step": "assess_need_for_clarifying_code", **m3})
    
    clarifying_results = []
    needs_clarifying = need_assessment.get("needs_clarifying_code", False)
    
    if needs_clarifying and engines:
        # Generate and run clarifying code
        clarifying_tests, m4 = generate_clarifying_code(client, generate_cfg, disc, spec_hits, engine_results, case)
        metrics["steps"].append({"step": "generate_clarifying_code", **m4})
        
        if clarifying_tests.get("clarifying_tests"):
            t0 = time.time()
            for test in clarifying_tests["clarifying_tests"]:
                if isinstance(test.get("code"), str) and test["code"].strip():
                    test_results = run_engines(runner, test["code"], engines)
                    clarifying_results.append({
                        "name": test.get("name", "unnamed"),
                        "purpose": test.get("purpose", ""),
                        "code": test["code"],
                        "results": test_results
                    })
            metrics["steps"].append({"step": "clarifying_exec", "latency_s": time.time() - t0, "tests_run": len(clarifying_results)})
        else:
            metrics["steps"].append({"step": "clarifying_exec", "latency_s": 0, "tests_run": 0, "note": "no tests generated"})
    else:
        metrics["steps"].append({"step": "generate_clarifying_code", "latency_s": 0, "note": "skipped - not needed"})
        metrics["steps"].append({"step": "clarifying_exec", "latency_s": 0, "tests_run": 0, "note": "skipped - not needed"})

    bundle = {
        "original_finding": case,
        "discrepancy": disc,
        "spec_snippets": spec_hits,
        "original_engine_results": engine_results,
        "clarifying_test_results": clarifying_results,
        "needs_clarifying_assessment": need_assessment,
        "divergence": case.get("divergence"),
    }
    decision, m5 = finalize_decision(client, generate_cfg, bundle)
    metrics["steps"].append({"step": "finalize_decision", **m5})

    # Aggregate tokens and cost
    total_in = sum((s.get("usage", {}).get("input_tokens", 0) for s in metrics["steps"]))
    total_out = sum((s.get("usage", {}).get("output_tokens", 0) for s in metrics["steps"]))
    total_tok = total_in + total_out
    input_cost = (total_in / 1000.0) * INPUT_COST_PER_1K if INPUT_COST_PER_1K else 0.0
    output_cost = (total_out / 1000.0) * OUTPUT_COST_PER_1K if OUTPUT_COST_PER_1K else 0.0
    total_elapsed = time.time() - start_time
    metrics["summary"] = {
        "input_tokens": total_in,
        "output_tokens": total_out,
        "total_tokens": total_tok,
        "input_cost_usd": round(input_cost, 6),
        "output_cost_usd": round(output_cost, 6),
        "total_cost_usd": round(input_cost + output_cost, 6),
        "total_elapsed_s": round(total_elapsed, 3),
    }

    out = {
        "decision": decision.get("decision", "SKIP"),
        "confidence": decision.get("confidence", 0.0),
        "rationale": decision.get("rationale", ""),
        "discrepancy": disc,
        "spec_snippets": spec_hits,
        "original_engine_results": engine_results,
        "clarifying_test_results": clarifying_results,
        "metrics": metrics,
    }
    return out


def run_and_maybe_report(case: Dict[str, Any]) -> Dict[str, Any]:
    result = triage_case(case)

    dup = DuplicateChecker(REPORTS_DIR)
    duplicates = dup.find_duplicates(
        title=case.get("title") or result["discrepancy"].get("title", ""),
        description=case.get("description") or result["discrepancy"].get("description", ""),
    )
    result["duplicates"] = duplicates

    if result.get("decision") != "REPORT":
        return {"action": "skipped", "result": result}

    reporter = BugReporter(REPORTS_DIR)
    payload = {
        "title": case.get("title", result["discrepancy"].get("title", "Untitled Finding")),
        "description": case.get("description", result["discrepancy"].get("description", "")),
        "minimal_repro": result["discrepancy"].get("minimal_repro", ""),
        "engine_results": result.get("engine_results", []),
        "divergence": case.get("divergence", {}),
        "triage_result": {"decision": result["decision"], "confidence": result["confidence"], "rationale": result["rationale"]},
        "duplicates": duplicates,
        "summary": f"Confidence: {result.get('confidence', 0.0)}, Rationale: {result.get('rationale', '')}",
        "metrics": result.get("metrics", {}),
    }
    rep = reporter.report(payload)
    return {"action": "reported", "report": rep, "result": result}


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--case", type=Path, required=True, help="Path to case JSON file")
    args = p.parse_args()
    case = json.loads(Path(args.case).read_text(encoding="utf-8"))
    out = run_and_maybe_report(case)
    print(json.dumps(out, ensure_ascii=False, indent=2))


