import os
import json
import re
import uuid
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import TypedDict, List, Dict, Any, Optional
from tools import search_tool
from dotenv import load_dotenv
load_dotenv()
from rich.console import Console
from rich.panel import Panel
console = Console()
from langgraph.graph import StateGraph, END
from prompts import (
    KNOWLEDGE_CHECK_PROMPT,
    PLANNER_PROMPT,
    GRADER_PROMPT,
    REFINER_PROMPT,
    EXTRACTOR_PROMPT,
    SYNTHESIZER_PROMPT,
    SELF_ANSWER_PROMPT
)
# ============== CONFIG ==============
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.7"))
MAX_SEARCH_RETRIES = int(os.getenv("MAX_SEARCH_RETRIES", "3"))
SEARCH_RESULTS_PER_QUERY = int(os.getenv("SEARCH_RESULTS_COUNT", "5"))
DEBUG_TRACE_ENABLED = os.getenv("DEBUG_TRACE_ENABLED", "1") != "0"
DEBUG_TRACE_DIR = os.getenv("DEBUG_TRACE_DIR", "debug_logs")
DEBUG_TEXT_LIMIT = int(os.getenv("DEBUG_TEXT_LIMIT", "8000"))
DEBUG_INCLUDE_PROMPT = os.getenv("DEBUG_INCLUDE_PROMPT", "1") != "0"
# ============== STATE SCHEMA ==============
class AgentState(TypedDict):
    """Agent State Schema"""
    # Planning
    original_question: str
    plan: List[str]
    current_step_index: int

    # Step results
    step_results: Dict[int, Dict[str, Any]]

    # Knowledge Check
    search_needed: bool
    current_search_query: str
    knowledge_check_result: Optional[Dict[str, Any]]

    # Search
    search_results_raw: List[Dict[str, Any]]
    filtered_results: List[Dict[str, Any]]
    should_retry_search: bool
    retry_reason: List[str]
    retry_count: int
    reflection_reason: str

    # Grader
    grader_result: Optional[Dict[str, Any]]

    # Final
    final_answer: str
    error: Optional[str]

    # Debug
    debug_trace_id: str
    debug_events: List[Dict[str, Any]]


# ============== LLM CALLS ==============
def _openai_config() -> Dict[str, str]:
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    api_key = os.getenv("OPENAI_API_KEY", "")
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    if not api_key:
        raise ValueError("Missing OPENAI_API_KEY")
    return {
        "base_url": base_url,
        "api_key": api_key,
        "model": model,
    }


def call_llm(prompt: str, temperature: float = 0.3) -> str:
    """Call OpenAI-compatible chat completions endpoint."""
    conf = _openai_config()
    payload = {
        "model": conf["model"],
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
    }
    req = urllib.request.Request(
        url=f"{conf['base_url']}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {conf['api_key']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            body = json.loads(resp.read().decode("utf-8"))
        content = body["choices"][0]["message"].get("content")
        return content if isinstance(content, str) else ""
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {e.code}: {error_body}") from e



def _extract_json_str(text: str) -> str:
    """Best-effort extraction of a JSON object/array from free-form text."""
    if not text:
        return ""
    stripped = text.strip()

    # Remove markdown code fence if present
    if "```" in stripped:
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)

    if stripped.startswith("{") or stripped.startswith("["):
        return stripped

    # Extract the first object/array block
    obj_match = re.search(r"\{[\s\S]*\}", stripped)
    arr_match = re.search(r"\[[\s\S]*\]", stripped)

    if obj_match and arr_match:
        return obj_match.group(0) if obj_match.start() < arr_match.start() else arr_match.group(0)
    if obj_match:
        return obj_match.group(0)
    if arr_match:
        return arr_match.group(0)
    return stripped


def render_prompt(template: str, **values: Any) -> str:
    """Render only known placeholders and keep other braces untouched."""
    rendered = template
    for key, val in values.items():
        rendered = rendered.replace("{" + key + "}", str(val))
    return rendered


def _clip_text(text: str, limit: int = DEBUG_TEXT_LIMIT) -> str:
    if not isinstance(text, str):
        text = str(text)
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} chars]"


def _to_jsonable(data: Any) -> Any:
    if data is None or isinstance(data, (int, float, bool, str)):
        if isinstance(data, str):
            return _clip_text(data)
        return data
    if isinstance(data, list):
        return [_to_jsonable(item) for item in data]
    if isinstance(data, dict):
        return {str(k): _to_jsonable(v) for k, v in data.items()}
    return _clip_text(str(data))


def log_event(state: AgentState, node: str, event: str, payload: Dict[str, Any]) -> None:
    if not DEBUG_TRACE_ENABLED:
        return
    record = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "node": node,
        "event": event,
        "step_index": state.get("current_step_index", 0),
        "payload": _to_jsonable(payload),
    }
    state.setdefault("debug_events", []).append(record)


def save_debug_log(state: AgentState) -> Optional[str]:
    if not DEBUG_TRACE_ENABLED:
        return None
    try:
        out_dir = Path(DEBUG_TRACE_DIR)
        out_dir.mkdir(parents=True, exist_ok=True)
        trace_id = state.get("debug_trace_id") or uuid.uuid4().hex[:12]
        out_path = out_dir / f"{trace_id}.json"
        data = {
            "trace_id": trace_id,
            "question": state.get("original_question", ""),
            "plan": state.get("plan", []),
            "step_results": state.get("step_results", {}),
            "final_answer": state.get("final_answer", ""),
            "error": state.get("error"),
            "event_count": len(state.get("debug_events", [])),
            "events": state.get("debug_events", []),
        }
        out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out_path)
    except Exception:
        return None


def call_llm_json_detailed(prompt: str) -> (Dict[str, Any], Dict[str, Any]):
    meta: Dict[str, Any] = {}
    if DEBUG_INCLUDE_PROMPT:
        meta["prompt"] = _clip_text(prompt)
    try:
        raw = call_llm(prompt)
        meta["raw_response"] = _clip_text(raw)
    except Exception as e:
        err = f"llm_call_failed: {e}"
        meta["error"] = err
        return {"error": err}, meta

    try:
        parsed = json.loads(_extract_json_str(raw))
        meta["parse_ok"] = True
        return parsed, meta
    except Exception as e:
        err = f"json_parse_failed: {e}"
        meta["parse_ok"] = False
        meta["error"] = err
        return {"error": err, "raw": raw or ""}, meta


def call_llm_json(prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON response"""
    parsed, _ = call_llm_json_detailed(prompt)
    return parsed


# ============== NODES ==============

def planner_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    prompt = render_prompt(PLANNER_PROMPT, question=question)
    result, llm_meta = call_llm_json_detailed(prompt)

    if "error" in result:
        plan = [question]
    else:
        plan = []
        if isinstance(result, dict):
            if "steps" in result:
                plan = result["steps"]
            elif "plan" in result:
                plan = result["plan"]
        if not plan:
            plan = [question]

    state["plan"] = plan
    state["current_step_index"] = 0
    state["step_results"] = {}

    log_event(
        state,
        "planner",
        "plan_built",
        {"question": question, "llm_meta": llm_meta, "llm_result": result, "plan": plan},
    )
    return state


def knowledge_check_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]
    step_results = state["step_results"]
    state["retry_count"] = 0
    state["search_results_raw"] = []
    state["filtered_results"] = []
    state["should_retry_search"] = False
    state["retry_reason"] = []

    step_question = plan[current_step] if current_step < len(plan) else question
    prompt = render_prompt(
        KNOWLEDGE_CHECK_PROMPT,
        question=step_question,
        step_results=step_results,
    )
    result, llm_meta = call_llm_json_detailed(prompt)

    if "error" in result or "mock" in result:
        state["search_needed"] = True
        state["current_search_query"] = step_question
        state["knowledge_check_result"] = {"decision": "NEED_SEARCH", "confidence": 0.5}
    else:
        decision = result.get("decision", "NEED_SEARCH")
        state["search_needed"] = decision == "NEED_SEARCH"
        state["knowledge_check_result"] = result
        state["current_search_query"] = result.get("search_keywords", step_question)

    log_event(
        state,
        "knowledge_check",
        "decision",
        {
            "step_question": step_question,
            "llm_meta": llm_meta,
            "llm_result": result,
            "search_needed": state["search_needed"],
            "current_search_query": state["current_search_query"],
        },
    )
    return state


def search_node(state: AgentState) -> AgentState:
    query = state.get("current_search_query", state["original_question"])
    current_step = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    step_question = plan[current_step] if current_step < len(plan) else state["original_question"]
    queries = query if isinstance(query, list) else [query]
    queries = [q.strip() for q in queries if isinstance(q, str) and q.strip()]

    results = []
    per_query_debug = []
    searcher = search_tool()
    for q in queries:
        current_results = searcher.search(
            q,
            num_results=SEARCH_RESULTS_PER_QUERY,
            context_query=step_question,
        )
        results.extend(current_results)
        per_query_debug.append(searcher.last_debug)
    
    # 结束mcp工具
    searcher.scraper_tool.stop()

    state["search_results_raw"] = results

    summary = []
    for item in results:
        summary.append(
            {
                "title": item.get("title", ""),
                "url": item.get("url", ""),
                "hop": item.get("hop", 1),
                "parent_url": item.get("parent_url", ""),
                "anchor_text": item.get("anchor_text", ""),
                "content_len": len(item.get("content", "") or ""),
                "content_preview": _clip_text(item.get("content", "") or "", 1200),
            }
        )

    log_event(
        state,
        "search",
        "search_complete",
        {
            "step_question": step_question,
            "queries": queries,
            "result_count": len(results),
            "results": summary,
            "per_query_debug": per_query_debug,
        },
    )
    return state


def grader_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]
    step_question = plan[current_step] if current_step < len(plan) else question
    search_results = state.get("search_results_raw", [])

    results_to_evaluate = []
    reject_reasons: List[str] = []
    if search_results:
        for i, item in enumerate(search_results):
            title = item.get("title", "")
            search_keyword = item.get("search keyword", "")
            content = (item.get("content", "") or "")[:500]
            results_to_evaluate.append(
                f"ID: {i}\nTitle: {title}\nContent: {content}\nSearch Keyword: {search_keyword}---"
            )
    else:
        results_to_evaluate = "No results"

    prompt = render_prompt(
        GRADER_PROMPT,
        question=step_question,
        search_results=results_to_evaluate,
    )
    result, llm_meta = call_llm_json_detailed(prompt)

    keep_ids = set()
    if isinstance(result, dict) and ("error" in result or "mock" in result):
        fallback_count = min(2, len(search_results))
        keep_ids = set(range(fallback_count))
        reject_reasons = [result.get("error", "grader_parse_error")]
    else:
        grader_items = result if isinstance(result, list) else []
        for item in grader_items:
            try:
                result_id = int(item.get("id", -1))
            except Exception:
                continue
            decision = item.get("decision", "").upper()
            if decision == "KEEP" and 0 <= result_id < len(search_results):
                keep_ids.add(result_id)
            elif decision == "REJECT":
                reason = str(item.get("reason", "")).strip()
                if reason:
                    reject_reasons.append(reason)

    state["filtered_results"] = [search_results[i] for i in keep_ids]
    min_valid_results = 1 if len(search_results) <= 2 else 2
    if len(state["filtered_results"]) < min_valid_results:
        state["should_retry_search"] = True
        state["retry_reason"] = reject_reasons or ["insufficient_relevant_results"]
    else:
        state["should_retry_search"] = False
        state["retry_reason"] = []

    log_event(
        state,
        "grader",
        "graded",
        {
            "step_question": step_question,
            "llm_meta": llm_meta,
            "llm_result": result,
            "input_result_count": len(search_results),
            "kept_result_count": len(state["filtered_results"]),
            "kept_urls": [x.get("url", "") for x in state["filtered_results"]],
            "should_retry_search": state["should_retry_search"],
            "retry_reason": state["retry_reason"],
        },
    )
    return state


def refiner_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]
    step_question = plan[current_step] if current_step < len(plan) else question

    original_query = state.get("current_search_query", step_question)
    failure_reason = state.get("retry_reason", ["Results not relevant"])
    prompt = render_prompt(
        REFINER_PROMPT,
        question=step_question,
        original_query=original_query,
        failure_reason=failure_reason,
    )
    result, llm_meta = call_llm_json_detailed(prompt)

    new_query = original_query
    if isinstance(result, dict) and "error" not in result:
        if "optimized_query" in result:
            new_query = result["optimized_query"]
        elif "raw" in result:
            new_query = result["raw"].strip()
    else:
        if isinstance(original_query, list):
            new_query = original_query + [step_question]
        else:
            new_query = [str(original_query), step_question]

    state["current_search_query"] = new_query
    state["retry_count"] = state.get("retry_count", 0) + 1

    log_event(
        state,
        "refiner",
        "query_refined",
        {
            "step_question": step_question,
            "llm_meta": llm_meta,
            "llm_result": result,
            "original_query": original_query,
            "new_query": new_query,
            "retry_count": state["retry_count"],
            "failure_reason": failure_reason,
        },
    )
    return state


def extractor_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]
    step_question = plan[current_step] if current_step < len(plan) else question
    search_results = state.get("filtered_results") or state.get("search_results_raw", [])

    results_text = "\n".join(
        [
            f"## {item.get('title', '')}\n{item.get('content', '')}\nSource: {item.get('url', '')}"
            for item in search_results
        ]
    )
    prompt = render_prompt(
        EXTRACTOR_PROMPT,
        question=step_question,
        search_results=results_text,
    )
    result, llm_meta = call_llm_json_detailed(prompt)

    extracted_content = ""
    if isinstance(result, dict):
        extracted_content = result.get("extracted_facts", result.get("raw", ""))
    elif isinstance(result, list):
        extracted_content = "\n".join([str(item) for item in result[:3]])
    else:
        extracted_content = str(result)

    if not extracted_content:
        fallback_lines = []
        for item in search_results[:3]:
            title = item.get("title", "")
            snippet = item.get("snippet", "")
            if title or snippet:
                fallback_lines.append(f"{title}: {snippet}".strip(": "))
        extracted_content = "\n".join(fallback_lines) or "No useful evidence extracted."

    state["step_results"][current_step] = {"content": extracted_content, "source_type": "search"}
    log_event(
        state,
        "extractor",
        "facts_extracted",
        {
            "step_question": step_question,
            "llm_meta": llm_meta,
            "llm_result": result,
            "source_count": len(search_results),
            "source_urls": [x.get("url", "") for x in search_results[:10]],
            "extracted_content": extracted_content,
        },
    )
    return state


def step_finalizer_node(state: AgentState) -> AgentState:
    current_step = state["current_step_index"]
    if not state.get("search_needed", True):
        answer = state.get("knowledge_check_result", {}).get("answer", "")
        confidence = state.get("knowledge_check_result", {}).get("confidence", 0.9)
        state["step_results"][current_step] = {
            "content": answer,
            "source_type": "internal",
            "confidence": confidence,
        }
    else:
        if current_step not in state.get("step_results", {}) and state.get("retry_count", 0) >= MAX_SEARCH_RETRIES:
            prompt = render_prompt(
                SELF_ANSWER_PROMPT,
                question=state["plan"][state["current_step_index"]],
                step_results=state["step_results"],
            )
            result, llm_meta = call_llm_json_detailed(prompt)
            state["step_results"][current_step] = {
                "content": result.get("answer", "No Results"),
                "source_type": "internal",
                "confidence": result.get("confidence", "0.5"),
            }
            log_event(
                state,
                "step_finalizer",
                "self_answer_fallback",
                {"llm_meta": llm_meta, "llm_result": result, "step_index": current_step},
            )

    log_event(
        state,
        "step_finalizer",
        "step_finalized",
        {
            "step_index": current_step,
            "step_result": state.get("step_results", {}).get(current_step, {}),
            "next_step_index": current_step + 1,
        },
    )
    state["current_step_index"] += 1
    return state


def synthesizer_node(state: AgentState) -> AgentState:
    question = state["original_question"]
    step_results = state.get("step_results", {})

    results_text = []
    for idx, item in sorted(step_results.items()):
        source = "Internal Knowledge" if item.get("source_type") == "internal" else "Search"
        results_text.append(f"Step {idx + 1} [{source}]: {item.get('content', '')[:200]}")

    prompt = render_prompt(
        SYNTHESIZER_PROMPT,
        question=question,
        step_results="\n\n".join(results_text),
    )
    result, llm_meta = call_llm_json_detailed(prompt)

    if isinstance(result, dict):
        final_answer = result.get("answer", result.get("raw", ""))
    else:
        final_answer = str(result)

    if not final_answer:
        final_answer = "\n\n".join(
            [f"**Step {idx + 1}**: {item.get('content', '')}" for idx, item in sorted(step_results.items())]
        )

    has_search = any(item.get("source_type") == "search" for item in step_results.values())
    if not has_search:
        final_answer += "\n\n---\n*Note: answer is based on internal knowledge.*"

    state["final_answer"] = final_answer
    log_event(
        state,
        "synthesizer",
        "answer_built",
        {
            "llm_meta": llm_meta,
            "llm_result": result,
            "step_results_count": len(step_results),
            "final_answer": final_answer,
        },
    )
    return state
def check_should_search(state: AgentState) -> str:
    """Condition: 判断是否需要搜索"""
    if state.get("search_needed", True):
        return "need_search"
    return "no_search"


def check_grader_pass(state: AgentState) -> str:
    """Condition: Grader 是否通过"""
    decision = state.get("should_retry_search", False)
    if decision == False:
        return "pass"
    return "fail"


def check_should_retry(state: AgentState) -> str:
    """Condition: 是否应该重试"""
    retry_count = state.get("retry_count", 0)
    if retry_count >= MAX_SEARCH_RETRIES:
        return "give_up"
    return "retry"


def check_more_steps(state: AgentState) -> str:
    """Condition: 是否还有更多步骤"""
    current = state.get("current_step_index", 0)
    plan = state.get("plan", [])
    if current >= len(plan):
        return "done"
    return "continue"


def increment_step(state: AgentState) -> AgentState:
    """Move to next step"""
    state["current_step_index"] = state.get("current_step_index", 0) + 1
    state["retry_count"] = 0
    return state


# ============== BUILD GRAPH ==============
def create_agent_graph() -> StateGraph:
    """创建 LangGraph 状态机"""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("planner", planner_node)
    graph.add_node("knowledge_check", knowledge_check_node)
    graph.add_node("search", search_node)
    graph.add_node("grader", grader_node)
    graph.add_node("refiner", refiner_node)
    graph.add_node("extractor", extractor_node)
    graph.add_node("step_finalizer", step_finalizer_node)
    graph.add_node("synthesizer", synthesizer_node)

    # Set entry point
    graph.set_entry_point("planner")

    # Add edges
    graph.add_edge("planner", "knowledge_check")

    # Knowledge Check conditional edge
    graph.add_conditional_edges(
        "knowledge_check",
        check_should_search,
        {
            "need_search": "search",
            "no_search": "step_finalizer"
        }
    )

    # Search flow with retry
    graph.add_edge("search", "grader")

    graph.add_conditional_edges(
        "grader",
        check_grader_pass,
        {
            "pass": "extractor",
            "fail": "refiner"
        }
    )

    graph.add_conditional_edges(
        "refiner",
        check_should_retry,
        {
            "retry": "search",
            "give_up": "step_finalizer"  # Fallback to internal knowledge
        }
    )

    graph.add_edge("extractor", "step_finalizer")

    # Step loop
    graph.add_conditional_edges(
        "step_finalizer",
        check_more_steps,
        {
            "continue": "knowledge_check",
            "done": "synthesizer"
        }
    )

    graph.add_edge("synthesizer", END)

    return graph

# ============== MAIN ==============
def run_agent(question: str):
    """运行 Agent"""
    trace_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:8]
    # Create initial state
    initial_state: AgentState = {
        "original_question": question,
        "plan": [],
        "current_step_index": 0,
        "step_results": {},
        "search_needed": False,
        "current_search_query": "",
        "knowledge_check_result": None,
        "search_results_raw": [],
        "filtered_results": [],
        "should_retry_search": False,
        "retry_reason": [],
        "retry_count": 0,
        "reflection_reason": "",
        "grader_result": None,
        "final_answer": "",
        "error": None,
        "debug_trace_id": trace_id,
        "debug_events": [],
    }

    # Build and run graph
    graph = create_agent_graph()
    app = graph.compile()

    console.print(Panel.fit(
        f"[bold]Hybrid Reasoning QA Agent[/bold]\nQuestion: {question}",
        border_style="cyan"
    ))

    try:
        log_event(initial_state, "run_agent", "start", {"trace_id": trace_id, "question": question})
        result = app.invoke(initial_state)
        result["debug_trace_id"] = trace_id
        log_event(
            result,
            "run_agent",
            "complete",
            {
                "trace_id": trace_id,
                "final_answer": result.get("final_answer", ""),
                "step_results_count": len(result.get("step_results", {})),
            },
        )
        log_path = save_debug_log(result)
        if log_path:
            console.print(f"[dim]Debug log saved: {log_path}[/dim]")
        return result["final_answer"]
    except Exception as e:
        initial_state["error"] = str(e)
        log_event(
            initial_state,
            "run_agent",
            "error",
            {"trace_id": trace_id, "error": str(e)},
        )
        log_path = save_debug_log(initial_state)
        if log_path:
            console.print(f"[dim]Debug log saved: {log_path}[/dim]")
        console.print(f"[red]Error:[/red] {e}")
        return f"抱歉，处理过程中出现错误: {str(e)}"


def main():
    """Main entry point"""
    # console.print("[bold cyan]🤖 Hybrid Reasoning QA Agent[/bold cyan]")
    # console.print("基于 LangGraph 的智能决策多跳问答系统\n")

    # Example usage
    question = input("请输入问题: ").strip()

    if not question:
        question = "什么是 LangGraph？它的主要特点是什么？"

    console.print()

    answer = run_agent(question)

    console.print("\n" + "=" * 50)
    console.print("[bold green]最终回答:[/bold green]")
    console.print("=" * 50)
    console.print(answer)


if __name__ == "__main__":
    main()
