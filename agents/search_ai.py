import os
import json
from typing import TypedDict, List, Dict, Any, Optional
from rich.console import Console

from rich.markdown import Markdown
from rich import print as rprint
from rich.panel import Panel
# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from tools import search_tools

# Import prompts and tools
from prompts import (
    KNOWLEDGE_CHECK_PROMPT,
    PLANNER_PROMPT,
    GRADER_PROMPT,
    REFINER_PROMPT,
    EXTRACTOR_PROMPT,
    SYNTHESIZER_PROMPT
)

# Load environment
from dotenv import load_dotenv
load_dotenv()
console = Console()
# ============== MAIN ==============
def run_agent(question: str):
    """运行 Agent"""
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
        "retry_count": 0,
        "reflection_reason": "",
        "grader_result": None,
        "final_answer": "",
        "error": None
    }

    # Build and run graph
    graph = create_agent_graph()
    app = graph.compile()

    console.print(Panel.fit(
        f"[bold]Hybrid Reasoning QA Agent[/bold]\nQuestion: {question}",
        border_style="cyan"
    ))

    try:
        result = app.invoke(initial_state)
        return result["final_answer"]
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return f"抱歉，处理过程中出现错误: {str(e)}"
    
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
    retry_count: int
    reflection_reason: str

    # Grader
    grader_result: Optional[Dict[str, Any]]

    # Final
    final_answer: str
    error: Optional[str]


# ============== CONFIG ==============
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.9"))
MAX_SEARCH_RETRIES = int(os.getenv("MAX_SEARCH_RETRIES", "3"))

# ============== LLM CALLS ==============
def call_llm(prompt: str, temperature: float = 0.7) -> Dict[str, Any]:
    """Call LLM with prompt - using DashScope qwen3-max"""
    try:
        import dashscope
        dashscope.api_key = os.getenv("OPENAI_API_KEY")

        response = dashscope.Generation.call(
            model='qwen3-max',
            prompt=prompt,
            temperature=temperature,
            result_format='message'
        )
        if response.status_code == 200:
            content = response.output.choices[0].message.content
            # Try to parse JSON
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                return {"raw": content}
        else:
            return {"error": f"API error: {response.code}"}
    except Exception as e:
        # Fallback: mock response for development
        return {"mock": True, "error": str(e)}


def call_llm_json(prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON response"""
    result = call_llm(prompt)
    if "error" in result:
        return result
    if "raw" in result:
        try:
            return json.loads(result["raw"])
        except:
            return {"error": "Failed to parse JSON"}
    return result


# ============== NODES ==============

def planner_node(state: AgentState) -> AgentState:
    """Planner: 将复杂问题拆解为逻辑步骤列表"""
    question = state["original_question"]
    prompt = PLANNER_PROMPT.format(question=question)

    console.print(f"\n[bold cyan]📋 Planning:[/bold cyan] {question}")

    result = call_llm_json(prompt)

    if "error" in result:
        # Fallback: simple split
        plan = [question]
    else:
        # Parse plan from result
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

    console.print(f"[green]✓[/green] Planned {len(plan)} steps")
    return state


def knowledge_check_node(state: AgentState) -> AgentState:
    """Knowledge Check: 核心节点 - 判断是否需要搜索"""
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]

    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    prompt = KNOWLEDGE_CHECK_PROMPT.format(question=step_question)

    console.print(f"\n[bold yellow]🔍 Knowledge Check (Step {current_step + 1}):[/bold yellow] {step_question}")

    result = call_llm_json(prompt)

    if "error" in result or "mock" in result:
        # Fallback: assume search needed
        state["search_needed"] = True
        state["current_search_query"] = step_question
        state["knowledge_check_result"] = {"decision": "NEED_SEARCH", "confidence": 0.5}
        console.print(f"[yellow]⚠[/yellow] Using fallback: NEED_SEARCH")
    else:
        decision = result.get("decision", "NEED_SEARCH")
        confidence = result.get("confidence", 0.0)
        state["search_needed"] = decision == "NEED_SEARCH"
        state["knowledge_check_result"] = result
        state["current_search_query"] = result.get("search_keywords", step_question)

        if state["search_needed"]:
            console.print(f"[yellow]🔍[/yellow] Need search (confidence: {confidence:.0%})")
        else:
            answer = result.get("answer", "")
            console.print(f"[green]💡[/green] Internal Knowledge: {answer[:100]}...")

    return state


def search_node(state: AgentState) -> AgentState:
    """Search: 执行 IQS 搜索"""
    query = state.get("current_search_query", state["original_question"])

    console.print(f"\n[bold magenta]🔎 Searching:[/bold magenta] {query}")

    results = search_tool.search(query, num_results=5)

    if results:
        state["search_results_raw"] = results
        console.print(f"[green]✓[/green] Found {len(results)} results")
        for i, r in enumerate(results[:2], 1):
            console.print(f"  {i}. {r.get('title', 'Untitled')[:50]}")
    else:
        state["search_results_raw"] = []
        console.print(f"[red]✗[/red] No results found")

    return state


def grader_node(state: AgentState) -> AgentState:
    """Grader: 评估搜索结果相关性"""
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]

    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    search_results = state.get("search_results_raw", [])
    results_text = "\n".join([
        f"- {r.get('title', '')}: {r.get('content', '')}"
        for r in search_results
    ]) if search_results else "No results"

    prompt = GRADER_PROMPT.format(
        question=step_question,
        search_results=results_text
    )

    console.print(f"\n[bold cyan]📊 Grading results...")

    result = call_llm_json(prompt)

    if "error" in result or "mock" in result:
        # Fallback: assume pass if we have results
        passed = len(search_results) > 0
        state["grader_result"] = {"decision": "PASS" if passed else "FAIL", "relevance_score": 0.5}
    else:
        state["grader_result"] = result

    decision = state["grader_result"].get("decision", "FAIL")
    if decision == "PASS":
        console.print(f"[green]✅[/green] Grader: PASS (score: {state['grader_result'].get('relevance_score', 0):.0%})")
    else:
        console.print(f"[red]❌[/red] Grader: FAIL - {state['grader_result'].get('reason', 'Unknown reason')}")

    return state


def refiner_node(state: AgentState) -> AgentState:
    """Refiner: 优化搜索关键词"""
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]

    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    original_query = state.get("current_search_query", step_question)
    failure_reason = state.get("grader_result", {}).get("reason", "Results not relevant")

    prompt = REFINER_PROMPT.format(
        question=step_question,
        original_query=original_query,
        failure_reason=failure_reason
    )

    console.print(f"\n[yellow]🔄 Refining search query...")

    result = call_llm(prompt)

    new_query = original_query
    if isinstance(result, dict):
        if "optimized_query" in result:
            new_query = result["optimized_query"]
        elif "raw" in result:
            new_query = result["raw"].strip()

    state["current_search_query"] = new_query
    state["retry_count"] = state.get("retry_count", 0) + 1

    console.print(f"[yellow]→[/yellow] New query: {new_query}")
    console.print(f"[dim]Retry {state['retry_count']}/{MAX_SEARCH_RETRIES}[/dim]")

    return state


def extractor_node(state: AgentState) -> AgentState:
    """Extractor: 从搜索结果中提取关键事实"""
    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]

    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    search_results = state.get("search_results_raw", [])
    results_text = "\n".join([
        f"## {r.get('title', '')}\n{r.get('content', '')}\nSource: {r.get('url', '')}"
        for r in search_results
    ])

    prompt = EXTRACTOR_PROMPT.format(
        question=step_question,
        search_results=results_text
    )

    console.print(f"\n[bold cyan]📥 Extracting key facts...")

    result = call_llm(prompt)

    extracted_content = ""
    if isinstance(result, dict):
        extracted_content = result.get("extracted_facts", result.get("raw", ""))
    else:
        extracted_content = str(result)

    state["step_results"][current_step] = {
        "content": extracted_content,
        "source_type": "search",
        "confidence": state.get("grader_result", {}).get("relevance_score", 0.5)
    }

    console.print(f"[green]✅[/green] Extracted: {extracted_content[:100]}...")
    return state


def step_finalizer_node(state: AgentState) -> AgentState:
    """Step Finalizer: 格式化步骤结论"""
    current_step = state["current_step_index"]

    # Check if this step used internal knowledge or search
    if not state.get("search_needed", True):
        # Internal knowledge path
        answer = state.get("knowledge_check_result", {}).get("answer", "")
        confidence = state.get("knowledge_check_result", {}).get("confidence", 0.9)
        state["step_results"][current_step] = {
            "content": answer,
            "source_type": "internal",
            "confidence": confidence
        }
        console.print(f"[green]💡[/green] Step {current_step + 1} finalized (Internal Knowledge)")
    else:
        # Search path - content already set in extractor_node
        console.print(f"[green]🔍[/green] Step {current_step + 1} finalized (Search)")

    return state


def synthesizer_node(state: AgentState) -> AgentState:
    """Synthesizer: 生成最终答案"""
    question = state["original_question"]
    step_results = state.get("step_results", {})

    # Build step results summary
    results_text = []
    for idx, result in sorted(step_results.items()):
        source = "Internal Knowledge" if result.get("source_type") == "internal" else "Search"
        results_text.append(f"Step {idx + 1} [{source}]: {result.get('content', '')[:200]}")

    prompt = SYNTHESIZER_PROMPT.format(
        question=question,
        step_results="\n\n".join(results_text)
    )

    console.print(f"\n[bold cyan]🎯 Synthesizing final answer...")

    result = call_llm(prompt)

    final_answer = ""
    if isinstance(result, dict):
        final_answer = result.get("answer", result.get("raw", ""))
    else:
        final_answer = str(result)

    if not final_answer:
        # Fallback: combine all step results
        final_answer = "\n\n".join([
            f"**Step {idx + 1}**: {result.get('content', '')}"
            for idx, result in sorted(step_results.items())
        ])

    # Add disclaimer if any search failed
    has_search = any(r.get("source_type") == "search" for r in step_results.values())
    if not has_search:
        final_answer += "\n\n---\n*注：以上回答基于内部知识，如有需要可进行搜索验证。*"

    state["final_answer"] = final_answer

    console.print(f"[green]✓[/green] Final answer generated")
    return state


def check_should_search(state: AgentState) -> str:
    """Condition: 判断是否需要搜索"""
    if state.get("search_needed", True):
        return "need_search"
    return "no_search"


def check_grader_pass(state: AgentState) -> str:
    """Condition: Grader 是否通过"""
    decision = state.get("grader_result", {}).get("decision", "FAIL")
    if decision == "PASS":
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
    if current >= len(plan) - 1:
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
