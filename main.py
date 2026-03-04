import os
import json
import urllib.request
import urllib.error
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
MAX_SEARCH_RETRIES = int(os.getenv("MAX_SEARCH_RETRIES", "1"))
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
        return body["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"LLM HTTP error {e.code}: {error_body}") from e



def call_llm_json(prompt: str) -> Dict[str, Any]:
    """Call LLM and parse JSON response"""
    # json
    result = call_llm(prompt)
    
    # print(json.loads(result))
    # 输出为字典
    return json.loads(result)


# ============== NODES ==============

def planner_node(state: AgentState) -> AgentState:
    """Planner: 将复杂问题拆解为逻辑步骤列表"""
    question = state["original_question"]
    prompt = PLANNER_PROMPT.format(question=question)

    console.print(f"\n[bold cyan]📋 Planning:[/bold cyan] {question}")

    result = call_llm_json(prompt)
    print(f"planner result:{result}")

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
    step_results = state['step_results']
    state['retry_count'] = 0


    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    prompt = KNOWLEDGE_CHECK_PROMPT.format(question=step_question,
                                           step_results = step_results
                                           )

    console.print(f"\n[bold yellow]🔍 Knowledge Check (Step {current_step + 1}):[/bold yellow] {step_question}")

    result = call_llm_json(prompt)
    print(f"knowledge_check result:{result}")

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
    results = []
    for q in query:
        r = search_tool().search(q, num_results=3)
        results += r
    # results = search_tool().search(query, num_results=3)
    print(f"knowledge_check result:{results}")

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

    results_to_evaluate = []
    if search_results:
        for i, r in enumerate(search_results):
            title = r.get('title', '')
            search_keyword = r.get('search keyword', '')
            content = r.get('content', '')[:500] # 截断过长的内容
            results_to_evaluate.append(f"ID: {i}\nTitle: {title}\nContent: {content}\nSearch Keyword: {search_keyword}---")
    else:
        results_to_evaluate = "No results"

    # results_text = "\n".join([
    #     f"- {r.get('title', '')}: {r.get('content', '')}"
    #     for r in search_results
    # ]) if search_results else "No results"

    prompt = GRADER_PROMPT.format(
        question=step_question,
        search_results=results_to_evaluate
    )

    console.print(f"\n[bold cyan]📊 Grading results...")

    result = call_llm_json(prompt)
    print(f"Grader result:{result}")

    keep_ids = set()
    if "error" in result or "mock" in result:
        # 兜底策略：如果 LLM 调用失败，默认保留前2条（或者全部？根据需求）
        # 这里我们保守一点，保留前2条，避免流程卡死
        fallback_count = min(2, len(search_results))
        keep_ids = set(range(fallback_count))
        console.print(f"[yellow]⚠️ LLM Grader error, fallback: keeping first {fallback_count} results.[/yellow]")
    else:
        # 解析 LLM 返回的 JSON 列表
        for item in result:
            # 注意：这里假设返回的 id 是字符串数字，转换为 int
            result_id = int(item.get("id", -1))
            decision = item.get("decision", "").upper()
            if decision == "KEEP" and 0 <= result_id < len(search_results):
                keep_ids.add(result_id)

    # --- 关键修改点 4: 执行过滤 ---
    filtered_results = [search_results[i] for i in keep_ids]
    state["filtered_results"] += filtered_results
    

    # --- 关键修改点 5: 判断是否需要再次搜索 ---
    MIN_VALID_RESULTS = 5 # 定义最少需要的有效结果数量
    
    if len(state["filtered_results"]) < MIN_VALID_RESULTS:
        state["should_retry_search"] = True
        fail_reason = [result[i]['reason'] for i in keep_ids]
        state['retry_reason'] = fail_reason
        console.print(f"[red]❌[/red] Not enough valid results ({len(state['filtered_results'])} < {MIN_VALID_RESULTS}). Marked for retry.")
    else:
        state["should_retry_search"] = False
        console.print(f"[green]✅[/green] Filtered results: {len(state['filtered_results'])} valid results kept.")
    # if "error" in result or "mock" in result:
    #     # Fallback: assume pass if we have results
    #     passed = len(search_results) > 0
    #     state["grader_result"] = {"decision": "PASS" if passed else "FAIL", "relevance_score": 0.5}
    # else:
    #     state["grader_result"] = result

    # decision = state["grader_result"].get("decision", "FAIL")
    # if decision == "PASS":
    #     console.print(f"[green]✅[/green] Grader: PASS (score: {state['grader_result'].get('relevance_score', 0):.0%})")
    # else:
    #     console.print(f"[red]❌[/red] Grader: FAIL - {state['grader_result'].get('reason', 'Unknown reason')}")

    return state


def refiner_node(state: AgentState) -> AgentState:
    """Refiner: 优化搜索关键词"""
    if state['retry_count'] == 0:
        state["retry_count"] = state.get("retry_count", 0) + 1
        return state

    question = state["original_question"]
    current_step = state["current_step_index"]
    plan = state["plan"]

    if current_step < len(plan):
        step_question = plan[current_step]
    else:
        step_question = question

    original_query = state.get("current_search_query", step_question)
    failure_reason = state.get("retry_reason", ["Results not relevant"])

    prompt = REFINER_PROMPT.format(
        question=step_question,
        original_query=original_query,
        failure_reason=failure_reason
    )

    console.print(f"\n[yellow]🔄 Refining search query...")

    result = call_llm_json(prompt)
    print(f"Refiner result:{result}")

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
    print(f"Extractor result:{result}")

    extracted_content = ""
    if isinstance(result, dict):
        extracted_content = result.get("extracted_facts", result.get("raw", ""))
    else:
        extracted_content = str(result)

    state["step_results"][current_step] = {
        "content": extracted_content,
        "source_type": "search",
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
        if state['retry_count'] == 3:
            prompt = SELF_ANSWER_PROMPT.format(
                question=state["plan"][state["current_step_index"]],
                step_results=state['step_results']
            )
            result = call_llm_json(prompt)
            state["step_results"][current_step] = {
                "content": result.get("answer", 'No Results'),
                "source_type": "internal",
                "confidence": result.get("confidence", '0.5')
            }
        console.print(f"[green]🔍[/green] Step {current_step + 1} finalized (Search)")
    state["current_step_index"] += 1
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
    print(f"Synthesizer result:{result}")
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
        "filtered_results": [],
        "should_retry_search": False,
        "retry_reason": [],
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
