#!/usr/bin/env python3
"""Run trigger evaluation for a skill description.

Tests whether a skill's description causes an LLM to trigger (load the skill)
for a set of queries. Uses EvoSci's multi-provider LLM layer with tool calling
to simulate the agent's skill selection behavior.
"""

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

# Ensure skill-creator root is on sys.path for `from scripts.xxx` imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.utils import parse_skill_md


def _init_config():
    """Initialize EvoSci config and apply env vars (once per process)."""
    from EvoScientist.config import apply_config_to_env, get_effective_config

    config = get_effective_config()
    apply_config_to_env(config)
    return config


def run_single_query(
    query: str,
    skill_name: str,
    skill_description: str,
    model: str | None = None,
    provider: str | None = None,
) -> bool:
    """Run a single query and return whether the skill was triggered.

    Creates a minimal LLM invocation that simulates EvoSci's skill presentation:
    the LLM sees a system prompt with available skills and decides whether to
    call load_skill.
    """
    from langchain_core.messages import HumanMessage, SystemMessage
    from langchain_core.tools import tool

    from EvoScientist.llm import get_chat_model

    config = _init_config()

    effective_model = model or config.model
    effective_provider = provider or config.provider

    @tool
    def load_skill(name: str) -> str:
        """Load a skill by name to get detailed instructions for a specialized task."""
        return f"Skill {name} loaded."

    system_prompt = f"""You are a helpful AI assistant with access to specialized skills.

Your available skills:
- {skill_name}: {skill_description}

If a user's request matches a skill's purpose, call load_skill with the skill name to get detailed instructions.
If no skill is relevant, respond directly to the user without calling any tools."""

    # Disable thinking/reasoning for eval queries — we only need the tool call decision
    eval_kwargs = {}
    if effective_provider == "anthropic":
        eval_kwargs["thinking"] = {"type": "disabled"}
    elif effective_provider == "openai":
        # Don't pass reasoning kwarg — let the model use its default for simple tool calls
        pass

    try:
        chat_model = get_chat_model(
            model=effective_model,
            provider=effective_provider,
            **eval_kwargs,
        )
        model_with_tools = chat_model.bind_tools([load_skill])
        response = model_with_tools.invoke(
            [
                SystemMessage(content=system_prompt),
                HumanMessage(content=query),
            ]
        )

        # Check if the model called load_skill with the right skill name
        if hasattr(response, "tool_calls") and response.tool_calls:
            for tc in response.tool_calls:
                if tc["name"] == "load_skill":
                    called_name = tc["args"].get("name", "")
                    if skill_name in called_name or called_name in skill_name:
                        return True
        return False

    except Exception as e:
        # Fallback for providers that don't support tool calling:
        # Use a text-based approach
        try:
            chat_model = get_chat_model(
                model=effective_model,
                provider=effective_provider,
                **eval_kwargs,
            )
            fallback_prompt = f"""You are a helpful AI assistant with specialized skills available.

Your available skills:
- {skill_name}: {skill_description}

A user sends this request: "{query}"

Would you load the "{skill_name}" skill to help with this request?
Answer with ONLY "YES" or "NO"."""
            response = chat_model.invoke([HumanMessage(content=fallback_prompt)])
            text = (
                response.content
                if isinstance(response.content, str)
                else str(response.content)
            )
            return text.strip().upper().startswith("YES")
        except Exception:
            print(
                f"Warning: query failed for both tool-calling and fallback: {e}",
                file=sys.stderr,
            )
            return False


def run_eval(
    eval_set: list[dict],
    skill_name: str,
    description: str,
    num_workers: int,
    runs_per_query: int = 1,
    trigger_threshold: float = 0.5,
    model: str | None = None,
    provider: str | None = None,
) -> dict:
    """Run the full eval set and return results."""
    results = []

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        future_to_info = {}
        for item in eval_set:
            for run_idx in range(runs_per_query):
                future = executor.submit(
                    run_single_query,
                    item["query"],
                    skill_name,
                    description,
                    model,
                    provider,
                )
                future_to_info[future] = (item, run_idx)

        query_triggers: dict[str, list[bool]] = {}
        query_items: dict[str, dict] = {}
        for future in as_completed(future_to_info):
            item, _ = future_to_info[future]
            query = item["query"]
            query_items[query] = item
            if query not in query_triggers:
                query_triggers[query] = []
            try:
                query_triggers[query].append(future.result())
            except Exception as e:
                print(f"Warning: query failed: {e}", file=sys.stderr)
                query_triggers[query].append(False)

    for query, triggers in query_triggers.items():
        item = query_items[query]
        trigger_rate = sum(triggers) / len(triggers)
        should_trigger = item["should_trigger"]
        if should_trigger:
            did_pass = trigger_rate >= trigger_threshold
        else:
            did_pass = trigger_rate < trigger_threshold
        results.append(
            {
                "query": query,
                "should_trigger": should_trigger,
                "trigger_rate": trigger_rate,
                "triggers": sum(triggers),
                "runs": len(triggers),
                "pass": did_pass,
            }
        )

    passed = sum(1 for r in results if r["pass"])
    total = len(results)

    return {
        "skill_name": skill_name,
        "description": description,
        "results": results,
        "summary": {
            "total": total,
            "passed": passed,
            "failed": total - passed,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        description="Run trigger evaluation for a skill description"
    )
    parser.add_argument("--eval-set", required=True, help="Path to eval set JSON file")
    parser.add_argument("--skill-path", required=True, help="Path to skill directory")
    parser.add_argument(
        "--description", default=None, help="Override description to test"
    )
    parser.add_argument(
        "--num-workers", type=int, default=10, help="Number of parallel workers"
    )
    parser.add_argument(
        "--runs-per-query", type=int, default=3, help="Number of runs per query"
    )
    parser.add_argument(
        "--trigger-threshold", type=float, default=0.5, help="Trigger rate threshold"
    )
    parser.add_argument(
        "--model", default=None, help="Model to use (default: user's configured model)"
    )
    parser.add_argument(
        "--provider",
        default=None,
        help="LLM provider (default: user's configured provider)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="Print progress to stderr"
    )
    args = parser.parse_args()

    eval_set = json.loads(Path(args.eval_set).read_text())
    skill_path = Path(args.skill_path)

    if not (skill_path / "SKILL.md").exists():
        print(f"Error: No SKILL.md found at {skill_path}", file=sys.stderr)
        sys.exit(1)

    name, original_description, content = parse_skill_md(skill_path)
    description = args.description or original_description

    if args.verbose:
        print(f"Evaluating: {description}", file=sys.stderr)

    output = run_eval(
        eval_set=eval_set,
        skill_name=name,
        description=description,
        num_workers=args.num_workers,
        runs_per_query=args.runs_per_query,
        trigger_threshold=args.trigger_threshold,
        model=args.model,
        provider=args.provider,
    )

    if args.verbose:
        summary = output["summary"]
        print(
            f"Results: {summary['passed']}/{summary['total']} passed", file=sys.stderr
        )
        for r in output["results"]:
            status = "PASS" if r["pass"] else "FAIL"
            rate_str = f"{r['triggers']}/{r['runs']}"
            print(
                f"  [{status}] rate={rate_str} expected={r['should_trigger']}: {r['query'][:70]}",
                file=sys.stderr,
            )

    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
