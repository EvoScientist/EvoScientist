"""Reflection tool for strategic decision-making."""

from langchain_core.tools import tool


@tool(parse_docstring=True)
def think_tool(reflection: str) -> str:
    """Tool for structured reflection and strategic decision-making.

    Use this tool to pause and reason carefully at any decision point — not just
    after searches, but before, during, and after any significant step. This creates
    a deliberate checkpoint for quality thinking.

    When to use:
    - Before starting work: What do I know? What skills and prior knowledge are available?
    - After obtaining results: What did I learn? Does this change the approach?
    - When choosing between options: What are the trade-offs? Which path is strongest?
    - When stuck or failing: What went wrong? Is there a proven strategy to apply?
    - Before concluding: Is the evidence sufficient? What does the next phase need from me?

    Your reflection should address the relevant dimensions below:

    1. Progress — What has been accomplished? What concrete steps remain?
    2. Evidence quality — Is the current evidence sufficient for the goal?
       Would a critical reviewer accept it, or are there gaps to fill?
    3. Skills leverage — Is there an installed skill that provides a structured
       workflow for what I'm doing? Check your available skills listing and read
       the relevant `SKILL.md` for full instructions. Skills cover various research
       phases — ideation, experiment execution, paper writing, review, and more.
       Follow a skill's workflow rather than improvising when one is available.
    4. Prior knowledge — Have I checked research memory before starting?
       `/memories/ideation-memory.md` records promising and failed research directions.
       `/memories/experiment-memory.md` records proven strategies from past cycles.
       Read these at the start of new work. After completing or failing a task,
       consider whether the outcome should be recorded back into memory.
       Skip this if the memory files do not exist yet.
    5. Strategy — Should I continue the current approach, adjust it, or try
       something different? What evidence supports this decision?
    6. Handoff — Is this phase complete? What artifacts and results does the
       next phase or the caller need? Am I leaving clear, well-organized outputs?
    7. Resource & compute — Before heavy operations (training, large evals),
       estimate runtime and memory. The sandbox has a 300s execution timeout
       and 100KB output limit. For tasks likely exceeding these, plan background
       execution with log files. After a timeout or OOM, reflect on whether to
       retry with reduced parameters (smaller model, fewer epochs, data subset)
       or switch to background execution.

    Not every reflection needs all seven dimensions. Pick the ones relevant to
    the current moment. A focused two or three dimension reflection is better
    than a shallow pass over all seven.

    Args:
        reflection: Your structured reflection addressing the relevant dimensions above

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"
