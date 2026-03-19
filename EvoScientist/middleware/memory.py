"""EvoScientist Memory Middleware.

Automatically extracts and persists long-term memory (user profile, research
preferences, experiment conclusions) from conversations.

Two mechanisms:
1. **Injection** (every LLM call): Reads ``/memory/MEMORY.md`` and appends it
   to the system prompt so the agent always has context.
2. **Extraction** (threshold-triggered): When the conversation exceeds a
   configurable message count, uses an LLM call to pull out structured facts
   and merges them into the appropriate MEMORY.md sections.

## Usage

```python
from EvoScientist.middleware import EvoMemoryMiddleware

middleware = EvoMemoryMiddleware(
    backend=my_backend,          # or backend factory
    memory_path="/memory/MEMORY.md",
    extraction_model=chat_model,
    trigger=("messages", 20),
)
agent = create_deep_agent(middleware=[middleware, ...])
```
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from contextvars import ContextVar
from typing import TYPE_CHECKING, Annotated, Any, NotRequired, cast

from langchain.agents.middleware.types import (
    AgentMiddleware,
    AgentState,
    ModelRequest,
    ModelResponse,
    PrivateStateAttr,
)
from langchain.tools import ToolRuntime
from langchain_core.messages import AnyMessage, HumanMessage, filter_messages
from langchain_core.runnables.config import RunnableConfig
from langgraph.runtime import Runtime
from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from deepagents.backends.protocol import BACKEND_TYPES, BackendProtocol
    from langchain.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

_CURRENT_MEMORY: ContextVar[str] = ContextVar("evo_memory_current", default="")
_STATE_MEMORY_KEY = "evo_memory_content"


class EvoMemoryState(AgentState):
    """State schema for EvoMemoryMiddleware."""

    evo_memory_content: NotRequired[Annotated[str, PrivateStateAttr]]


# ---------------------------------------------------------------------------
# Structured extraction schemas
# ---------------------------------------------------------------------------


class UserProfile(BaseModel):
    """Extracted user profile information."""

    name: str | None = Field(None, description="User's name")
    role: str | None = Field(None, description="User's role (e.g. researcher, student)")
    institution: str | None = Field(
        None, description="User's institution or organization"
    )
    language: str | None = Field(None, description="User's preferred language")


class ResearchPreferences(BaseModel):
    """Extracted research preference information."""

    primary_domain: str | None = Field(None, description="Primary research domain")
    sub_fields: str | None = Field(None, description="Research sub-fields")
    preferred_frameworks: str | None = Field(
        None, description="Preferred software frameworks"
    )
    preferred_models: str | None = Field(None, description="Preferred AI/ML models")
    hardware: str | None = Field(None, description="Available hardware (GPUs, etc.)")
    constraints: str | None = Field(None, description="Resource or time constraints")


class ExperimentConclusion(BaseModel):
    """Extracted experiment conclusion (only when a complete experiment was run)."""

    title: str = Field(description="Experiment name")
    question: str | None = Field(None, description="Research question")
    method: str | None = Field(None, description="Method summary")
    key_result: str | None = Field(None, description="Primary metric or outcome")
    conclusion: str | None = Field(None, description="One-line conclusion")
    artifacts: str | None = Field(None, description="Report path if any")


class ExtractedMemory(BaseModel):
    """Structured output schema for memory extraction.

    Only fields with genuinely new information should be populated.
    """

    user_profile: UserProfile | None = Field(
        None, description="New user profile information"
    )
    research_preferences: ResearchPreferences | None = Field(
        None, description="New research preferences"
    )
    experiment_conclusion: ExperimentConclusion | None = Field(
        None, description="Completed experiment conclusion"
    )
    learned_preferences: list[str] | None = Field(
        None, description="New preferences or habits observed"
    )


# ---------------------------------------------------------------------------
# Extraction prompt – sent to a (cheap) LLM to pull structured facts
# ---------------------------------------------------------------------------

EXTRACTION_PROMPT = """\
You are a memory extraction assistant for a scientific experiment agent called EvoScientist.

Analyze the following conversation and extract any NEW information that should be
remembered long-term. Only extract facts that are **not already present** in the
current memory shown below.

<current_memory>
{current_memory}
</current_memory>

<conversation>
{conversation}
</conversation>

Rules:
- Only populate fields with genuinely new information.
- Leave fields as null if there is nothing new.
- Do NOT repeat information already in <current_memory>.
- For experiment_conclusion, only include if a complete experiment was actually run.
- Be concise. Each value should be a short phrase, not a paragraph.
"""

# ---------------------------------------------------------------------------
# System-prompt snippet injected every turn
# ---------------------------------------------------------------------------

MEMORY_INJECTION_TEMPLATE = """<evo_memory>
{memory_content}
</evo_memory>

<memory_instructions>
The above <evo_memory> contains your long-term memory about the user and past experiments.
Use this to personalize your responses and avoid re-asking known information.

**When to update memory:**
- User shares their name, role, institution, or language
- User mentions their research domain, preferred frameworks, models, or hardware
- User explicitly asks you to remember something
- An experiment completes with notable conclusions

**How to update memory:**
- If `/memory/MEMORY.md` does not exist yet, use `write_file` to create it
- If it already exists, use `edit_file` to update specific sections
- Use this markdown structure:

```markdown
# EvoScientist Memory

## User Profile
- **Name**: ...
- **Role**: ...
- **Institution**: ...
- **Language**: ...

## Research Preferences
- **Primary Domain**: ...
- **Sub-fields**: ...
- **Preferred Frameworks**: ...
- **Preferred Models**: ...
- **Hardware**: ...
- **Constraints**: ...

## Experiment History
### [YYYY-MM-DD] Experiment Title
- **Question**: ...
- **Key Result**: ...
- **Conclusion**: ...

## Learned Preferences
- ...
```

**Priority:** Update memory IMMEDIATELY when the user provides personal or research
information — before composing your main response.
</memory_instructions>"""

DEFAULT_MEMORY_TEMPLATE = """# EvoScientist Memory

## User Profile
- **Name**: (unknown)
- **Role**: (unknown)
- **Institution**: (unknown)
- **Language**: (unknown)

## Research Preferences
- **Primary Domain**: (unknown)
- **Sub-fields**: (unknown)
- **Preferred Frameworks**: (unknown)
- **Preferred Models**: (unknown)
- **Hardware**: (unknown)
- **Constraints**: (unknown)

## Experiment History
(No experiments yet)

## Learned Preferences
- (none yet)
"""


def _get_thread_id(runtime: Runtime) -> str:
    try:
        config = cast("RunnableConfig", getattr(runtime, "config", {}))
        if isinstance(config, dict):
            thread_id = config.get("configurable", {}).get("thread_id")
            if thread_id is not None:
                return str(thread_id)
    except Exception:
        logger.debug("Failed to resolve thread_id from runtime config")
    return "default"


def _ensure_section(content: str, marker: str, body: str) -> str:
    if marker in content:
        return content
    content = content.rstrip()
    if content:
        content += "\n\n"
    return f"{content}{marker}\n{body.rstrip()}\n"


def _ensure_memory_template(existing_md: str) -> str:
    if not existing_md.strip():
        return DEFAULT_MEMORY_TEMPLATE

    result = existing_md
    if "# EvoScientist Memory" not in result:
        result = "# EvoScientist Memory\n\n" + result.lstrip()

    result = _ensure_section(
        result,
        "## User Profile",
        "\n".join(
            [
                "- **Name**: (unknown)",
                "- **Role**: (unknown)",
                "- **Institution**: (unknown)",
                "- **Language**: (unknown)",
            ],
        ),
    )
    result = _ensure_section(
        result,
        "## Research Preferences",
        "\n".join(
            [
                "- **Primary Domain**: (unknown)",
                "- **Sub-fields**: (unknown)",
                "- **Preferred Frameworks**: (unknown)",
                "- **Preferred Models**: (unknown)",
                "- **Hardware**: (unknown)",
                "- **Constraints**: (unknown)",
            ],
        ),
    )
    result = _ensure_section(result, "## Experiment History", "(No experiments yet)")
    result = _ensure_section(result, "## Learned Preferences", "- (none yet)")
    return result


def _section_bounds(content: str, marker: str) -> tuple[int | None, int | None]:
    idx = content.find(marker)
    if idx == -1:
        return None, None
    start = idx + len(marker)
    next_marker = content.find("\n## ", start)
    if next_marker == -1:
        next_marker = len(content)
    return start, next_marker


def _normalize_item(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


# ---------------------------------------------------------------------------
# Helper: merge extracted JSON into MEMORY.md markdown
# ---------------------------------------------------------------------------


def _merge_memory(existing_md: str, extracted: dict[str, Any]) -> str:
    """Merge extracted fields into the existing MEMORY.md content.

    Performs targeted replacements within the known sections.  Unknown
    sections or empty extractions are left untouched.
    """
    if not extracted:
        return existing_md

    result = _ensure_memory_template(existing_md)

    # --- User Profile ---
    profile = extracted.get("user_profile")
    if profile and isinstance(profile, dict):
        field_map = {
            "name": "Name",
            "role": "Role",
            "institution": "Institution",
            "language": "Language",
        }
        for key, label in field_map.items():
            value = profile.get(key)
            if value and value != "null":
                # Replace the line  "- **Label**: ..." with new value
                pattern = rf"(- \*\*{label}\*\*: ).*"
                result = re.sub(pattern, lambda m, v=value: m.group(1) + v, result)

    # --- Research Preferences ---
    prefs = extracted.get("research_preferences")
    if prefs and isinstance(prefs, dict):
        field_map = {
            "primary_domain": "Primary Domain",
            "sub_fields": "Sub-fields",
            "preferred_frameworks": "Preferred Frameworks",
            "preferred_models": "Preferred Models",
            "hardware": "Hardware",
            "constraints": "Constraints",
        }
        for key, label in field_map.items():
            value = prefs.get(key)
            if value and value != "null":
                pattern = rf"(- \*\*{label}\*\*: ).*"
                result = re.sub(pattern, lambda m, v=value: m.group(1) + v, result)

    # --- Experiment History (append) ---
    exp = extracted.get("experiment_conclusion")
    should_add_exp = bool(exp and isinstance(exp, dict) and exp.get("title"))
    if should_add_exp:
        from datetime import datetime

        date_str = datetime.now().strftime("%Y-%m-%d")
        title = str(exp.get("title", "Untitled")).strip()
        entry = f"\n### [{date_str}] {title}\n"
        entry += f"- **Question**: {exp.get('question', 'N/A')}\n"
        entry += f"- **Method**: {exp.get('method', 'N/A')}\n"
        entry += f"- **Key Result**: {exp.get('key_result', 'N/A')}\n"
        entry += f"- **Conclusion**: {exp.get('conclusion', 'N/A')}\n"
        if exp.get("artifacts"):
            entry += f"- **Artifacts**: {exp['artifacts']}\n"

        # Remove placeholder if present
        exp_start, exp_end = _section_bounds(result, "## Experiment History")
        if exp_start is not None and exp_end is not None:
            exp_section = result[exp_start:exp_end]
            exp_lines = [
                line
                for line in exp_section.splitlines()
                if "(No experiments yet)" not in line
            ]
            result = (
                result[:exp_start]
                + "\n"
                + "\n".join(exp_lines).strip("\n")
                + "\n"
                + result[exp_end:]
            )

        # De-duplicate by title if already present
        if re.search(rf"### \[[0-9-]+\] {re.escape(title)}\b", result):
            should_add_exp = False

    if should_add_exp and exp and isinstance(exp, dict) and exp.get("title"):
        # Insert before "## Learned Preferences"
        marker = "## Learned Preferences"
        if marker in result:
            result = result.replace(marker, entry + "\n" + marker)
        else:
            # Fallback: append at end
            result = result.rstrip() + "\n" + entry

    # --- Learned Preferences (append) ---
    learned = extracted.get("learned_preferences")
    if learned and isinstance(learned, list):
        marker = "## Learned Preferences"
        start, end = _section_bounds(result, marker)
        if start is None or end is None:
            result = _ensure_section(result, marker, "- (none yet)")
            start, end = _section_bounds(result, marker)

        if start is not None and end is not None:
            section = result[start:end]
            section_lines = [
                line
                for line in section.splitlines()
                if line.strip() and line.strip() not in {"- (none yet)", "- (none)"}
            ]
            existing_items = {
                _normalize_item(line[2:])
                for line in section_lines
                if line.strip().startswith("- ")
            }
            new_lines = []
            for item in learned:
                if not item:
                    continue
                normalized = _normalize_item(str(item))
                if normalized in existing_items:
                    continue
                existing_items.add(normalized)
                new_lines.append(f"- {item}")

            if new_lines:
                section_lines.extend(new_lines)
                rebuilt = "\n" + "\n".join(section_lines).strip("\n") + "\n"
                result = result[:start] + rebuilt + result[end:]

    return result


# ---------------------------------------------------------------------------
# Middleware
# ---------------------------------------------------------------------------


class EvoMemoryMiddleware(AgentMiddleware):
    """Middleware that injects and auto-extracts long-term memory.

    Args:
        backend: Backend instance or factory for reading/writing memory files.
        memory_path: Virtual path to MEMORY.md (default ``/memory/MEMORY.md``).
        extraction_model: Chat model used for extraction (can be a cheap/fast
            model like ``claude-haiku``). If ``None``, automatic extraction is
            disabled and only prompt injection + manual ``edit_file`` works.
        trigger: When to run automatic extraction.  Supports
            ``("messages", N)`` to trigger every *N* human messages.
            Defaults to ``("messages", 20)``.
    """

    state_schema = EvoMemoryState

    def __init__(
        self,
        *,
        backend: BACKEND_TYPES,
        memory_path: str = "/memory/MEMORY.md",
        extraction_model: BaseChatModel | None = None,
        trigger: tuple[str, int] = ("messages", 20),
    ) -> None:
        self._backend = backend
        self._memory_path = memory_path
        self._extraction_model = extraction_model
        self._trigger = trigger
        self._last_extraction_at: dict[str, int] = {}  # message count per thread

    # -- backend resolution --------------------------------------------------

    def _get_backend(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> BackendProtocol:
        if callable(self._backend):
            config = cast("RunnableConfig", getattr(runtime, "config", {}))
            tool_runtime = ToolRuntime(
                state=state,
                context=runtime.context,
                stream_writer=runtime.stream_writer,
                store=runtime.store,
                config=config,
                tool_call_id=None,
            )
            return self._backend(tool_runtime)
        return self._backend

    # -- agent-level preload -------------------------------------------------

    def before_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any] | None:
        if state.get(_STATE_MEMORY_KEY) is not None:
            return None
        backend = self._get_backend(state, runtime)
        memory = self._read_memory(backend)
        _CURRENT_MEMORY.set(memory)
        return {_STATE_MEMORY_KEY: memory}

    async def abefore_agent(
        self,
        state: AgentState[Any],
        runtime: Runtime,
        config: RunnableConfig,
    ) -> dict[str, Any] | None:
        if state.get(_STATE_MEMORY_KEY) is not None:
            return None
        backend = self._get_backend(state, runtime)
        memory = await self._aread_memory(backend)
        _CURRENT_MEMORY.set(memory)
        return {_STATE_MEMORY_KEY: memory}

    # -- read / write helpers ------------------------------------------------

    def _read_memory(self, backend: BackendProtocol) -> str:
        """Read MEMORY.md content (raw bytes → str)."""
        try:
            responses = backend.download_files([self._memory_path])
            if (
                responses
                and responses[0].content is not None
                and responses[0].error is None
            ):
                return responses[0].content.decode("utf-8")
        except Exception as e:
            logger.debug("Failed to read memory at %s: %s", self._memory_path, e)
        return ""

    async def _aread_memory(self, backend: BackendProtocol) -> str:
        try:
            responses = await backend.adownload_files([self._memory_path])
            if (
                responses
                and responses[0].content is not None
                and responses[0].error is None
            ):
                return responses[0].content.decode("utf-8")
        except Exception as e:
            logger.debug("Failed to read memory at %s: %s", self._memory_path, e)
        return ""

    def _write_memory(
        self, backend: BackendProtocol, old_content: str, new_content: str
    ) -> None:
        """Write updated MEMORY.md (edit if exists, write if new)."""
        try:
            if old_content:
                result = backend.edit(self._memory_path, old_content, new_content)
            else:
                result = backend.write(self._memory_path, new_content)
            if result and result.error:
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as e:
            logger.warning("Exception writing memory: %s", e)

    async def _awrite_memory(
        self, backend: BackendProtocol, old_content: str, new_content: str
    ) -> None:
        try:
            if old_content:
                result = await backend.aedit(
                    self._memory_path, old_content, new_content
                )
            else:
                result = await backend.awrite(self._memory_path, new_content)
            if result and result.error:
                logger.warning("Failed to write memory: %s", result.error)
        except Exception as e:
            logger.warning("Exception writing memory: %s", e)

    # -- threshold check -----------------------------------------------------

    def _should_extract(self, thread_id: str, messages: list[AnyMessage]) -> bool:
        """Check if we should run automatic extraction."""
        if self._extraction_model is None:
            return False

        trigger_type, trigger_value = self._trigger
        if trigger_type == "messages":
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            last = self._last_extraction_at.get(thread_id, 0)
            return (human_count - last) >= trigger_value
        return False

    # -- extraction ----------------------------------------------------------

    @staticmethod
    def _build_extraction_prompt(memory: str, messages: list[AnyMessage]) -> str:
        """Build the extraction prompt from recent human/AI messages."""
        recent = filter_messages(messages[-30:], include_types=["human", "ai"])
        conv_parts = []
        for msg in recent:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            conv_parts.append(f"[{role}]: {content}")
        return EXTRACTION_PROMPT.format(
            current_memory=memory,
            conversation="\n".join(conv_parts),
        )

    @staticmethod
    def _structured_output_kwargs(model: BaseChatModel) -> dict[str, Any]:
        """Return extra kwargs for with_structured_output based on provider.

        OpenAI's Structured Outputs (default since langchain-openai 0.3)
        requires ``additionalProperties: false`` and all-required fields.
        The ExtractedMemory schema uses Optional unions which violate these
        rules.  Fall back to function_calling for OpenAI models.
        """
        model_module = type(model).__module__ or ""
        if model_module.startswith("langchain_openai"):
            return {"method": "function_calling"}
        return {}

    @staticmethod
    def _disable_thinking(model: BaseChatModel) -> BaseChatModel:
        """Return a copy of the model with thinking/reasoning disabled.

        Anthropic's API does not allow extended thinking when tool_choice
        forces tool use (as with_structured_output does).  Similarly,
        OpenAI reasoning can conflict.  Strip these settings so extraction
        works reliably.

        Uses model_copy() to produce a real new instance — bind() only
        wraps the model in a RunnableBinding whose kwargs do NOT override
        first-class Pydantic fields like ``thinking`` on ChatAnthropic.
        """
        updates: dict[str, Any] = {}
        model_kwargs = getattr(model, "model_kwargs", {}) or {}

        if getattr(model, "thinking", None) or "thinking" in model_kwargs:
            updates["thinking"] = None
        if getattr(model, "reasoning", None) or "reasoning" in model_kwargs:
            updates["reasoning"] = None

        if not updates:
            return model

        # Prefer Pydantic model_copy (creates a true new instance with the
        # field cleared) over bind() which only adds invocation kwargs.
        try:
            return model.model_copy(update=updates)
        except Exception:
            # Fallback for non-Pydantic or unusual model classes
            return model.bind(**{k: v for k, v in updates.items() if v is not None})

    def _extract(
        self, model: BaseChatModel, memory: str, messages: list[AnyMessage]
    ) -> dict[str, Any]:
        """Run LLM extraction on recent messages using structured output."""
        prompt = self._build_extraction_prompt(memory, messages)
        try:
            plain_model = self._disable_thinking(model)
            so_kwargs = self._structured_output_kwargs(plain_model)
            structured_model = plain_model.with_structured_output(
                ExtractedMemory, **so_kwargs
            )
            result = structured_model.invoke(prompt)
            return result.model_dump(exclude_none=True)
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return {}

    async def _aextract(
        self, model: BaseChatModel, memory: str, messages: list[AnyMessage]
    ) -> dict[str, Any]:
        """Async: Run LLM extraction on recent messages using structured output."""
        prompt = self._build_extraction_prompt(memory, messages)
        try:
            plain_model = self._disable_thinking(model)
            so_kwargs = self._structured_output_kwargs(plain_model)
            structured_model = plain_model.with_structured_output(
                ExtractedMemory, **so_kwargs
            )
            result = await structured_model.ainvoke(prompt)
            return result.model_dump(exclude_none=True)
        except Exception as e:
            logger.warning("Memory extraction failed: %s", e)
            return {}

    # -- middleware hooks -----------------------------------------------------

    def modify_request(self, request: ModelRequest) -> ModelRequest:
        """Inject memory content and instructions into the system message.

        Always injects ``<memory_instructions>`` so the agent knows it can
        save memories, even when MEMORY.md does not exist yet.
        """
        state = request.state or {}
        memory_content = state.get(_STATE_MEMORY_KEY, "")
        if not memory_content:
            memory_content = _CURRENT_MEMORY.get()
        if not memory_content and request.runtime is not None:
            try:
                backend = self._get_backend(state, request.runtime)
                memory_content = self._read_memory(backend)
                _CURRENT_MEMORY.set(memory_content)
            except Exception as e:
                logger.debug("Failed to load memory during modify_request: %s", e)
        # Use placeholder when memory file doesn't exist yet
        if not memory_content:
            memory_content = "(No memory saved yet. Create `/memory/MEMORY.md` when you learn important information.)"

        from deepagents.middleware._utils import append_to_system_message

        injection = MEMORY_INJECTION_TEMPLATE.format(memory_content=memory_content)
        new_system = append_to_system_message(request.system_message, injection)
        return request.override(system_message=new_system)

    def wrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], ModelResponse],
    ) -> ModelResponse:
        """Inject memory into system prompt before every LLM call."""
        modified = self.modify_request(request)
        return handler(modified)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        modified = self.modify_request(request)
        return await handler(modified)

    def before_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Read memory and optionally run extraction before each LLM call."""
        backend = self._get_backend(state, runtime)
        messages = state["messages"]
        thread_id = _get_thread_id(runtime)

        # Always read memory for injection
        memory = self._read_memory(backend)
        _CURRENT_MEMORY.set(memory)
        state_update: dict[str, Any] | None = None
        if state.get(_STATE_MEMORY_KEY) != memory:
            state_update = {_STATE_MEMORY_KEY: memory}

        # Check extraction threshold
        if self._should_extract(thread_id, messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = self._extract(self._extraction_model, memory, messages)
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    self._write_memory(backend, memory, new_memory)
                    _CURRENT_MEMORY.set(new_memory)
                    logger.info("Auto-extracted and updated memory")
                    state_update = {_STATE_MEMORY_KEY: new_memory}
            self._last_extraction_at[thread_id] = human_count

        return state_update

    async def abefore_model(
        self,
        state: AgentState[Any],
        runtime: Runtime,
    ) -> dict[str, Any] | None:
        """Async: Read memory and optionally run extraction."""
        backend = self._get_backend(state, runtime)
        messages = state["messages"]
        thread_id = _get_thread_id(runtime)

        memory = await self._aread_memory(backend)
        _CURRENT_MEMORY.set(memory)
        state_update: dict[str, Any] | None = None
        if state.get(_STATE_MEMORY_KEY) != memory:
            state_update = {_STATE_MEMORY_KEY: memory}

        if self._should_extract(thread_id, messages):
            human_count = sum(1 for m in messages if isinstance(m, HumanMessage))
            extracted = await self._aextract(self._extraction_model, memory, messages)
            if extracted:
                new_memory = _merge_memory(memory, extracted)
                if new_memory != memory:
                    await self._awrite_memory(backend, memory, new_memory)
                    _CURRENT_MEMORY.set(new_memory)
                    logger.info("Auto-extracted and updated memory")
                    state_update = {_STATE_MEMORY_KEY: new_memory}
            self._last_extraction_at[thread_id] = human_count

        return state_update


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def create_memory_middleware(
    memory_dir: str | None = None,
    extraction_model: BaseChatModel | None = None,
    trigger: tuple[str, int] = ("messages", 20),
) -> EvoMemoryMiddleware:
    """Create an EvoMemoryMiddleware for long-term memory.

    Uses a FilesystemBackend rooted at ``memory_dir`` so that memory
    persists across threads and sessions.

    Args:
        memory_dir: Path to the shared memory directory (not per-session).
            Defaults to ``paths.MEMORY_DIR``.
        extraction_model: Chat model for auto-extraction (optional; if None,
            only prompt-guided manual memory updates via edit_file will work).
        trigger: When to auto-extract. Default: every 20 human messages.

    Returns:
        Configured EvoMemoryMiddleware instance.
    """
    from deepagents.backends import FilesystemBackend

    from ..paths import MEMORY_DIR as _DEFAULT_MEMORY_DIR

    if memory_dir is None:
        memory_dir = str(_DEFAULT_MEMORY_DIR)

    memory_backend = FilesystemBackend(
        root_dir=memory_dir,
        virtual_mode=True,
    )
    return EvoMemoryMiddleware(
        backend=memory_backend,
        memory_path="/MEMORY.md",
        extraction_model=extraction_model,
        trigger=trigger,
    )
