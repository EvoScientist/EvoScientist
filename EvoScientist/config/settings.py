"""Configuration management for EvoScientist.

Handles loading, saving, and merging configuration from multiple sources.
See :func:`get_effective_config` for the authoritative priority chain —
``EVOSCIENTIST_*`` shell values and third-party keys are treated
asymmetrically with respect to workspace ``.env`` handling.
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import asdict, dataclass, fields
from enum import StrEnum
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal, get_args, get_origin, get_type_hints

import yaml
from dotenv import dotenv_values, find_dotenv

from .. import paths
from . import store

# Tools that run shell commands and need manual HITL approval (subject to
# shell_allow_list). Single source of truth for every interrupt consumer
# (stream/display.py, channels/interaction.py) — keep aligned with the agent's
# `interrupt_on` set in EvoScientist.py.
HITL_SHELL_TOOLS = ("execute", "run_in_background")

# Armed non-shell destructive tools must always prompt — no allow-list carve-outs
# (their args carry paths, not commands). Keep aligned with HITL_INTERRUPT_ON.
HITL_ALWAYS_PROMPT_TOOLS = ("delete", "schedule_task")


class MemoryObservationTarget(StrEnum):
    """Runtime locations that can receive `record_observation`."""

    AGENT = "agent"
    TURN_WORKER = "turn_worker"
    SUBAGENT_WORKER = "subagent_worker"


class MemoryObservationWriter(StrEnum):
    """Configured observation-writing policy."""

    OFF = "off"
    AGENT = "agent"
    WORKER = "worker"
    ALL = "all"

    def enables(self, target: MemoryObservationTarget) -> bool:
        match self:
            case MemoryObservationWriter.OFF:
                return False
            case MemoryObservationWriter.AGENT:
                return target == MemoryObservationTarget.AGENT
            case MemoryObservationWriter.WORKER:
                return target in (
                    MemoryObservationTarget.TURN_WORKER,
                    MemoryObservationTarget.SUBAGENT_WORKER,
                )
            case MemoryObservationWriter.ALL:
                return target in (
                    MemoryObservationTarget.AGENT,
                    MemoryObservationTarget.TURN_WORKER,
                    MemoryObservationTarget.SUBAGENT_WORKER,
                )


class MemorySkillSynthesisMode(StrEnum):
    """Configured AutoSkills approval behavior."""

    REVIEW = "review"
    AUTO = "auto"


class MemorySkillSynthesisCadence(StrEnum):
    """Preset cadence for the built-in AutoSkills schedule."""

    NIGHTLY = "nightly"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


DEFAULT_MEMORY_OBSERVATION_WRITER = MemoryObservationWriter.ALL
DEFAULT_MEMORY_SKILL_SYNTHESIS_MODE = MemorySkillSynthesisMode.REVIEW
DEFAULT_MEMORY_SKILL_SYNTHESIS_CADENCE = MemorySkillSynthesisCadence.WEEKLY
DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME = "03:00"
DEFAULT_PROACTIVE_IDLE_MINUTES = 120
DEFAULT_PROACTIVE_QUIET_HOURS = "22:00-08:00"


def _normalize_hhmm(value: Any) -> str | None:
    parts = str(value).strip().split(":")
    if len(parts) != 2:
        return None
    hour, minute = parts
    if not (hour.isdecimal() and minute.isdecimal()):
        return None
    try:
        hour_int = int(hour)
        minute_int = int(minute)
    except ValueError:
        return None
    if not (0 <= hour_int <= 23 and 0 <= minute_int <= 59):
        return None
    return f"{hour_int:02d}:{minute_int:02d}"


def _normalize_quiet_hours(value: str) -> str | None:
    """Normalise an ``HH:MM-HH:MM`` window, or ``None`` when it is not one.

    An empty or blank value is ``""`` (no quiet hours).
    """
    text = value.strip()
    if not text:
        return ""
    parts = text.split("-")
    if len(parts) != 2:
        return None
    start, end = _normalize_hhmm(parts[0]), _normalize_hhmm(parts[1])
    if start is None or end is None:
        return None
    return f"{start}-{end}"


def _valid_timezone(value: str) -> str | None:
    """The IANA zone name in *value*, ``""`` when blank, ``None`` when unknown."""
    name = value.strip()
    if not name:
        return ""
    from zoneinfo import ZoneInfo

    try:
        ZoneInfo(name)
    except (ValueError, OSError, KeyError):
        # ZoneInfoNotFoundError is a KeyError; malformed keys raise ValueError.
        return None
    return name


# =============================================================================
# Configuration paths
# =============================================================================


def get_config_dir() -> Path:
    """Get the configuration directory path.

    ``$XDG_CONFIG_HOME/evoscientist`` when that is set, otherwise
    ``~/.config/evoscientist``, read once at startup (``paths.CONFIG_DIR``).
    """
    return paths.CONFIG_DIR


def get_config_path() -> Path:
    """Get the path to the configuration file."""
    return get_config_dir() / "config.yaml"


class GatewaySurface(StrEnum):
    """A production surface that selects a gateway backend.

    Each member maps to a per-surface ``gateway_backend_*`` config field; see
    :func:`resolve_gateway_backend`. Only one surface runs per process, so a
    surface resolves its backend once at startup and threads it explicitly.
    """

    SERVE = "serve"
    SINGLE_SHOT = "single_shot"
    INTERACTIVE = "interactive"
    TUI = "tui"
    STANDALONE = "standalone"


# =============================================================================
# Configuration dataclass
# =============================================================================

# OpenRouter app-attribution defaults (issue #339). Single source of truth: the
# EvoScientistConfig fields below default to these, and llm/models.py imports
# them for its env-fallback, so the values never drift across the two layers.
OPENROUTER_DEFAULT_HTTP_REFERER = "https://github.com/EvoScientist/EvoScientist"
OPENROUTER_DEFAULT_APP_TITLE = "EvoScientist"
# OpenRouter honors only the first 2 categories per request (server-side limit)
# and silently ignores the rest, so keep the two most relevant ones. Chosen per
# maintainer review — creative-writing is a less competitive marketplace group.
OPENROUTER_DEFAULT_APP_CATEGORIES = "creative-writing,personal-agent"


@dataclass
class EvoScientistConfig:
    """EvoScientist configuration settings.

    Attributes:
        anthropic_api_key: Anthropic API key for Claude models.
        openai_api_key: OpenAI API key for GPT models.
        nvidia_api_key: NVIDIA API key for NVIDIA models.
        google_api_key: Google API key for Gemini models.
        tavily_api_key: Tavily API key for web search.
        provider: Default LLM provider ('anthropic', 'openai', 'google-genai', or 'nvidia').
        model: Default model name (short name or full ID).
        auxiliary_provider: Provider for auxiliary_model (empty = use main provider).
        auxiliary_model: Model for memory workers + tool selector + scheduler (empty = use main model).
        default_mode: Default workspace mode ('daemon' or 'run').
        default_workdir: Default workspace directory (empty = use current working directory).
        show_thinking: Whether to show thinking panels in CLI.
    """

    # API Keys
    anthropic_api_key: str = ""
    anthropic_base_url: str = ""
    anthropic_auth_mode: str = "api_key"  # "api_key" | "oauth"
    openai_api_key: str = ""
    openai_auth_mode: str = "api_key"  # "api_key" | "oauth"
    nvidia_api_key: str = ""
    google_api_key: str = ""
    minimax_api_key: str = ""
    minimax_base_url: str = ""
    siliconflow_api_key: str = ""
    openrouter_api_key: str = ""
    atlascloud_api_key: str = ""
    requesty_api_key: str = ""
    novita_api_key: str = ""
    mimo_api_key: str = ""
    mimo_token_plan_api_key: str = ""
    mimo_token_plan_base_url: str = ""
    opper_api_key: str = ""
    deepseek_api_key: str = ""
    zhipu_api_key: str = ""
    volcengine_api_key: str = ""
    dashscope_api_key: str = ""
    moonshot_api_key: str = ""
    kimi_api_key: str = ""
    custom_openai_api_key: str = ""
    custom_openai_base_url: str = ""
    custom_anthropic_api_key: str = ""
    custom_anthropic_base_url: str = ""
    ollama_base_url: str = ""
    tavily_api_key: str = ""

    # LLM Settings
    provider: str = "anthropic"
    model: str = "claude-sonnet-5-5"
    model_fallbacks: str = ""  # "model:provider,model:provider" fallback chain
    # Transient model-call failures (5xx, 429, connection drops): how long one
    # model call keeps retrying after it first fails, in seconds; 0 = no retry.
    model_retry_wait_seconds: int = 60  # interactive sessions; memory workers, linker
    auto_mode_retry_wait_seconds: int = 1800  # auto_mode, scheduler, AutoSkills
    # Optional auxiliary model for background/helper LLM calls (memory workers +
    # tool selector). Empty = fall back to the main model/provider.
    auxiliary_provider: str = ""  # empty = use main provider
    auxiliary_model: str = ""  # empty = use main model

    # Async Sub-agent Settings
    # When True (default), the EvoSci CLI auto-starts a langgraph dev subprocess
    # so any sub-agent flagged ``async: true`` in subagents/<name>.yaml runs
    # non-blocking via AsyncSubAgent. Currently affects writing-agent and
    # data-analysis-agent. Adds ~10-15s to CLI startup (langgraph dev cold
    # start, mostly MCP server spawn time).
    #
    # Set False to run fully in-process — saves the startup cost in scenarios
    # where async isn't useful: short scripted EvoSci runs (CI / one-shot
    # ``-p "..."``), low-RAM environments, or workflows that only need the
    # synchronous sub-agents (planner / research / code / debug).
    enable_async_subagents: bool = True

    # Port for the auto-started langgraph dev subprocess. 6174 is Kaprekar's
    # constant — a memorable EvoScientist-themed default that avoids collisions
    # with common dev ports (3000/5000/8000/8080) and the langgraph CLI default
    # 2024. Override if it conflicts with another local service.
    langgraph_dev_port: int = 6174

    # Network interface the langgraph dev subprocess binds to. Loopback by
    # default — this is the unauthenticated agent API (the agent can run
    # shell), so "0.0.0.0" is opt-in and every launcher prints a PUBLIC BIND
    # banner while exposed. Internal callers *connect* via manager._probe_host,
    # so widening never redirects their traffic off-box.
    langgraph_dev_host: str = "127.0.0.1"

    # Port for the WebUI front-end (Next.js server from @evoscientist/webui),
    # used only when ui_backend == "webui". 4716 is 6174 reversed — a memorable
    # pairing with the langgraph dev port that it connects to. The backend keeps
    # its own port (langgraph_dev_port); this is just the browser server.
    webui_port: int = 4716

    # Network interface the WebUI front-end binds to. Loopback by default,
    # matching langgraph_dev_host: this server is not a passive app shell —
    # its API reads, writes and uploads workspace files and installs skills,
    # all unauthenticated. Set "0.0.0.0" (with langgraph_dev_host) for LAN.
    webui_host: str = "127.0.0.1"

    # --- Scheduled tasks (cron) ---
    # Master switch for scheduled tasks (/schedule, NL tools, scheduler context). Defaults
    # True so the feature is available out-of-the-box; set False to disable.
    enable_scheduler: bool = True
    # Default IANA timezone for cron schedules created without an explicit tz.
    # Empty string => the host's local IANA zone (resolved via tzlocal), falling
    # back to UTC if it can't be determined; set e.g. "Europe/London" to pin one.
    scheduler_default_timezone: str = ""

    # Whether langgraph dev persists its runtime state to .langgraph_api/ next
    # to the subprocess cwd. True (default) keeps async-task, scheduler, and
    # Store API state across subprocess restarts — useful for future
    # cross-session async, cron, and Store features. Set False to suppress
    # writes (workspace stays cleaner; state is in-memory only and lost on
    # CLI exit). EvoScientist's main thread persistence uses sessions.db
    # regardless of this setting.
    langgraph_dev_file_persistence: bool = True

    # Concurrency: how many runs each langgraph dev worker processes in parallel.
    # 10 is the langgraph dev recommended default and works well on a typical
    # dev machine. Lower it (e.g., 4) on memory-constrained or low-core
    # machines if multiple async sub-agents in flight cause noticeable
    # slowdown.
    langgraph_dev_jobs_per_worker: int = 10

    # Keep the auto-started langgraph dev subprocess running after the CLI
    # exits. The next `EvoSci` start in the same workspace reuses it instantly
    # instead of paying the cold boot (~15s). Starting in a DIFFERENT workspace
    # raises WorkspaceMismatchError with the leftover server's pid — stop it
    # manually (the server is pinned to one workspace per process). Known
    # limitation: changing langgraph_dev_port/host while a keepalive server
    # runs orphans its records — run `EvoSci server stop` before switching.
    langgraph_dev_keepalive: bool = False

    # Which backend serves the CLI/TUI's graph runs: the in-process local
    # gateway (default) or the langgraph dev server via LangGraphServerGateway.
    # Server-gateway-migration rollback flag: flipping back to "local" restores
    # the in-process path without code changes. The CLI keeps building its
    # in-process agent (and its MCP sessions) until a surface actually cuts
    # over to the server gateway - skipping that init lands with the surface
    # cutover, not here.
    gateway_backend: Literal["local", "langgraph_server"] = "local"

    # Per-surface overrides of ``gateway_backend``. Each production surface can
    # pick its own backend; ``"inherit"`` defers to the global ``gateway_backend``
    # above. The fixed-workspace surfaces (serve, single-shot, standalone) default
    # to ``inherit``, so the global flag moves them. The interactive CLI and TUI
    # default to ``local`` instead: the server path is still lossy there (the
    # per-session workspace is not applied server-side, #413, and per-run config
    # is dropped on resume, #454), so a global ``langgraph_server`` must not move
    # them - reaching the server backend on those surfaces takes an explicit
    # ``langgraph_server`` here. Resolve with :func:`resolve_gateway_backend`;
    # never read these fields directly for a routing decision.
    gateway_backend_serve: Literal["inherit", "local", "langgraph_server"] = "inherit"
    gateway_backend_single_shot: Literal["inherit", "local", "langgraph_server"] = (
        "inherit"
    )
    gateway_backend_interactive: Literal["inherit", "local", "langgraph_server"] = (
        "local"
    )
    gateway_backend_tui: Literal["inherit", "local", "langgraph_server"] = "local"
    gateway_backend_standalone: Literal["inherit", "local", "langgraph_server"] = (
        "inherit"
    )

    # Proactive pushes (#263): a conversation armed for proactive pushes in
    # ``EvoSci serve`` may get one unprompted message once it has gone quiet.
    # These settings tune when; arming is per conversation, so there is no
    # global on/off switch.
    # Minutes since the conversation's last activity before a push is considered.
    proactive_idle_minutes: int = DEFAULT_PROACTIVE_IDLE_MINUTES
    # Window "HH:MM-HH:MM" (may wrap midnight) in which no push is generated.
    # An empty string, or equal start and end, means no quiet hours.
    proactive_quiet_hours: str = DEFAULT_PROACTIVE_QUIET_HOURS
    # IANA timezone the quiet-hours window is evaluated in. Empty string means
    # the host's local zone.
    proactive_timezone: str = ""

    # Max LangGraph super-steps (LLM call / tool call / sub-agent delegation
    # each count as 1) before raising GraphRecursionError. Resets on every
    # ``agent.invoke()`` — i.e., this is per-turn, NOT per-conversation. For
    # long conversations the relevant mechanisms are checkpointer persistence
    # (sessions.db), ContextEditingMiddleware (window management), and
    # EvoMemoryMiddleware (cross-turn memory).
    #
    # 1,000,000 is "effectively unlimited" — typical research turns use
    # 200-1000 steps; reaching 1M would cost ~$10K in tokens, by which point
    # rate limits, context overflow, or API quota errors would trip first.
    # Lower (e.g., 5000) if you want a tighter safety net against runaway loops.
    recursion_limit: int = 1_000_000

    # Memory Settings
    # Profile memory injects and maintains `/memories/profile/...` files.
    memory_profile_enabled: bool = True
    # Observation memory indexes `/memories/observations/...` and adds
    # observation-read guidance/context. Writes require this switch plus an
    # allowed `memory_observation_writer` role below.
    memory_observations_enabled: bool = True
    # Which observation-writing path receives the `record_observation` tool:
    # "off" disables writes; "agent" means live agents; "worker" means
    # post-run memory workers; "all" means live agents and post-run memory
    # workers.
    memory_observation_writer: MemoryObservationWriter = (
        DEFAULT_MEMORY_OBSERVATION_WRITER
    )
    # Post-turn and post-subagent memory workers. Disable for no-background-memory
    # controls while still allowing live agents to read configured memory.
    memory_workers_enabled: bool = True
    # Slow EvoMemory maintenance that periodically scans observation clusters
    # and drafts reusable skills.
    memory_skill_synthesis_enabled: bool = True
    memory_skill_synthesis_mode: MemorySkillSynthesisMode = (
        DEFAULT_MEMORY_SKILL_SYNTHESIS_MODE
    )
    memory_skill_synthesis_cadence: MemorySkillSynthesisCadence = (
        DEFAULT_MEMORY_SKILL_SYNTHESIS_CADENCE
    )
    memory_skill_synthesis_time: str = DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME
    # Max number of parsed observation files kept in the process-wide parse
    # cache. Each entry holds one parsed document keyed on the file path; at
    # the end of a call the LRU trims down to max(cap, entries touched by
    # the call), so an active store larger than the cap temporarily exceeds
    # it instead of thrashing. 2048 is generous for the single-workspace
    # deploy model; raise for a long-running server that cycles through many
    # large workspaces.
    memory_observation_cache_max_files: int = 2048

    # Workspace Settings
    default_mode: Literal["daemon", "run"] = "daemon"
    default_workdir: str = ""

    # UI Settings
    show_thinking: bool = True
    # "webui" launches the browser front-end (@evoscientist/webui, installed
    # locally by `EvoSci setup` or on the first launch) +
    # a deploy-style langgraph server instead of the in-terminal CLI/TUI.
    ui_backend: Literal["cli", "tui", "webui"] = "tui"
    # Download source for `EvoSci setup` and on-demand installs (Node.js).
    # "cn" uses mainland China mirrors; set by `EvoSci setup --cn`.
    mirror: Literal["default", "cn"] = "default"
    log_level: str = "warning"
    # Empty means use the provider/model default. A non-empty value is an
    # explicit user override exported as EVOSCIENTIST_REASONING_EFFORT.
    reasoning_effort: str = ""
    # Anthropic prompt caching for OpenRouter anthropic/* models. Opt out if
    # cache-write costs outweigh the benefit for a workflow.
    openrouter_anthropic_prompt_cache: bool = True
    # OpenRouter app attribution (issue #339). Sent only for the openrouter
    # provider; identifies EvoScientist in OpenRouter's app rankings/analytics.
    # Override (e.g. a private fork) via these fields or their env vars. A custom
    # title only takes effect together with a custom referer: OpenRouter keys app
    # pages by referer, so a lone title would rename the shared EvoScientist page.
    # Defaults live in the module constants above (also imported by llm/models.py).
    openrouter_http_referer: str = OPENROUTER_DEFAULT_HTTP_REFERER
    openrouter_app_title: str = OPENROUTER_DEFAULT_APP_TITLE
    # Comma-separated; split into a list before being passed to
    # langchain-openrouter (its app_categories kwarg expects list[str]).
    openrouter_app_categories: str = OPENROUTER_DEFAULT_APP_CATEGORIES

    # Channel Settings
    channel_enabled: str = ""  # "imessage" | "telegram" | "discord" | "slack" | "wechat" | "dingtalk" | "feishu" | "email" | "qq" | "signal" | "" (comma-separated for multiple)
    channel_send_thinking: bool = True  # forward thinking to any channel
    channel_debug_tracing: bool = False  # emit extra inbound diagnostics at DEBUG
    require_mention: str = "group"  # "always" | "group" | "off"
    text_chunk_limit: int = 0  # 0 = use capability default
    allowed_channels: str = ""  # comma-separated channel IDs, empty = allow all

    # iMessage Settings
    imessage_enabled: bool = False  # legacy compat
    imessage_allowed_senders: str = ""

    # Telegram Settings
    telegram_bot_token: str = ""
    telegram_allowed_senders: str = ""
    telegram_proxy: str = ""

    # Discord Settings
    discord_bot_token: str = ""
    discord_allowed_senders: str = ""
    discord_allowed_channels: str = ""
    discord_proxy: str = ""

    # Slack Settings
    slack_bot_token: str = ""
    slack_app_token: str = ""
    slack_allowed_senders: str = ""
    slack_allowed_channels: str = ""
    slack_proxy: str = ""

    # Feishu Settings
    feishu_app_id: str = ""
    feishu_app_secret: str = ""
    feishu_verification_token: str = ""
    feishu_encrypt_key: str = ""
    feishu_webhook_port: int = 9000
    feishu_allowed_senders: str = ""
    feishu_domain: str = "https://open.feishu.cn"
    feishu_proxy: str = ""
    feishu_subscription_mode: str = "webhook"  # "webhook" | "websocket"

    # WeChat Settings
    wechat_backend: str = "wecom"
    wechat_webhook_port: int = 9001
    wechat_allowed_senders: str = ""
    wechat_proxy: str = ""
    wechat_wecom_corp_id: str = ""
    wechat_wecom_agent_id: str = ""
    wechat_wecom_secret: str = ""
    wechat_wecom_token: str = ""
    wechat_wecom_encoding_aes_key: str = ""
    wechat_mp_app_id: str = ""
    wechat_mp_app_secret: str = ""
    wechat_mp_token: str = ""
    wechat_mp_encoding_aes_key: str = ""
    # Personal WeChat (iLink Bot) — credentials obtained via QR-code login.
    # Run: python -m EvoScientist.channels.wechat.serve --qr-login
    wechat_personal_account_id: str = ""
    wechat_personal_token: str = ""
    wechat_personal_base_url: str = ""
    wechat_personal_cdn_base_url: str = ""
    wechat_personal_dm_policy: str = "open"
    wechat_personal_group_policy: str = "disabled"
    wechat_personal_group_allowed: str = ""

    # DingTalk Settings
    dingtalk_client_id: str = ""
    dingtalk_client_secret: str = ""
    dingtalk_allowed_senders: str = ""
    dingtalk_proxy: str = ""

    # Email Settings
    email_imap_host: str = ""
    email_imap_port: int = 993
    email_imap_username: str = ""
    email_imap_password: str = ""
    email_imap_mailbox: str = "INBOX"
    email_imap_use_ssl: bool = True
    email_smtp_host: str = ""
    email_smtp_port: int = 587
    email_smtp_username: str = ""
    email_smtp_password: str = ""
    email_smtp_use_tls: bool = True
    email_from_address: str = ""
    email_poll_interval: int = 30
    email_mark_seen: bool = True
    email_max_body_chars: int = 12000
    email_subject_prefix: str = "Re: "
    email_allowed_senders: str = ""

    # QQ Settings
    qq_app_id: str = ""
    qq_app_secret: str = ""
    qq_allowed_senders: str = ""

    # Signal Settings
    signal_phone_number: str = ""
    signal_cli_path: str = "signal-cli"
    signal_config_dir: str = ""
    signal_allowed_senders: str = ""
    signal_rpc_port: int = 7583

    # Shared webhook port (0 = disabled)
    shared_webhook_port: int = 9000

    # HITL (Human-in-the-Loop) Settings
    auto_approve: bool = False  # Auto-approve all tool executions without prompting
    auto_mode: bool = False  # Run unattended: imply auto_approve and disable ask_user
    shell_allow_list: str = ""  # Comma-separated shell command prefixes to auto-approve
    # Comma-separated origins (``https://host[:port]``) whose pages may use the
    # backend besides this machine's own. Read from config.yaml only: no
    # environment variable sets it, so a workspace ``.env`` cannot widen it.
    backend_allowed_origins: str = ""

    # Dangerous mode: real-filesystem access (no workspace confinement). The agent
    # operates on real absolute paths anywhere on disk; the privileged-command
    # blocklist (sudo/chmod/dd/...) still applies. Implies auto_approve.
    dangerous_mode: bool = False

    # Agent features
    enable_ask_user: bool = True  # Enable ask_user tool for agent-initiated questions

    # CodeInterpreterMiddleware (PTC — Parallel Tool Calls) tuning
    # The PTC allowlist itself is hardcoded in
    # ``EvoScientist/middleware/code_interpreter.py`` as a load-bearing safety
    # decision (excludes ``execute`` so PTC can't bypass HITL approval,
    # excludes ``write_file``/``edit_file`` because batched writes have no
    # benefit). Only the resource budget knobs are user-tunable.
    code_interpreter_timeout: float = 60.0  # seconds per JS eval
    code_interpreter_max_result_chars: int = 10000  # truncate large JSON results

    # Default per-command timeout (seconds) for the sandbox `execute` tool.
    # Only the default — the agent can still override per command up to the
    # deepagents max_execute_timeout cap (3600s).
    sandbox_execute_timeout: int = 300

    # Checkpoint pruning (sessions.db retention per (thread_id, checkpoint_ns))
    # Safety net for runaway conversations. Under DeltaChannel (deepagents 0.6+)
    # normal usage produces linear growth, so this default is set well above
    # any realistic conversation length (~180-450 turns of dialogue) while
    # still capping legacy bloat at upgrade time. 0 disables ongoing pruning
    # entirely; the one-time legacy migration sweep still runs.
    checkpoint_keep_per_thread: int = 1000

    # DM access control policy
    dm_policy: str = "allowlist"

    # OpenAI API mode - "" = auto, "true" = force Responses, "false" = force Completions
    use_responses_api: str = ""

    # ccproxy
    ccproxy_port: int = 8000

    # STT (Speech-to-Text) Settings
    stt_enabled: bool = False
    stt_language: str = "auto"  # "auto" | "zh" | "en"
    stt_model: str = ""  # override model id; empty = auto-select by language
    stt_device: str = "cpu"  # "cpu" | "cuda"
    stt_compute_type: str = "int8"  # "int8" | "float16" | "float32"

    def __post_init__(self) -> None:
        # A non-positive or non-int sandbox_execute_timeout (e.g. a hand-edited
        # config file value — load_config does not coerce file values — or a
        # 0/negative env value) would raise inside CustomSandboxBackend.__init__
        # and crash agent/CLI startup. Fall back to the default instead, matching
        # how malformed env values already degrade to defaults.
        t = self.sandbox_execute_timeout
        if not isinstance(t, int) or isinstance(t, bool) or t <= 0:
            logging.getLogger(__name__).warning(
                "Invalid sandbox_execute_timeout %r; falling back to 300.", t
            )
            self.sandbox_execute_timeout = 300

        # A retry budget that is not a whole number of seconds (a hand-edited
        # ``60s``) goes back to its default; ``0`` or less switches the retry off.
        for name in ("model_retry_wait_seconds", "auto_mode_retry_wait_seconds"):
            budget = getattr(self, name)
            if not isinstance(budget, int) or isinstance(budget, bool):
                default = type(self).__dataclass_fields__[name].default
                logging.getLogger(__name__).warning(
                    "Invalid %s %r; falling back to %d.", name, budget, default
                )
                setattr(self, name, default)

        # A non-positive cache cap would evict every file entry immediately,
        # defeating the cache entirely.
        cap = self.memory_observation_cache_max_files
        if not isinstance(cap, int) or isinstance(cap, bool) or cap < 1:
            logging.getLogger(__name__).warning(
                "Invalid memory_observation_cache_max_files %r; falling back to 2048.",
                cap,
            )
            self.memory_observation_cache_max_files = 2048

        # shell_allow_list is typed as a comma-separated string, but a YAML list
        # spelling (``shell_allow_list: [ls, cat]``) survives here as a list.
        # The policy resolver calls ``.split`` on it, so normalise to CSV once at
        # the source rather than guarding every consumer.
        if isinstance(self.shell_allow_list, list | tuple):
            self.shell_allow_list = ",".join(str(s) for s in self.shell_allow_list)
        if isinstance(self.backend_allowed_origins, list | tuple):
            self.backend_allowed_origins = ",".join(
                str(s) for s in self.backend_allowed_origins
            )

        # auto_mode and dangerous_mode both imply auto_approve regardless of
        # source (CLI, env, config file, direct construction) — done here so the
        # "unattended → zero prompts" contract holds even when either is set via
        # `config set` or a config file rather than a CLI flag.
        if self.auto_mode or self.dangerous_mode:
            self.auto_approve = True

        _normalize_str_enum_fields(self)
        _normalize_literal_fields(self)

        # Bind hosts reach socket.bind() / the langgraph CLI verbatim, where a
        # stray-whitespace or empty value surfaces as an opaque gaierror at
        # startup. Normalize to the field's own default instead.
        for _host_field, _host_default in (
            ("langgraph_dev_host", "127.0.0.1"),
            ("webui_host", "127.0.0.1"),
        ):
            _host = getattr(self, _host_field, _host_default)
            _host = _host.strip() if isinstance(_host, str) else ""
            setattr(self, _host_field, _host or _host_default)

        synthesis_time = _normalize_hhmm(self.memory_skill_synthesis_time)
        if synthesis_time is None:
            logging.getLogger(__name__).warning(
                "Invalid memory_skill_synthesis_time %r; falling back to %s.",
                self.memory_skill_synthesis_time,
                DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME,
            )
            self.memory_skill_synthesis_time = DEFAULT_MEMORY_SKILL_SYNTHESIS_TIME
        else:
            self.memory_skill_synthesis_time = synthesis_time

        self._validate_proactive_settings()

    def _validate_proactive_settings(self) -> None:
        """Replace an invalid proactive setting by its default, with a warning.

        A typo must neither crash startup nor open the quiet hours: a bad
        window falls back to the default window, not to "no quiet hours".
        """
        log = logging.getLogger(__name__)

        idle = self.proactive_idle_minutes
        if not isinstance(idle, int) or isinstance(idle, bool) or idle < 0:
            log.warning(
                "Invalid proactive_idle_minutes %r; falling back to %d.",
                idle,
                DEFAULT_PROACTIVE_IDLE_MINUTES,
            )
            self.proactive_idle_minutes = DEFAULT_PROACTIVE_IDLE_MINUTES

        quiet_hours = self.proactive_quiet_hours
        window = (
            _normalize_quiet_hours(quiet_hours)
            if isinstance(quiet_hours, str)
            else None
        )
        if window is None:
            log.warning(
                "Invalid proactive_quiet_hours %r (expected 'HH:MM-HH:MM'); "
                "falling back to %s.",
                quiet_hours,
                DEFAULT_PROACTIVE_QUIET_HOURS,
            )
            window = DEFAULT_PROACTIVE_QUIET_HOURS
        self.proactive_quiet_hours = window

        timezone = self.proactive_timezone
        zone = _valid_timezone(timezone) if isinstance(timezone, str) else None
        if zone is None:
            log.warning(
                "Invalid proactive_timezone %r; falling back to the host's zone.",
                timezone,
            )
            zone = ""
        self.proactive_timezone = zone


@dataclass(frozen=True)
class MemoryControls:
    """Resolved memory feature switches used by agent and worker wiring."""

    profile_enabled: bool
    observations_enabled: bool
    observation_writer: MemoryObservationWriter
    workers_enabled: bool

    @classmethod
    def from_config(cls, config: EvoScientistConfig) -> MemoryControls:
        return cls(
            profile_enabled=config.memory_profile_enabled,
            observations_enabled=config.memory_observations_enabled,
            observation_writer=config.memory_observation_writer,
            workers_enabled=config.memory_workers_enabled,
        )

    @property
    def memory_enabled(self) -> bool:
        return self.profile_enabled or self.observations_enabled

    def observation_tool_enabled(self, target: MemoryObservationTarget) -> bool:
        return self.observations_enabled and self.observation_writer.enables(target)

    def worker_needed(self, target: MemoryObservationTarget) -> bool:
        if not self.workers_enabled:
            return False
        match target:
            case MemoryObservationTarget.TURN_WORKER:
                return self.profile_enabled or self.observation_tool_enabled(target)
            case MemoryObservationTarget.SUBAGENT_WORKER:
                return self.profile_enabled or self.observation_tool_enabled(target)
            case MemoryObservationTarget.AGENT:
                return False


# =============================================================================
# Config file operations
# =============================================================================


def load_config() -> EvoScientistConfig:
    """Load configuration from file.

    Returns:
        EvoScientistConfig instance with values from file, or defaults if
        file doesn't exist.
    """
    config_path = get_config_path()

    if not config_path.exists():
        return EvoScientistConfig()

    try:
        with open(config_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        return config_from_mapping(data)
    except Exception:
        # On any error, return defaults
        return EvoScientistConfig()


def config_from_mapping(data: dict[str, Any]) -> EvoScientistConfig:
    """Build a config from the file's mapping, defaults for what it lacks."""
    # Filter to only valid fields; a blank ``key:`` loads as None, which no
    # field accepts, so it falls back to the default.
    valid_fields = {f.name for f in fields(EvoScientistConfig)}
    filtered_data = {
        k: v for k, v in data.items() if k in valid_fields and v is not None
    }
    return EvoScientistConfig(**filtered_data)


def save_config(config: EvoScientistConfig) -> None:
    """Save the whole configuration to file, atomically and under the lock.

    Raises :class:`~EvoScientist.config.store.ConfigParseError` when the
    existing file does not parse: ``config`` then most likely holds the
    defaults :func:`load_config` fell back to, and writing it would replace
    the user's configuration. Raises
    :class:`~EvoScientist.config.store.ConfigLockedError` while another
    process (``EvoSci onboard``) holds the lock.

    Args:
        config: EvoScientistConfig instance to save.
    """
    # Save all fields including empty API keys (users can set them via env vars instead)
    data = store.dump_config_mapping(_config_to_dict(config))
    with store.config_lock():
        store.read_config_mapping()
        store.write_config_bytes(data)


def reset_config() -> None:
    """Reset configuration to defaults by deleting the config file."""
    with store.config_lock():
        store.remove_config_file()


def _config_to_dict(config: EvoScientistConfig) -> dict[str, Any]:
    """Return a plain serializable config dict."""
    return {key: _plain_config_value(value) for key, value in asdict(config).items()}


# Secret fields, listed by name rather than matched on it: a name rule would
# miss ``feishu_encrypt_key`` and catch ``mimo_token_plan_base_url``.
API_KEY_FIELDS: frozenset[str] = frozenset(
    {
        "anthropic_api_key",
        "openai_api_key",
        "nvidia_api_key",
        "google_api_key",
        "minimax_api_key",
        "siliconflow_api_key",
        "openrouter_api_key",
        "atlascloud_api_key",
        "requesty_api_key",
        "novita_api_key",
        "mimo_api_key",
        "mimo_token_plan_api_key",
        "opper_api_key",
        "deepseek_api_key",
        "zhipu_api_key",
        "volcengine_api_key",
        "dashscope_api_key",
        "moonshot_api_key",
        "kimi_api_key",
        "custom_openai_api_key",
        "custom_anthropic_api_key",
        "tavily_api_key",
    }
)
CHANNEL_SECRET_FIELDS: frozenset[str] = frozenset(
    {
        "telegram_bot_token",
        "discord_bot_token",
        "slack_bot_token",
        "slack_app_token",
        "feishu_app_secret",
        "feishu_verification_token",
        "feishu_encrypt_key",
        "wechat_wecom_secret",
        "wechat_wecom_token",
        "wechat_wecom_encoding_aes_key",
        "wechat_mp_app_secret",
        "wechat_mp_token",
        "wechat_mp_encoding_aes_key",
        "wechat_personal_token",
        "dingtalk_client_secret",
        "email_imap_password",
        "email_smtp_password",
        "qq_app_secret",
    }
)
SECRET_FIELDS: frozenset[str] = API_KEY_FIELDS | CHANNEL_SECRET_FIELDS


def secret_hint(key: str, value: str) -> str | None:
    """The last four characters of an API key longer than eight, else ``None``.

    Passwords, tokens and channel secrets get no hint: it would show a large
    part of a short password.
    """
    if key in API_KEY_FIELDS and len(value) > 8:
        return "…" + value[-4:]
    return None


def display_config_value(key: str, value: Any) -> str:
    """A value for printing: secrets masked, an empty value as ``(not set)``."""
    if value == "" or value is None:
        return "(not set)"
    if key in SECRET_FIELDS:
        hint = secret_hint(key, str(value))
        return "***" + hint[1:] if hint else "***"
    return str(value)


# =============================================================================
# Config value operations
# =============================================================================

_LITERAL_ALIASES = {"rich": "cli", "textual": "tui"}
"""Legacy ``ui_backend`` spellings written by earlier onboarding versions.

Kept in step with ``_LEGACY_BACKEND_MAP`` in ``cli/tui_runtime.py``; config is a
foundational module, so the map is duplicated here rather than importing across
the config -> cli boundary.
"""


def _match_literal(value: Any, allowed: tuple[Any, ...]) -> str | None:
    """Case-insensitive Literal match that also accepts legacy aliases."""
    text = str(value).strip().lower()
    text = _LITERAL_ALIASES.get(text, text)
    return text if text in allowed else None


def _coerce_value(value: Any, field_type: Any) -> Any:
    """Coerce a value to the expected field type.

    Args:
        value: The value to coerce.
        field_type: The target type (from dataclass field).

    Returns:
        The coerced value.

    Raises:
        ValueError: If the value cannot be coerced.
        TypeError: If the value cannot be coerced.
    """
    if _is_str_enum_type(field_type):
        return field_type(str(value).strip().lower())
    if get_origin(field_type) is Literal:
        allowed = get_args(field_type)
        text = _match_literal(value, allowed)
        if text is None:
            raise ValueError(f"expected one of {allowed}, got {value!r}")
        return text
    if field_type == "bool" or field_type is bool:
        if isinstance(value, str):
            return value.lower() in ("true", "1", "yes", "on")
        return bool(value)
    if field_type == "int" or field_type is int:
        return int(value)
    if field_type == "float" or field_type is float:
        return float(value)
    return str(value)


def _plain_config_value(value: Any) -> Any:
    """Return the persisted/user-facing representation for a config value."""
    return value.value if isinstance(value, StrEnum) else value


@lru_cache(maxsize=1)
def _config_field_types() -> dict[str, Any]:
    """Return resolved dataclass annotations for config fields."""
    return get_type_hints(EvoScientistConfig)


def _config_field_type(key: str, fallback: Any) -> Any:
    """Return the resolved dataclass annotation for a config field."""
    return _config_field_types().get(key, fallback)


def _is_str_enum_type(field_type: Any) -> bool:
    return isinstance(field_type, type) and issubclass(field_type, StrEnum)


def _normalize_str_enum_fields(config: EvoScientistConfig) -> None:
    """Normalize all StrEnum config fields, falling back to field defaults."""
    for field in fields(config):
        field_type = _config_field_type(field.name, field.type)
        if not _is_str_enum_type(field_type):
            continue

        raw_value = getattr(config, field.name)
        try:
            value = _coerce_value(raw_value, field_type)
        except (ValueError, TypeError):
            default = field.default
            logging.getLogger(__name__).warning(
                "Invalid %s %r; falling back to %s.",
                field.name,
                raw_value,
                _plain_config_value(default),
            )
            value = default
        setattr(config, field.name, value)


def _normalize_literal_fields(config: EvoScientistConfig) -> None:
    """Validate Literal fields, falling back to defaults on malformed values.

    ``load_config`` does not coerce file values, so a typo'd Literal in the
    config file (or in direct construction) would otherwise flow verbatim
    into mode selection — e.g. ``gateway_backend: langgraph-server`` (hyphen)
    silently selecting the local path in the manager.
    """
    for field in fields(config):
        field_type = _config_field_type(field.name, field.type)
        if get_origin(field_type) is not Literal:
            continue

        raw_value = getattr(config, field.name)
        value = _match_literal(raw_value, get_args(field_type))
        if value is None:
            value = field.default
            logging.getLogger(__name__).warning(
                "Invalid %s %r; falling back to %s.",
                field.name,
                raw_value,
                _plain_config_value(value),
            )
        setattr(config, field.name, value)


_GATEWAY_SURFACE_FIELDS: dict[GatewaySurface, str] = {
    GatewaySurface.SERVE: "gateway_backend_serve",
    GatewaySurface.SINGLE_SHOT: "gateway_backend_single_shot",
    GatewaySurface.INTERACTIVE: "gateway_backend_interactive",
    GatewaySurface.TUI: "gateway_backend_tui",
    GatewaySurface.STANDALONE: "gateway_backend_standalone",
}


def resolve_gateway_backend(
    config: Any,
    surface: GatewaySurface,
) -> Literal["local", "langgraph_server"]:
    """Resolve the effective gateway backend for one surface.

    Returns the surface's ``gateway_backend_*`` override when it is not
    ``"inherit"``, else the global ``gateway_backend``. This is the single
    place routing decisions read the flag from; callers thread the returned
    value down (to the runtime factory and the dev-server spawn), so the
    factory and manager never re-read the global and get a different answer.
    """
    override = getattr(config, _GATEWAY_SURFACE_FIELDS[surface], "inherit")
    if override in ("local", "langgraph_server"):
        return override
    return getattr(config, "gateway_backend", "local")


def get_config_value(key: str) -> Any:
    """Get a single configuration value.

    Args:
        key: Configuration key name.

    Returns:
        The value, or None if key doesn't exist.
    """
    config = load_config()
    value = getattr(config, key, None)
    return _plain_config_value(value)


class SettingsError(ValueError):
    """One or more settings failed validation; ``errors`` maps field to reason."""

    def __init__(self, errors: dict[str, str]) -> None:
        self.errors = errors
        super().__init__(
            "; ".join(f"{key}: {reason}" for key, reason in errors.items())
        )


def validate_setting(key: str, value: Any) -> tuple[Any, str | None]:
    """Validate and coerce one setting; return ``(value, warning)``.

    The one rule set for ``EvoSci config set``, the TUI's writers and the
    settings API. Raises ``ValueError`` with the reason when the value is not
    accepted. The returned value is the plain form written to the file.
    """
    field_info = next((f for f in fields(EvoScientistConfig) if f.name == key), None)
    if field_info is None:
        raise ValueError("unknown setting")
    field_type = _config_field_type(key, field_info.type)

    # __post_init__ only clamps on load, so validate here too. Reject bool before coercion
    # (_coerce_value(True, int) would turn it into 1 and slip past).
    if key in ("sandbox_execute_timeout", "proactive_idle_minutes") and isinstance(
        value, bool
    ):
        raise ValueError("expected a number")

    try:
        value = _coerce_value(value, field_type)
    except (ValueError, TypeError) as e:
        if get_origin(field_type) is Literal or _is_str_enum_type(field_type):
            raise ValueError(str(e)) from None
        raise ValueError(
            f"expected {getattr(field_type, '__name__', field_type)}"
        ) from None
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("must be a finite number")

    warning = None
    if key == "sandbox_execute_timeout" and value <= 0:
        raise ValueError("must be greater than 0")
    if key == "memory_skill_synthesis_time":
        value = _normalize_hhmm(value)
        if value is None:
            raise ValueError("expected a time as HH:MM")
    if key == "proactive_idle_minutes" and value < 0:
        raise ValueError("must be 0 or more")
    if key == "proactive_quiet_hours":
        value = _normalize_quiet_hours(value)
        if value is None:
            raise ValueError("expected a time range as HH:MM-HH:MM")
    if key == "proactive_timezone":
        value = _valid_timezone(value)
        if value is None:
            raise ValueError("unknown time zone")
    if key in ("provider", "auxiliary_provider") and (value or key == "provider"):
        from .provider_keys import PROVIDERS

        if value not in PROVIDERS:
            raise ValueError("unknown provider")
    if key == "default_workdir" and value:
        path = Path(os.path.expanduser(value))
        if not path.is_absolute():
            raise ValueError("must be an absolute path")
        if path.exists() and not path.is_dir():
            raise ValueError("is a file, not a folder")
        if not path.exists():
            warning = "the folder does not exist yet; it is created at start"
    return _plain_config_value(value), warning


def set_config_values(
    changes: dict[str, Any], *, expected_revision: Any = store.UNSET
) -> tuple[str | None, dict[str, str]]:
    """Validate ``changes`` and write them to the file in one locked step.

    A ``None`` value removes the setting from the file, so its default
    applies. Nothing is written when any value fails: :class:`SettingsError`
    names every failing field. Returns ``(new_revision, warnings)``.

    Raises :class:`~EvoScientist.config.store.ConfigError` subclasses when
    the file does not parse, is locked, or changed since
    ``expected_revision`` (``None`` = "no file expected"; omitted = no check).
    """
    coerced: dict[str, Any] = {}
    warnings: dict[str, str] = {}
    errors: dict[str, str] = {}
    for key, value in changes.items():
        if value is None:
            if key in {f.name for f in fields(EvoScientistConfig)}:
                coerced[key] = None
            else:
                errors[key] = "unknown setting"
            continue
        try:
            coerced[key], warning = validate_setting(key, value)
        except ValueError as e:
            errors[key] = str(e)
            continue
        if warning:
            warnings[key] = warning
    if errors:
        raise SettingsError(errors)
    revision = store.update_config_mapping(coerced, expected_revision=expected_revision)
    return revision, warnings


def set_config_value(key: str, value: Any) -> bool:
    """Set a single configuration value.

    Args:
        key: Configuration key name.
        value: New value.

    Returns:
        True if successful, False if the key or value is invalid.

    Raises:
        ConfigError: the file does not parse or another process holds the
            lock (see :mod:`EvoScientist.config.store`).
    """
    try:
        set_config_values({key: value})
    except SettingsError:
        return False
    return True


def list_config() -> dict[str, Any]:
    """List all configuration values.

    Returns:
        Dictionary of all configuration key-value pairs.
    """
    return _config_to_dict(load_config())


# =============================================================================
# Effective configuration (merging sources)
# =============================================================================

# Environment variable mappings
_ENV_MAPPINGS = {
    "anthropic_api_key": "ANTHROPIC_API_KEY",
    "anthropic_base_url": "ANTHROPIC_BASE_URL",
    "anthropic_auth_mode": "EVOSCIENTIST_ANTHROPIC_AUTH_MODE",
    "openai_api_key": "OPENAI_API_KEY",
    "openai_auth_mode": "EVOSCIENTIST_OPENAI_AUTH_MODE",
    "nvidia_api_key": "NVIDIA_API_KEY",
    "google_api_key": "GOOGLE_API_KEY",
    "minimax_api_key": "MINIMAX_API_KEY",
    "minimax_base_url": "MINIMAX_BASE_URL",
    "siliconflow_api_key": "SILICONFLOW_API_KEY",
    "openrouter_api_key": "OPENROUTER_API_KEY",
    "atlascloud_api_key": "ATLASCLOUD_API_KEY",
    "requesty_api_key": "REQUESTY_API_KEY",
    "novita_api_key": "NOVITA_API_KEY",
    "mimo_api_key": "MIMO_API_KEY",
    "mimo_token_plan_api_key": "MIMO_TOKEN_PLAN_API_KEY",
    "mimo_token_plan_base_url": "MIMO_TOKEN_PLAN_BASE_URL",
    "opper_api_key": "OPPER_API_KEY",
    "deepseek_api_key": "DEEPSEEK_API_KEY",
    "zhipu_api_key": "ZHIPU_API_KEY",
    "volcengine_api_key": "VOLCENGINE_API_KEY",
    "dashscope_api_key": "DASHSCOPE_API_KEY",
    "moonshot_api_key": "MOONSHOT_API_KEY",
    "kimi_api_key": "KIMI_API_KEY",
    "custom_openai_api_key": "CUSTOM_OPENAI_API_KEY",
    "custom_openai_base_url": "CUSTOM_OPENAI_BASE_URL",
    "custom_anthropic_api_key": "CUSTOM_ANTHROPIC_API_KEY",
    "custom_anthropic_base_url": "CUSTOM_ANTHROPIC_BASE_URL",
    "ollama_base_url": "OLLAMA_BASE_URL",
    "tavily_api_key": "TAVILY_API_KEY",
    "default_mode": "EVOSCIENTIST_DEFAULT_MODE",
    "default_workdir": "EVOSCIENTIST_WORKSPACE_DIR",
    "ui_backend": "EVOSCIENTIST_UI_BACKEND",
    "log_level": "EVOSCIENTIST_LOG_LEVEL",
    "gateway_backend": "EVOSCIENTIST_GATEWAY_BACKEND",
    "gateway_backend_serve": "EVOSCIENTIST_GATEWAY_BACKEND_SERVE",
    "gateway_backend_single_shot": "EVOSCIENTIST_GATEWAY_BACKEND_SINGLE_SHOT",
    "gateway_backend_interactive": "EVOSCIENTIST_GATEWAY_BACKEND_INTERACTIVE",
    "gateway_backend_tui": "EVOSCIENTIST_GATEWAY_BACKEND_TUI",
    "gateway_backend_standalone": "EVOSCIENTIST_GATEWAY_BACKEND_STANDALONE",
    "proactive_idle_minutes": "EVOSCIENTIST_PROACTIVE_IDLE_MINUTES",
    "proactive_quiet_hours": "EVOSCIENTIST_PROACTIVE_QUIET_HOURS",
    "proactive_timezone": "EVOSCIENTIST_PROACTIVE_TIMEZONE",
    "model_fallbacks": "EVOSCIENTIST_MODEL_FALLBACKS",
    "model_retry_wait_seconds": "EVOSCIENTIST_MODEL_RETRY_WAIT_SECONDS",
    "auto_mode_retry_wait_seconds": "EVOSCIENTIST_AUTO_MODE_RETRY_WAIT_SECONDS",
    "auxiliary_provider": "EVOSCIENTIST_AUXILIARY_PROVIDER",
    "auxiliary_model": "EVOSCIENTIST_AUXILIARY_MODEL",
    "reasoning_effort": "EVOSCIENTIST_REASONING_EFFORT",
    "openrouter_anthropic_prompt_cache": (
        "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE"
    ),
    "openrouter_http_referer": "EVOSCIENTIST_OPENROUTER_HTTP_REFERER",
    "openrouter_app_title": "EVOSCIENTIST_OPENROUTER_APP_TITLE",
    "openrouter_app_categories": "EVOSCIENTIST_OPENROUTER_APP_CATEGORIES",
    "dangerous_mode": "EVOSCIENTIST_DANGEROUS_MODE",
    "channel_debug_tracing": "EVOSCIENTIST_CHANNEL_DEBUG_TRACING",
    "ccproxy_port": "EVOSCIENTIST_CCPROXY_PORT",
    "use_responses_api": "EVOSCIENTIST_USE_RESPONSES_API",
    "checkpoint_keep_per_thread": "EVOSCIENTIST_CHECKPOINT_KEEP_PER_THREAD",
    "enable_async_subagents": "EVOSCIENTIST_ENABLE_ASYNC_SUBAGENTS",
    "langgraph_dev_port": "EVOSCIENTIST_LANGGRAPH_DEV_PORT",
    "langgraph_dev_host": "EVOSCIENTIST_LANGGRAPH_DEV_HOST",
    "webui_port": "EVOSCIENTIST_WEBUI_PORT",
    "webui_host": "EVOSCIENTIST_WEBUI_HOST",
    "enable_scheduler": "EVOSCIENTIST_ENABLE_SCHEDULER",
    "scheduler_default_timezone": "EVOSCIENTIST_SCHEDULER_DEFAULT_TIMEZONE",
    "code_interpreter_timeout": "EVOSCIENTIST_CODE_INTERPRETER_TIMEOUT",
    "code_interpreter_max_result_chars": "EVOSCIENTIST_CODE_INTERPRETER_MAX_RESULT_CHARS",
    "sandbox_execute_timeout": "EVOSCIENTIST_SANDBOX_EXECUTE_TIMEOUT",
    "langgraph_dev_file_persistence": "EVOSCIENTIST_LANGGRAPH_DEV_FILE_PERSISTENCE",
    "langgraph_dev_jobs_per_worker": "EVOSCIENTIST_LANGGRAPH_DEV_JOBS_PER_WORKER",
    "langgraph_dev_keepalive": "EVOSCIENTIST_LANGGRAPH_DEV_KEEPALIVE",
    "recursion_limit": "EVOSCIENTIST_RECURSION_LIMIT",
    "memory_profile_enabled": "EVOSCIENTIST_MEMORY_PROFILE_ENABLED",
    "memory_observations_enabled": "EVOSCIENTIST_MEMORY_OBSERVATIONS_ENABLED",
    "memory_observation_writer": "EVOSCIENTIST_MEMORY_OBSERVATION_WRITER",
    "memory_workers_enabled": "EVOSCIENTIST_MEMORY_WORKERS_ENABLED",
    "memory_skill_synthesis_enabled": "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_ENABLED",
    "memory_skill_synthesis_mode": "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_MODE",
    "memory_skill_synthesis_cadence": "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_CADENCE",
    "memory_skill_synthesis_time": "EVOSCIENTIST_MEMORY_SKILL_SYNTHESIS_TIME",
    "memory_observation_cache_max_files": "EVOSCIENTIST_MAX_CACHED_FILES",
}


_ENV_FIELDS: dict[str, str] = {env: field for field, env in _ENV_MAPPINGS.items()}


# =============================================================================
# Where environment values came from
# =============================================================================

# The process environment cannot tell the user's own exports from values our
# code put there: apply_config_to_env() copies saved keys, get_effective_config()
# merges a ``.env``, flags and ccproxy write variables too. Each such write is
# recorded here, by variable name: ``{"source": "file" | "default" | "cli" |
# "ccproxy" | "dotenv", "path"?: <.env path>}``. A variable with no record is
# the user's (``env``). The settings API reports it as ``sources``.
#
# The launcher passes its record to the langgraph dev server in
# ENV_ORIGINS_VAR (server environment only); this module reads and removes it
# at import, so nothing the server starts inherits it.
ENV_ORIGINS_VAR = "EVOSCIENTIST_CONFIG_ORIGINS"
_ENV_ORIGINS: dict[str, dict[str, str]] = {}


def _load_env_origins() -> None:
    raw = os.environ.pop(ENV_ORIGINS_VAR, None)
    if not raw:
        return
    try:
        data = json.loads(raw)
    except ValueError:
        return
    if isinstance(data, dict):
        _ENV_ORIGINS.update(
            {
                str(k): {str(a): str(b) for a, b in v.items()}
                for k, v in data.items()
                if isinstance(v, dict)
            }
        )


_load_env_origins()


def record_env_origin(env_key: str, source: str, path: str | None = None) -> None:
    """Record that our code set ``env_key``, and from where."""
    origin = {"source": source}
    if path:
        origin["path"] = path
    _ENV_ORIGINS[env_key] = origin


def env_origin(env_key: str) -> dict[str, str] | None:
    """Where ``env_key`` came from, or ``None`` when our code did not set it."""
    return _ENV_ORIGINS.get(env_key)


def env_origins() -> dict[str, dict[str, str]]:
    return {key: dict(value) for key, value in _ENV_ORIGINS.items()}


def _same_value(field: str, a: Any, b: Any) -> bool:
    if field == "default_workdir":
        a, b = (
            os.path.realpath(os.path.expanduser(str(v))) if v else "" for v in (a, b)
        )
    return str(_plain_config_value(a)) == str(_plain_config_value(b))


def _saved_mapping() -> dict[str, Any]:
    """The file's mapping, or ``{}`` when it does not parse or cannot be read."""
    try:
        return store.read_config_mapping()
    except (store.ConfigError, OSError):
        return {}


def config_value_origin(
    field: str, value: Any, mapping: dict[str, Any] | None = None
) -> str:
    """``file`` when the file holds ``value`` for ``field``, ``default`` when
    the file lacks the field and ``value`` is its default, else ``cli``: the
    value can then only have come from a command-line flag.

    ``mapping`` is the file's mapping when the caller has already read it.
    """
    if mapping is None:
        mapping = _saved_mapping()
    saved = mapping.get(field)
    if saved is not None and saved != "":
        return "file" if _same_value(field, saved, value) else "cli"
    default = next(f.default for f in fields(EvoScientistConfig) if f.name == field)
    if field == "default_workdir":
        # An empty default_workdir means "the current folder".
        return "default" if _same_value(field, os.getcwd(), value) else "cli"
    return "default" if _same_value(field, default, value) else "cli"


# Placeholder ``ccproxy_manager.setup_ccproxy_env`` writes for the Anthropic API
# key when routing through ccproxy OAuth; the matching ``*_BASE_URL`` rides along.
_OAUTH_PLACEHOLDER_KEY = "ccproxy-oauth"


def _oauth_routed(env_key: str) -> bool:
    """Return True when ``env_key`` currently carries ccproxy OAuth routing."""
    env_key = env_key.upper()  # Windows env keys are case-insensitive
    if env_key.endswith("_BASE_URL"):
        env_key = env_key[: -len("_BASE_URL")] + "_API_KEY"
    elif not env_key.endswith("_API_KEY"):
        return False
    return os.environ.get(env_key) == _OAUTH_PLACEHOLDER_KEY


def get_effective_config(
    cli_overrides: dict[str, Any] | None = None,
) -> EvoScientistConfig:
    """Get effective configuration by merging all sources.

    Priority (highest to lowest):
        1. CLI arguments (``cli_overrides``)
        2. Parent-process environment variables for any ``EVOSCIENTIST_*`` key
        3. ``.env`` file at (or above) the current working directory
        4. Parent-process environment variables for everything else
           (third-party API keys / base URLs, plus arbitrary unmapped keys)
        5. Config file (``~/.config/evoscientist/config.yaml``)
        6. Dataclass defaults

    Rows 2 and 4 differ because ``.env`` values need different treatment
    for our own namespaced config knobs vs third-party credentials.
    Third-party keys (``ANTHROPIC_API_KEY``, ``OPENAI_API_KEY``, ...)
    follow the industry convention that ``.env`` is the per-project
    credential store; extending shell-wins to them would silently flip
    a workspace key back to a global ``.bashrc`` key. Our own
    ``EVOSCIENTIST_*`` keys are the opposite: an explicit CLI/parent-
    process value (e.g. the bind port that ``EvoSci deploy --port X``
    hands to the langgraph dev subprocess) must not be shadowed by a
    workspace ``.env``. We implement this by reading ``.env`` into a
    dict via ``dotenv_values`` (no ``os.environ`` mutation), then
    writing third-party keys unconditionally and ``EVOSCIENTIST_*`` keys
    only when the shell doesn't already have a non-empty value.

    Tradeoff: ``OPENAI_API_KEY=xxx evoscientist ...`` inline overrides
    still lose to a workspace ``.env`` containing ``OPENAI_API_KEY``,
    because the merge writes third-party keys from ``.env``
    unconditionally. Users who need to override a ``.env``-defined
    credential inline must edit or unset the ``.env`` entry.

    Args:
        cli_overrides: Dictionary of CLI argument overrides.

    Returns:
        EvoScientistConfig with merged values.
    """
    # Merge workspace ``.env`` into ``os.environ`` without going through
    # ``load_dotenv``. The previous snapshot → ``load_dotenv`` → restore
    # sequence was a read-modify-write on ``os.environ`` that could race with
    # concurrent ``get_effective_config`` calls in the langgraph dev subprocess
    # (per-request threads in ``langgraph_dev/http.py``, ``sessions.py``
    # checkpoint writes, memory workers): one thread's mid-flight ``.env``
    # value could be re-captured by another as "parent env" and then restored
    # last, promoting the ``.env`` value into the snapshot permanently.
    #
    # ``dotenv_values`` returns a dict without touching ``os.environ``, so the
    # merge below is a pure write sequence and idempotent under interleaving.
    # Third-party keys keep ``.env``-wins (industry convention).
    # ``EVOSCIENTIST_*`` keys are our own namespaced config knobs where
    # CLI/parent-process intent should stay authoritative — write from ``.env``
    # only when the shell doesn't already have a non-empty value. Treating an
    # empty shell value as "unset" matches the ``if env_value:`` truthy check
    # in the ``_ENV_MAPPINGS`` loop below; without this, an empty parent export
    # would silently regress vs main by falling through to file/defaults.
    dotenv_path = find_dotenv(usecwd=True)
    dotenv_map = dotenv_values(dotenv_path) if dotenv_path else {}
    for env_key, env_value in dotenv_map.items():
        if env_value is None:
            continue  # bare ``FOO`` without ``=`` — nothing to write
        if env_key.startswith("EVOSCIENTIST_"):
            if not os.environ.get(env_key):
                os.environ[env_key] = env_value
                record_env_origin(env_key, "dotenv", dotenv_path)
        elif _oauth_routed(env_key):
            continue  # ccproxy OAuth routing must survive later re-merges
        else:
            os.environ[env_key] = env_value
            record_env_origin(env_key, "dotenv", dotenv_path)

    # Start with file config (includes defaults for missing values)
    config = load_config()
    data = _config_to_dict(config)

    # Apply environment variable overrides
    for config_key, env_key in _ENV_MAPPINGS.items():
        env_value = os.environ.get(env_key)
        if env_value:
            field_info = next(
                f for f in fields(EvoScientistConfig) if f.name == config_key
            )
            try:
                data[config_key] = _coerce_value(
                    env_value,
                    _config_field_type(config_key, field_info.type),
                )
            except (ValueError, TypeError):
                pass

    # Apply CLI overrides (highest priority)
    if cli_overrides:
        for key, value in cli_overrides.items():
            if value is not None and key in data:
                data[key] = value

    return EvoScientistConfig(**data)


def apply_config_to_env(config: EvoScientistConfig) -> None:
    """Apply config API keys to environment variables if not already set.

    This allows the config file to provide API keys that downstream
    libraries (like langchain-anthropic) can pick up. Every variable it
    sets is recorded as coming from the file, the defaults or a flag (see
    :func:`record_env_origin`).

    Args:
        config: Configuration to apply.
    """
    before = dict(os.environ)
    _apply_config_to_env(config)
    mapping = None  # read once, when the first variable needs it
    for env_key, field in _ENV_FIELDS.items():
        value = os.environ.get(env_key)
        if value is None:
            _ENV_ORIGINS.pop(env_key, None)
        elif value != before.get(env_key):
            if mapping is None:
                mapping = _saved_mapping()
            record_env_origin(
                env_key, config_value_origin(field, getattr(config, field), mapping)
            )


def _apply_config_to_env(config: EvoScientistConfig) -> None:
    if config.anthropic_api_key and not os.environ.get("ANTHROPIC_API_KEY"):
        os.environ["ANTHROPIC_API_KEY"] = config.anthropic_api_key
    if config.anthropic_base_url and not os.environ.get("ANTHROPIC_BASE_URL"):
        os.environ["ANTHROPIC_BASE_URL"] = config.anthropic_base_url
    if config.openai_api_key and not os.environ.get("OPENAI_API_KEY"):
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    if config.nvidia_api_key and not os.environ.get("NVIDIA_API_KEY"):
        os.environ["NVIDIA_API_KEY"] = config.nvidia_api_key
    if config.google_api_key and not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = config.google_api_key
    if config.minimax_api_key and not os.environ.get("MINIMAX_API_KEY"):
        os.environ["MINIMAX_API_KEY"] = config.minimax_api_key
    if config.minimax_base_url and not os.environ.get("MINIMAX_BASE_URL"):
        os.environ["MINIMAX_BASE_URL"] = config.minimax_base_url
    if config.siliconflow_api_key and not os.environ.get("SILICONFLOW_API_KEY"):
        os.environ["SILICONFLOW_API_KEY"] = config.siliconflow_api_key
    if config.openrouter_api_key and not os.environ.get("OPENROUTER_API_KEY"):
        os.environ["OPENROUTER_API_KEY"] = config.openrouter_api_key
    if config.atlascloud_api_key and not os.environ.get("ATLASCLOUD_API_KEY"):
        os.environ["ATLASCLOUD_API_KEY"] = config.atlascloud_api_key
    if config.requesty_api_key and not os.environ.get("REQUESTY_API_KEY"):
        os.environ["REQUESTY_API_KEY"] = config.requesty_api_key
    if config.novita_api_key and not os.environ.get("NOVITA_API_KEY"):
        os.environ["NOVITA_API_KEY"] = config.novita_api_key
    if config.mimo_api_key and not os.environ.get("MIMO_API_KEY"):
        os.environ["MIMO_API_KEY"] = config.mimo_api_key
    if config.mimo_token_plan_api_key and not os.environ.get("MIMO_TOKEN_PLAN_API_KEY"):
        os.environ["MIMO_TOKEN_PLAN_API_KEY"] = config.mimo_token_plan_api_key
    if config.mimo_token_plan_base_url and not os.environ.get(
        "MIMO_TOKEN_PLAN_BASE_URL"
    ):
        os.environ["MIMO_TOKEN_PLAN_BASE_URL"] = config.mimo_token_plan_base_url
    if config.opper_api_key and not os.environ.get("OPPER_API_KEY"):
        os.environ["OPPER_API_KEY"] = config.opper_api_key
    if config.deepseek_api_key and not os.environ.get("DEEPSEEK_API_KEY"):
        os.environ["DEEPSEEK_API_KEY"] = config.deepseek_api_key
    if config.zhipu_api_key and not os.environ.get("ZHIPU_API_KEY"):
        os.environ["ZHIPU_API_KEY"] = config.zhipu_api_key
    if config.volcengine_api_key and not os.environ.get("VOLCENGINE_API_KEY"):
        os.environ["VOLCENGINE_API_KEY"] = config.volcengine_api_key
    if config.dashscope_api_key and not os.environ.get("DASHSCOPE_API_KEY"):
        os.environ["DASHSCOPE_API_KEY"] = config.dashscope_api_key
    if config.moonshot_api_key and not os.environ.get("MOONSHOT_API_KEY"):
        os.environ["MOONSHOT_API_KEY"] = config.moonshot_api_key
    if config.kimi_api_key and not os.environ.get("KIMI_API_KEY"):
        os.environ["KIMI_API_KEY"] = config.kimi_api_key
    if config.custom_openai_api_key and not os.environ.get("CUSTOM_OPENAI_API_KEY"):
        os.environ["CUSTOM_OPENAI_API_KEY"] = config.custom_openai_api_key
    if config.custom_openai_base_url and not os.environ.get("CUSTOM_OPENAI_BASE_URL"):
        os.environ["CUSTOM_OPENAI_BASE_URL"] = config.custom_openai_base_url
    if config.custom_anthropic_api_key and not os.environ.get(
        "CUSTOM_ANTHROPIC_API_KEY"
    ):
        os.environ["CUSTOM_ANTHROPIC_API_KEY"] = config.custom_anthropic_api_key
    if config.custom_anthropic_base_url and not os.environ.get(
        "CUSTOM_ANTHROPIC_BASE_URL"
    ):
        os.environ["CUSTOM_ANTHROPIC_BASE_URL"] = config.custom_anthropic_base_url
    if config.ollama_base_url and not os.environ.get("OLLAMA_BASE_URL"):
        os.environ["OLLAMA_BASE_URL"] = config.ollama_base_url
    if config.tavily_api_key and not os.environ.get("TAVILY_API_KEY"):
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key
    if config.reasoning_effort and not os.environ.get("EVOSCIENTIST_REASONING_EFFORT"):
        os.environ["EVOSCIENTIST_REASONING_EFFORT"] = config.reasoning_effort
    if config.openrouter_http_referer and not os.environ.get(
        "EVOSCIENTIST_OPENROUTER_HTTP_REFERER"
    ):
        os.environ["EVOSCIENTIST_OPENROUTER_HTTP_REFERER"] = (
            config.openrouter_http_referer
        )
    if config.openrouter_app_title and not os.environ.get(
        "EVOSCIENTIST_OPENROUTER_APP_TITLE"
    ):
        os.environ["EVOSCIENTIST_OPENROUTER_APP_TITLE"] = config.openrouter_app_title
    if config.openrouter_app_categories and not os.environ.get(
        "EVOSCIENTIST_OPENROUTER_APP_CATEGORIES"
    ):
        os.environ["EVOSCIENTIST_OPENROUTER_APP_CATEGORIES"] = (
            config.openrouter_app_categories
        )
    if not config.openrouter_anthropic_prompt_cache and not os.environ.get(
        "EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE"
    ):
        os.environ["EVOSCIENTIST_OPENROUTER_ANTHROPIC_PROMPT_CACHE"] = "false"
    # Written for both values: get_chat_model reads it, and the langgraph dev
    # subprocess re-resolves config without the CLI's --auth-mode override.
    os.environ["EVOSCIENTIST_OPENAI_AUTH_MODE"] = config.openai_auth_mode
    # Round-trip dangerous_mode to env so it survives a fresh get_effective_config()
    # (warning banner, run_in_background) and is inherited by the langgraph dev
    # subprocess — otherwise a --dangerous CLI flag (not persisted to file/env)
    # is invisible to those consumers while the backend is already unconfined.
    # Bidirectional: clear it when off so a re-apply with a lower config (or a
    # stale value) can't leave the process stuck in dangerous mode.
    if config.dangerous_mode:
        os.environ["EVOSCIENTIST_DANGEROUS_MODE"] = "true"
    else:
        os.environ.pop("EVOSCIENTIST_DANGEROUS_MODE", None)
    if config.use_responses_api and not os.environ.get(
        "EVOSCIENTIST_USE_RESPONSES_API"
    ):
        os.environ["EVOSCIENTIST_USE_RESPONSES_API"] = config.use_responses_api
