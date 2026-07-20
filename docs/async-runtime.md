# Async runtime ownership

EvoScientist historically mixed top-level event-loop ownership, temporary
loops, cross-thread scheduling, and global `nest_asyncio` patches. Those are
different mechanisms and should not be migrated mechanically.

The target is not literally one event loop per process. The target is one
explicit owner for application async work, while frameworks and transports
that must own their loop keep doing so. Incidental synchronous callers submit
coroutine factories to `AsyncRuntime`; async callers on another loop await the
same runtime without blocking their frontend.

## Runtime contract

- The application creates and closes `AsyncRuntime`; there is no module
  singleton and callers cannot obtain its raw loop.
- `run_sync` is for bounded synchronous-to-async calls and rejects callers
  already running an event loop.
- `run_async` bridges from a foreign async loop without blocking it.
- `spawn` is for work whose lifetime is owned by the application runtime.
- APIs accept coroutine factories so loop-bound objects are created on the
  owned loop.
- Shutdown seals intake, cancels and drains owned tasks, finalizes async
  generators, stops the loop, and joins its thread.

## Ownership map

| Owner | Lifetime | Boundary |
| --- | --- | --- |
| CLI `AsyncRuntime` | Created once by the CLI callback and closed after the selected mode exits | Application work submitted by synchronous commands, serve workers, MCP discovery, and Rich streaming |
| Scoped `AsyncRuntime` | Created and closed by a bounded API only when no application owner was supplied | Library/onboarding calls that need a synchronous facade and return no loop-bound resources |
| Prompt-toolkit and Textual frontend loops | Created at their synchronous UI entry points and closed when the frontend exits | Input, widgets, signals, and UI tasks; synchronous Rich rendering is offloaded before it submits work to `AsyncRuntime` |
| Channel bus loop | Created by `_start_channels_bus_mode` and closed after channel manager and inbound-consumer shutdown | Channel transports and outbound delivery; cross-thread callers explicitly schedule onto this loop |
| Feishu SDK loop | Created in the vendor WebSocket thread | Required by `lark-oapi`, which stores and drives its own loop |
| Standalone channel/WeChat command loops | Created with `asyncio.run` at process or command entry points | Top-level execution only; never used as nested bridges |

The application runtime is therefore shared, not universal. Framework and
transport loops remain separate where thread affinity or lifecycle ownership
requires it.

## Bridge inventory

| Area | Current role | Treatment |
| --- | --- | --- |
| `cli/commands.py` session statistics | Bounded database query from a synchronous command | Migrated to the CLI-owned runtime |
| `config/onboard/channels.py` login and credential probes | Bounded network calls returning plain data | Migrated to the CLI-owned runtime; direct callers get a runtime scoped to the channel step |
| `mcp/client.py` synchronous tool loading | Discovers tool adapters on the supplied runtime; direct callers receive a scoped runtime | Migrated after confirming discovery does not retain a client session or discovery-loop resource |
| `middleware/model_fallback.py` synchronous fallback | Traverses the fallback policy synchronously | Migrated without an async bridge; async middleware retains its native async traversal |
| `channels/base.py` synchronous inbound conversion | Uses the supplied runtime; async callers await the native async method | Migrated; running-loop callers are rejected by the sync facade instead of being deadlocked |
| `stream/display.py`, CLI single-shot, and Rich interactive paths | Rich owns synchronous terminal rendering while stream I/O runs on the application runtime | Migrated; frontend callers offload rendering, propagate cancellation, and wait for renderer cleanup |
| `cli/commands.py` serve slash dispatch and notification drain | Serve workers share the CLI-owned runtime | Migrated without making serve the runtime owner |
| `cli/channel.py` and `commands/channel_ui.py` | Dedicated channel-bus loop plus cross-thread scheduling onto that loop | Intentional owner/bridge. Keep until a channel lifecycle redesign can preserve transport affinity and shutdown ordering |
| `cli/tui_interactive.py` | Textual owns the frontend loop | Intentional owner. Use `run_async` for application work, but do not move Textual's loop under `AsyncRuntime` |
| `channels/standalone.py` and standalone WeChat login | Top-level async process/command entry points | `asyncio.run` is legitimate here; no nested bridge to remove |
| `channels/feishu/channel.py` | SDK-specific thread and loop required by the vendor client | Preserve until that adapter is redesigned and tested against the SDK lifecycle |
| Probe/RPC executor and future creation inside async functions | Operations on the already-running owner loop | Use `get_running_loop()`; these are not sync/async bridges |

`asyncio.run_coroutine_threadsafe` is not automatically a defect: it is the
correct primitive when code deliberately targets an already-owned foreign
loop, such as the channel bus. The defect is hidden or accidental ownership,
especially temporary loops and global re-entrancy patches.

## Remaining design boundary

The channel bus remains a separate explicit owner. Folding it into the
application runtime would be a channel lifecycle redesign, not bridge cleanup:
transport affinity, startup reporting, outbound ordering, health services, and
shutdown would all need to move together. Keeping that owner is consistent
with the contract as long as all crossings target it explicitly.

New code should not introduce `nest_asyncio`, a temporary
`run_until_complete`, or a nested `asyncio.run`. Any exception should identify
the top-level framework or transport that owns the loop and document its
shutdown boundary.
