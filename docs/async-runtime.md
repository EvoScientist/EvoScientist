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

## Bridge inventory

| Area | Current role | Treatment |
| --- | --- | --- |
| `cli/commands.py` session statistics | Bounded database query from a synchronous command | Migrated to the CLI-owned runtime |
| `config/onboard/channels.py` login and credential probes | Bounded network calls returning plain data | Migrated to the CLI-owned runtime; direct callers get a runtime scoped to the channel step |
| `mcp/client.py` synchronous tool loading | Uses `asyncio.run` and globally patches a running caller loop | Migrate next only after confirming the returned tool proxies do not retain loop-bound sessions, or keep their sessions owned by the runtime |
| `middleware/model_fallback.py` synchronous fallback | Runs async fallback policy from synchronous middleware | Inject runtime through agent construction or separate the synchronous policy; do not add a singleton |
| `channels/base.py` synchronous inbound conversion | Schedules on a known channel loop, otherwise creates a temporary loop | Preserve known-loop scheduling; replace the temporary fallback when channel runtime ownership is available at the adapter boundary |
| `stream/display.py`, CLI single-shot, and Rich interactive paths | Rendering, refresh tasks, signals, and nested UI entry points | High risk; migrate after the bounded sites and test cancellation, terminal restoration, and interrupt behavior together |
| `cli/commands.py` serve slash dispatch and notification drain | Temporary loops inside a signal-sensitive synchronous poll loop | High risk; serve is a consumer, not the runtime architecture. Address after its loop/resource ownership is explicit |
| `cli/channel.py` and `commands/channel_ui.py` | Dedicated channel-bus loop plus cross-thread scheduling onto that loop | Intentional owner/bridge. Keep until a channel lifecycle redesign can preserve transport affinity and shutdown ordering |
| `cli/tui_interactive.py` | Textual owns the frontend loop | Intentional owner. Use `run_async` for application work, but do not move Textual's loop under `AsyncRuntime` |
| `channels/standalone.py` and standalone WeChat login | Top-level async process/command entry points | `asyncio.run` is legitimate here; no nested bridge to remove |
| `channels/feishu/channel.py` | SDK-specific thread and loop required by the vendor client | Preserve until that adapter is redesigned and tested against the SDK lifecycle |
| Probe/RPC calls to `get_event_loop().run_in_executor()` or `.create_future()` from async functions | Operations on the already-running owner loop | Not sync/async bridges. They can independently move to `get_running_loop()` or `asyncio.to_thread()` |

`asyncio.run_coroutine_threadsafe` is not automatically a defect: it is the
correct primitive when code deliberately targets an already-owned foreign
loop, such as the channel bus. The defect is hidden or accidental ownership,
especially temporary loops and global re-entrancy patches.

## Migration order

1. Continue with bounded operations that return plain values and own no
   background resources.
2. Handle MCP loading after establishing the lifetime of returned tool
   clients and subprocesses.
3. Inject runtime ownership into synchronous middleware and channel adapter
   boundaries.
4. Migrate display, Rich, single-shot, and serve paths with scenario tests for
   signals, cancellation, terminal cleanup, and channel coexistence.
5. Revisit whether the channel bus remains a separate explicit owner. A
   separate loop is acceptable if its ownership and shutdown contract remain
   deliberate.

New code should not introduce `nest_asyncio`, a temporary
`run_until_complete`, or a nested `asyncio.run`. Any exception should identify
the top-level framework or transport that owns the loop and document its
shutdown boundary.
