# Security Policy

## Trust Model

EvoScientist is a **single-user local tool**. Its background server (port `6174`) and WebUI (port `4716`) assume anyone who can reach them *is* the machine's operator—so the WebUI, CLI, SDK clients, and LangGraph Studio all act with your local user permissions.

Keep these five boundaries in mind:

1. **Approval settings apply globally to the operator's runs.** Setting `auto_approve`, `auto_mode`, or `dangerous_mode` disables interactive confirmation for shell commands, deletions, background processes, and scheduled tasks—even if untrusted content the model reads tries to steer it. That trade-off is explicit when you disable approvals.
2. **Hard-blocked commands vs. dangerous mode.** Privileged/destructive commands (`sudo`, `chmod`, `chown`, `mkfs`, `dd`, `shutdown`, `reboot`, `pkill`, `killall`) and catastrophic deletions (`rm -rf /`, `rm -rf ~`) are **always blocked**, even in `dangerous_mode`. What `dangerous_mode` changes is lifting workspace confinement (allowing real filesystem paths), skipping heuristic dangerous-command checks (such as piping into an interpreter), and implying `auto_approve`.
3. **The WebUI cannot weaken safety settings.** The settings API refuses writes to `auto_approve`, `auto_mode`, `dangerous_mode`, `shell_allow_list`, bind hosts, `backend_allowed_origins`, and channel access lists. Change those only from your own terminal (`EvoSci config set`, `EvoSci onboard`, CLI flags, env vars) or `~/.config/evoscientist/config.yaml`. *(Note: if approvals are already off, an agent running shell commands as you can still edit `config.yaml` directly.)*
4. **Exposing `--host` is for trusted networks only.** Binding beyond `127.0.0.1` exposes an unauthenticated API that can execute shell commands. Only widen it on a private network you control (or tunnel over SSH instead—see [README.md](README.md#opening-from-another-machine)).
5. **One user per machine.** Loopback (`127.0.0.1`) blocks the network, not other local OS accounts. On a shared server, isolate each person in their own VM or container, or place authentication in front of the service.

Behavior that follows directly from this model (such as a command running without a prompt when approvals are turned off) is expected, not a vulnerability.

## Supported Versions

Security fixes ship in the latest release. Update with:

```bash
EvoSci update
```

## Reporting a Vulnerability

Please report vulnerabilities privately via GitHub's [Private Vulnerability Reporting](https://github.com/EvoScientist/EvoScientist/security/advisories/new) (**Security** tab → **Report a vulnerability**). Do not open a public issue.
