# Security Policy

## Trust model

EvoScientist runs on one person's machine, for that person. Its background server (the backend on port 6174) and the WebUI (port 4716) treat anyone who can reach them as their operator: the WebUI, SDK clients, LangGraph Studio and the `EvoSci` command all act with the operator's rights.

- **Approval settings belong to the operator.** `auto_approve`, `auto_mode` and `dangerous_mode` apply to every run that does not bring its own choice. Runs from the WebUI, SDK clients and Studio follow the server's setting, as the terminal follows its own. With approvals off, the agent's shell commands, deletions, background processes and scheduled tasks run without a prompt, including when content the model reads steers them; that is the risk the setting accepts.
- **The command guard stays on.** Blocked commands (`sudo`, `chmod`, `pkill` and others) and catastrophic deletions (such as `rm -rf ~`) are refused in every mode. `dangerous_mode` turns every approval off (it implies `auto_approve`) and lifts the workspace sandbox and the dangerous-command check.
- **The WebUI's settings page cannot change protection settings.** The settings API refuses `auto_approve`, `auto_mode`, `dangerous_mode`, `shell_allow_list`, the bind hosts, the allowed browser origins and the channel access settings; they change from the operator's own terminal (`EvoSci config set`, `EvoSci onboard`, command-line flags, environment variables) or in `config.yaml`. With approvals off, an agent run from any client runs commands as the operator, so it can change that file too.
- **Other machines are the operator's own.** A server opened to other machines (`--host`) is meant for the operator's own devices on a network the operator trusts; see "Opening it from another machine" in the README.
- **One user per machine.** Loopback keeps the servers off the network, not away from other accounts on the same machine. EvoScientist does not separate users: for a shared machine or a team, give each person a machine, VM or container of their own, or put authentication and isolation in front of the service.

Behaviour that follows from this model, such as a run that executes without approval on a server with approvals off, is expected.

## Supported versions

Fixes go into the next release. Earlier releases are not patched; update with `EvoSci update`.

## Reporting a vulnerability

Please report security issues privately through GitHub's [private vulnerability reporting](https://github.com/EvoScientist/EvoScientist/security/advisories/new) (the repository's **Security** tab, **Report a vulnerability**), not in a public issue.
