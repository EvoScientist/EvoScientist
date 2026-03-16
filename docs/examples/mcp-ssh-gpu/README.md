# Remote SSH GPU Execution with MCP

This guide explains how to configure EvoScientist for remote GPU experiment execution using SSH MCP servers.

## Overview

Instead of building a custom `RemoteSSHBackend`, EvoScientist leverages the existing MCP infrastructure to enable remote GPU execution. By configuring an SSH MCP server, your `code-agent` and `debug-agent` gain access to SSH tools (`ssh_execute`, `ssh_upload`, `ssh_download`) that allow them to:

- Execute GPU-dependent commands (training, inference) on remote servers
- Sync experiment code to remote servers
- Retrieve results and artifacts
- Check remote GPU status via `nvidia-smi`
- Run long-lived jobs using `screen`/`tmux`

## Prerequisites

1. **SSH MCP Server**: Install an SSH MCP server. Recommended options:
   - [`mcp-server-ssh`](https://github.com/modelcontextprotocol/servers/tree/main/src/ssh) (Node.js-based)
   - Any other SSH-capable MCP server that provides `ssh_execute`, `ssh_upload`, `ssh_download` tools

2. **SSH Key Authentication**: Set up SSH key-based authentication to your remote GPU server

3. **Remote Environment**: Ensure your remote server has:
   - CUDA drivers installed
   - Required Python packages (or conda environment)
   - Sufficient disk space for experiments

## Configuration

### Step 1: Install SSH MCP Server

For Node.js-based SSH server:

```bash
# Install globally
npm install -g mcp-server-ssh

# Or use npx (no installation required)
npx -y mcp-server-ssh --help
```

### Step 2: Create MCP Configuration

Create or edit `~/.config/evoscientist/mcp.yaml`:

```yaml
ssh-gpu:
  transport: stdio
  command: npx
  args: ["-y", "mcp-server-ssh"]
  env:
    SSH_HOST: "your-gpu-server.example.com"
    SSH_USER: "your-username"
    SSH_KEY_PATH: "~/.ssh/id_rsa"
    # Optional: SSH port (default: 22)
    # SSH_PORT: "2222"
  expose_to: [code-agent, debug-agent]
```

### Step 3: Verify Configuration

Add the SSH MCP server via CLI:

```bash
EvoSci mcp add ssh-gpu npx -- -y mcp-server-ssh \
  --transport stdio \
  --tools ssh_execute,ssh_upload,ssh_download
```

Or start an agent session and use the in-session command:

```bash
/mcp add ssh-gpu npx -- -y mcp-server-ssh
```

## Usage

### Remote Experiment Execution

When SSH MCP tools are available, the `code-agent` will automatically:

1. **Sync code to remote server**:
   ```
   Use ssh_upload to transfer experiment files
   ```

2. **Execute GPU-dependent commands**:
   ```
   Use ssh_execute to run:
   - nvidia-smi (check GPU status)
   - python train.py (run training)
   - python inference.py (run inference)
   ```

3. **Handle long-running jobs**:
   ```
   Consider using screen/tmux via ssh_execute:
   ssh_execute "screen -dmS experiment python train.py"
   ```

4. **Retrieve results**:
   ```
   Use ssh_download to pull results/artifacts
   ```

### Remote Debugging

The `debug-agent` will use SSH tools to:

1. Reproduce failures on the remote server
2. Check remote environment (CUDA version, package versions)
3. Retrieve remote logs for analysis

## Example Workflow

```bash
# 1. Start agent session
EvoScientist

# 2. Add SSH MCP server
/mcp add ssh-gpu npx -- -y mcp-server-ssh

# 3. Task: Run training experiment
"Run training on remote GPU server with dataset X"

# Agent will:
# - Upload code via ssh_upload
# - Execute via ssh_execute
# - Monitor progress
# - Download results via ssh_download
```

## Troubleshooting

### SSH Connection Fails

1. Verify SSH key is correct and has proper permissions:
   ```bash
   chmod 600 ~/.ssh/id_rsa
   ssh-add ~/.ssh/id_rsa
   ```

2. Test manual SSH connection:
   ```bash
   ssh your-username@your-gpu-server.example.com
   ```

### SSH MCP Tools Not Available

1. Check `mcp.yaml` configuration
2. Verify `expose_to` includes `code-agent` and/or `debug-agent`
3. Reload agent session: `/new` or restart CLI

### Remote Commands Hang

Use `screen` or `tmux` for long-running jobs:

```bash
# Start detached screen session
ssh_execute "screen -dmS train python train.py"

# Attach later to check status
ssh_execute "screen -r train"
```

## Security Considerations

- **SSH Keys**: Never commit SSH private keys to repositories
- **Environment Variables**: Use secure methods to manage SSH credentials
- **Network**: Consider using VPN or bastion hosts for production deployments

## Backward Compatibility

When no SSH MCP server is configured:
- Agents execute experiments locally as before
- No changes to existing behavior
- No new dependencies required

## Alternative SSH MCP Servers

| Server | Language | Tools | Notes |
|--------|----------|-------|-------|
| `mcp-server-ssh` | Node.js | ssh_execute, ssh_upload, ssh_download | Recommended, well-maintained |
| `mcp-server-shell` | Python | shell_execute | Limited to shell commands |
| Custom | Any | Varies | Can implement custom SSH logic |

## Further Reading

- [MCP Documentation](https://modelcontextprotocol.io/)
- [MCP Server Directory](https://github.com/modelcontextprotocol/servers)
- [SSH Server Implementation](https://github.com/modelcontextprotocol/servers/tree/main/src/ssh)
