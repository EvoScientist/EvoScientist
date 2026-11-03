"""Runs ``EvoSci update --version <target> --yes --json --skip webui`` for the
installer smoke workflow and checks its output: exactly one terminal
``update`` line, the last one, with ``done``, then ``EvoSci --version``
naming the target.

Run it with a Python outside the tool environment (uv's base Python): on
Windows the update waits until nothing runs from that environment. It reads
stdout to its end, because there the helper writes the last lines after
EvoSci has exited. stdin is NUL, so the update does not wait for anything
it finds running but stops at once.

    python install_smoke_update.py <EvoSci> <target version>
"""

import json
import subprocess
import sys


def fail(message: str) -> None:
    print(f"::error::{message}")
    sys.exit(1)


def main() -> None:
    evosci, target = sys.argv[1:3]
    cmd = [evosci, "update", "--version", target, "--yes", "--json", "--skip", "webui"]
    print("+ " + " ".join(cmd), flush=True)
    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        text=True,
        encoding="utf-8",
    )
    assert proc.stdout is not None
    lines = []
    for line in proc.stdout:
        print(line, end="", flush=True)
        lines.append(line)
    proc.wait()
    try:
        events = [json.loads(line) for line in lines if line.strip()]
    except ValueError as exc:
        fail(f"EvoSci update --json wrote a line that is not JSON: {exc}")
    ends = [
        e
        for e in events
        if e.get("stage") == "update" and e.get("status") in ("done", "error")
    ]
    if len(ends) != 1 or events[-1] is not ends[0]:
        fail(f"expected one terminal update line, the last one; got {len(ends)}")
    if ends[0]["status"] != "done":
        fail(
            f"EvoSci update ended with {ends[0].get('code')}: {ends[0].get('message')}"
        )
    out = subprocess.run(
        [evosci, "--version", "--json"],
        capture_output=True,
        text=True,
        encoding="utf-8",
    ).stdout
    version = json.loads(out)["version"]
    if version != target:
        fail(f"EvoSci --version after the update reports {version}, not {target}")
    print(f"EvoSci update to {target}: ok")


if __name__ == "__main__":
    main()
