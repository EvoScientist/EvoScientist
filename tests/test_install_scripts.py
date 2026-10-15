"""The receipt readers embedded in scripts/install.sh and scripts/install.ps1.

Both scripts carry the same Python code, run with the old tool environment's
interpreter on a re-run. These tests run each copy on the same receipts and
check the lines it prints, so the two copies cannot drift apart unnoticed.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
# Our own delimiters around the reader in each script.
READERS = {
    "install.sh": r"^RECEIPT_READER='\n(.*?)\n'$",
    "install.ps1": r"^ *\$ReceiptReader = @'\n(.*?)\n'@$",
}
THIS_PYTHON = f"{sys.version_info.major}.{sys.version_info.minor}"


def _reader(script: str) -> str:
    text = (SCRIPTS / script).read_text(encoding="utf-8")
    match = re.search(READERS[script], text, re.DOTALL | re.MULTILINE)
    assert match, f"no receipt reader found in scripts/{script}"
    return match.group(1)


def _read(script: str, receipt: Path) -> list[str]:
    result = subprocess.run(
        [sys.executable, "-I", "-c", _reader(script), str(receipt)],
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    return result.stdout.splitlines()


def _cases(local: Path) -> dict[str, tuple[str, list[str]]]:
    """name -> (receipt text, expected lines)."""
    local_toml = str(local).replace("\\", "\\\\")
    return {
        "own extras and python": (
            '[tool]\npython = "3.13"\nrequirements = [\n'
            '  { name = "evoscientist", extras = ["telegram", "oauth"], specifier = "==0.3.6" },\n'
            "]\n",
            ["python\t3.13", "extras\ttelegram,oauth"],
        ),
        "no python recorded": (
            '[tool]\nrequirements = [{ name = "EvoScientist" }]\n',
            [f"python\t{THIS_PYTHON}"],
        ),
        "specifier, extras and marker": (
            "[tool]\nrequirements = [\n"
            '  { name = "evoscientist" },\n'
            '  { name = "six", specifier = ">=1.10,<2", marker = "python_full_version >= \'3.11\'" },\n'
            '  { name = "idna", extras = ["all"], specifier = ">=3" },\n'
            "]\n",
            [
                f"python\t{THIS_PYTHON}",
                "with\tsix>=1.10,<2 ; python_full_version >= '3.11'",
                "with\tidna[all]>=3",
            ],
        ),
        "git and url sources": (
            "[tool]\nrequirements = [\n"
            '  { name = "pyjokes", git = "https://github.com/pyjokes/pyjokes?tag=v0.6.0" },\n'
            '  { name = "sub", git = "https://github.com/o/r?rev=abc123&subdirectory=pkg" },\n'
            '  { name = "arch", url = "https://example.com/arch.tar.gz", subdirectory = "src" },\n'
            "]\n",
            [
                f"python\t{THIS_PYTHON}",
                "with\tpyjokes @ git+https://github.com/pyjokes/pyjokes@v0.6.0",
                "with\tsub @ git+https://github.com/o/r@abc123#subdirectory=pkg",
                "with\tarch @ https://example.com/arch.tar.gz#subdirectory=src",
            ],
        ),
        "local sources": (
            "[tool]\nrequirements = [\n"
            f'  {{ name = "byfile", path = "{local_toml}" }},\n'
            f'  {{ name = "bydir", directory = "{local_toml}" }},\n'
            f'  {{ name = "byedit", editable = "{local_toml}" }},\n'
            "]\n",
            [
                f"python\t{THIS_PYTHON}",
                f"with\tbyfile @ {local.as_uri()}",
                f"with\tbydir @ {local.as_uri()}",
                f"with-editable\t{local}",
            ],
        ),
        "entries that cannot be kept": (
            "[tool]\nrequirements = [\n"
            '  { name = "gitpath", git = "https://github.com/o/r?path=pkg" },\n'
            '  { name = "virt", virtual = "/somewhere" },\n'
            '  { name = "six" },\n'
            "]\n",
            [
                f"python\t{THIS_PYTHON}",
                "warn\tcannot keep 'gitpath' (git path source)",
                "warn\tcannot keep 'virt' (virtual source)",
                "with\tsix",
            ],
        ),
    }


# Any absolute path: only the names are needed here.
CASE_NAMES = list(_cases(Path(__file__).resolve().parent / "local pkg"))


@pytest.mark.parametrize("script", list(READERS))
@pytest.mark.parametrize("case", CASE_NAMES)
def test_reader_keeps_what_the_receipt_records(script, case, tmp_path):
    receipt_text, expected = _cases(tmp_path / "local pkg")[case]
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text(receipt_text, encoding="utf-8")
    assert _read(script, receipt) == expected


@pytest.mark.parametrize("script", list(READERS))
def test_reader_warns_on_an_unreadable_receipt(script, tmp_path):
    receipt = tmp_path / "uv-receipt.toml"
    receipt.write_text("[tool\n", encoding="utf-8")
    (line,) = _read(script, receipt)
    assert line.startswith(f"warn\tcannot read {receipt}: ")
