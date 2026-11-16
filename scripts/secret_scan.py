#!/usr/bin/env python3
"""W5: a small, stdlib-only secret scan of the files git would commit.

Scope: the files ``git ls-files --cached --others --exclude-standard`` lists, i.e. tracked
files plus untracked files that are not ignored. Ignored paths (``data/``, ``docs/``,
``.venv/`` ...) are never opened. Symlinks, files that look binary (a NUL byte in the
first 8 KiB) and files over ``MAX_BYTES`` are skipped and counted.

Output: one line per finding, ``path:line:col: rule [redacted, N chars]``. The matched
value and the line's text are NEVER printed, so a real secret is not copied into a
terminal, a CI log or an error message.

Exit codes: 0 no findings, 1 findings, 2 the scan could not run (e.g. not a git repo).

Limits: this is a regex scan for a fixed list of well-known credential formats. It
cannot find secrets in other formats, encoded or split secrets, high-entropy strings
without a known prefix, secrets in git history, or anything in ignored files. A clean
result does not mean the repository contains no secrets.

Test fixtures must build fake credentials at runtime (e.g. by concatenation), so that no
complete credential-shaped string appears in any tracked file.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

MAX_BYTES = 2 * 1024 * 1024
BINARY_SNIFF = 8192

# (rule id, pattern). Prefix-anchored formats only, to keep false positives low.
RULES: list[tuple[str, re.Pattern]] = [
    ("aws-access-key-id", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b")),
    ("aws-secret-access-key",
     re.compile(r"(?i)aws_secret_access_key\s*[:=]\s*[\"']?[A-Za-z0-9/+=]{40}(?![A-Za-z0-9/+=])")),
    ("private-key-block",
     re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP |ENCRYPTED )?PRIVATE KEY(?: BLOCK)?-----")),
    ("github-token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("github-fine-grained-pat", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{22,}\b")),
    ("slack-token", re.compile(r"\bxox[abposr]-[A-Za-z0-9-]{10,}")),
    ("google-api-key", re.compile(r"\bAIza[0-9A-Za-z_-]{35}(?![0-9A-Za-z_-])")),
    ("stripe-live-key", re.compile(r"\b[sr]k_live_[0-9A-Za-z]{24,}\b")),
    ("anthropic-api-key", re.compile(r"\bsk-ant-[A-Za-z0-9_-]{20,}")),
    ("openai-api-key", re.compile(r"\bsk-(?:proj-)?[A-Za-z0-9_-]{32,}")),
    ("credential-assignment",
     re.compile(r"(?i)\b(?:password|passwd|secret_key|client_secret|api_key|apikey|"
                r"access_token|auth_token)\s*[:=]\s*[\"'][^\"'\s]{12,}[\"']")),
]


@dataclass(frozen=True)
class Finding:
    path: str
    line: int
    col: int
    rule: str
    length: int

    def render(self) -> str:
        return f"{self.path}:{self.line}:{self.col}: {self.rule} [redacted, {self.length} chars]"


@dataclass
class ScanResult:
    findings: list[Finding]
    scanned: int = 0
    skipped: int = 0


class ScanError(RuntimeError):
    pass


def candidate_files(root: Path) -> list[str]:
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z", "--cached", "--others",
             "--exclude-standard"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError) as e:
        raise ScanError(f"cannot list files with git in {root}") from e
    return sorted({p for p in out.stdout.decode("utf-8", "surrogateescape").split("\0") if p})


def scan_text(path: str, text: str) -> list[Finding]:
    found = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        for rule, pattern in RULES:
            for m in pattern.finditer(line):
                found.append(Finding(path, lineno, m.start() + 1, rule, len(m.group(0))))
    return found


def scan(root: Path) -> ScanResult:
    root = Path(root)
    result = ScanResult(findings=[])
    for rel in candidate_files(root):
        p = root / rel
        if p.is_symlink() or not p.is_file():       # a deleted tracked file, or a link
            result.skipped += 1
            continue
        if p.stat().st_size > MAX_BYTES:
            result.skipped += 1
            continue
        data = p.read_bytes()
        if b"\0" in data[:BINARY_SNIFF]:
            result.skipped += 1
            continue
        result.scanned += 1
        result.findings.extend(scan_text(rel, data.decode("utf-8", "replace")))
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", type=Path, default=Path.cwd(), help="git work tree to scan")
    args = ap.parse_args(argv)
    try:
        r = scan(args.root)
    except ScanError as e:
        print(f"secret scan: error: {e}", file=sys.stderr)
        return 2
    for f in r.findings:
        print(f.render())
    print(f"secret scan: {len(r.findings)} finding(s); {r.scanned} file(s) scanned, "
          f"{r.skipped} skipped (binary, symlink, missing or > {MAX_BYTES} bytes). "
          "Regex scan of known formats only; a clean result is not proof of no secrets.",
          file=sys.stderr)
    return 1 if r.findings else 0


if __name__ == "__main__":
    sys.exit(main())
