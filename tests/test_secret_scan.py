"""W5: scripts/secret_scan.py on throw-away git repositories under tmp_path.

Every fake credential is assembled at runtime from fragments, so no complete
credential-shaped string appears in this (tracked) file; test_repo_hygiene scans the
real repository and would fail otherwise. None of these values is a real credential."""
from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "secret_scan.py"
_spec = importlib.util.spec_from_file_location("secret_scan", SCRIPT)
ss = importlib.util.module_from_spec(_spec)
sys.modules["secret_scan"] = ss          # dataclasses look the module up
_spec.loader.exec_module(ss)


def j(*parts: str) -> str:
    return "".join(parts)


FAKES = {
    "aws-access-key-id": j("AK", "IA", "FAKEFAKEFAKE0000"),
    "aws-secret-access-key": j("aws_secret_", "access_key = ", "F" * 20, "a/b+" * 5),
    "private-key-block": j("-----BEGIN ", "RSA PRIV", "ATE KEY-----"),
    "github-token": j("gh", "p_", "Fake" * 9),
    "github-fine-grained-pat": j("github", "_pat_", "Fake_" * 6),
    "slack-token": j("xo", "xb-", "0000-fake-fake"),
    "google-api-key": j("AI", "za", "F" * 35),
    "stripe-live-key": j("sk", "_live_", "Fake" * 6),
    "anthropic-api-key": j("sk-", "ant-", "fake" * 6),
    "openai-api-key": j("sk-", "proj-", "fake" * 9),
    "credential-assignment": j("pass", "word = '", "not-a-real-one", "'"),
}


def git_repo(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    return root


def git_add(root: Path, *names: str) -> None:
    subprocess.run(["git", "-C", str(root), "add", "--", *names], check=True)


@pytest.mark.parametrize("rule,value", sorted(FAKES.items()))
def test_each_rule_finds_its_fake_credential(rule, value):
    found = ss.scan_text("x.txt", f"prefix {value}\n")
    assert rule in {f.rule for f in found}


@pytest.mark.parametrize("text", [
    "KEY = bytes(range(32))",
    "aws_secret_access_key = os.environ['X']",      # no literal value
    "password = ''",                                  # empty
    "the AKIA prefix alone, and sk-short",
    "BEGIN PUBLIC KEY",
    "sha256 = '" + "ab" * 32 + "'",                   # a hash, not a credential
])
def test_ordinary_code_is_not_flagged(text):
    assert ss.scan_text("x.py", text) == []


def test_findings_report_location_and_rule_but_never_the_value(tmp_path, capsys):
    repo = git_repo(tmp_path / "r")
    lines = ["clean line"] + [f"x = '{v}'" for v in FAKES.values()]
    (repo / "config.txt").write_text("\n".join(lines) + "\n")
    git_add(repo, "config.txt")
    assert ss.main(["--root", str(repo)]) == 1
    out = capsys.readouterr()
    text = out.out + out.err
    for v in FAKES.values():
        assert v not in text
        assert v[4:16] not in text                     # not even a fragment of it
    assert "config.txt:2:" in out.out and "aws-access-key-id" in out.out
    assert "[redacted," in out.out
    assert "not proof of no secrets" in out.err


def test_scope_is_tracked_plus_untracked_not_ignored(tmp_path):
    repo = git_repo(tmp_path / "r")
    key = FAKES["aws-access-key-id"]
    (repo / ".gitignore").write_text("ignored/\n*.secret\n")
    (repo / "ignored").mkdir()
    (repo / "ignored" / "a.txt").write_text(key)
    (repo / "b.secret").write_text(key)
    (repo / "tracked.txt").write_text(key)
    (repo / "new_file.txt").write_text(key)            # untracked, not ignored
    git_add(repo, ".gitignore", "tracked.txt")
    paths = {f.path for f in ss.scan(repo).findings}
    assert paths == {"tracked.txt", "new_file.txt"}


def test_ignored_files_are_never_opened(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "r")
    (repo / ".gitignore").write_text("data/\n")
    (repo / "data").mkdir()
    (repo / "data" / "x.parquet").write_text("x")
    (repo / "ok.txt").write_text("fine")
    git_add(repo, ".gitignore", "ok.txt")
    opened = []
    real = Path.read_bytes
    monkeypatch.setattr(Path, "read_bytes", lambda self: opened.append(self.name) or real(self))
    ss.scan(repo)
    assert "x.parquet" not in opened and "ok.txt" in opened


def test_binary_symlink_and_large_files_are_skipped(tmp_path, monkeypatch):
    repo = git_repo(tmp_path / "r")
    key = FAKES["aws-access-key-id"]
    (repo / "bin.dat").write_bytes(b"\0\1" + key.encode())
    (repo / "outside.txt").write_text(key)
    (repo / "link.txt").symlink_to(repo / "outside.txt")
    (repo / "big.txt").write_text("y" * 64 + key)
    monkeypatch.setattr(ss, "MAX_BYTES", 32)
    git_add(repo, "bin.dat", "link.txt", "big.txt")
    (repo / "outside.txt").unlink()
    r = ss.scan(repo)
    assert r.findings == [] and r.skipped == 3


def test_clean_repo_exits_0(tmp_path, capsys):
    repo = git_repo(tmp_path / "r")
    (repo / "a.py").write_text("print('hello')\n")
    git_add(repo, "a.py")
    assert ss.main(["--root", str(repo)]) == 0
    assert "0 finding(s); 1 file(s) scanned" in capsys.readouterr().err


def test_not_a_git_repo_exits_2(tmp_path, capsys):
    (tmp_path / "plain").mkdir()
    assert ss.main(["--root", str(tmp_path / "plain")]) == 2
    assert "error" in capsys.readouterr().err
