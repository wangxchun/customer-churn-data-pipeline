"""W5 (plan 6.5): a new release directory is 0700 and its files 0600.

Synthetic run, split, model and key only, all under pytest's tmp_path; no real release,
production pointer, audit log or holdout guard is touched. The umask is process-wide,
so every test that changes it restores it in ``finally``."""
from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from churn_pipeline import release as rel
from churn_pipeline import tracking
from churn_pipeline.features import build_features
from test_inference import fit_model, synthetic_master
from test_release import KEY, RUN_ID, make_env

pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX mode bits only")


def mode(p: Path) -> int:
    return stat.S_IMODE(p.lstat().st_mode)


@pytest.fixture(scope="module")
def master():
    return synthetic_master()


@pytest.fixture(scope="module")
def model(master):
    return fit_model(master)


@pytest.fixture(scope="module")
def assignments(master):
    return build_features(master).assignments


@pytest.fixture
def env(tmp_path, model, assignments):
    paths = make_env(tmp_path, model, assignments)
    rel._releases_dir(paths).mkdir(parents=True)     # parents exist before any umask change
    return paths


def release_dir(paths, rid):
    return rel._releases_dir(paths) / rid


def leftovers(paths):
    return [p.name for p in rel._releases_dir(paths).iterdir() if p.name.startswith(".tmp-")]


def events(paths):
    return [e["event"] for e in rel.read_audit_log(paths)]


class Umask:
    def __init__(self, value):
        self.value = value

    def __enter__(self):
        self.old = os.umask(self.value)

    def __exit__(self, *exc):
        os.umask(self.old)


# --- new releases ------------------------------------------------------------------------
@pytest.mark.parametrize("umask", [0o000, 0o022, 0o027, 0o077, 0o277])
def test_new_release_is_0700_with_0600_files_under_any_umask(env, umask):
    with Umask(umask):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-perm")
    d = release_dir(env, "rel-perm")
    assert mode(d) == 0o700
    files = sorted(p.name for p in d.iterdir())
    assert files == sorted([rel.GUARD_FILE, rel.RELEASE_FILE])
    assert all(mode(d / f) == 0o600 for f in files)
    assert leftovers(env) == []


def test_release_still_loads_and_passes_check_release(env):
    m = rel.create_release(env, RUN_ID, KEY, release_id="rel-ok")
    saved = json.loads((release_dir(env, "rel-ok") / rel.RELEASE_FILE).read_text())
    assert saved == json.loads(json.dumps(tracking._jsonable(m)))     # same content as before
    assert rel.load_release(env, "rel-ok", KEY).release_id == "rel-ok"
    assert set(rel.check_release(env, "rel-ok", KEY)) == set(rel.CHECK_GATES)


def test_release_json_bytes_match_the_unrestricted_writer(tmp_path):
    obj = {"a": 1, "b": [1.5, None], "c": "é"}
    tracking.write_json(tmp_path / "plain.json", obj)
    tracking.write_json(tmp_path / "private.json", obj, mode=0o600)
    assert (tmp_path / "plain.json").read_bytes() == (tmp_path / "private.json").read_bytes()


# --- existing directories and files --------------------------------------------------------
def test_existing_release_directory_is_refused_and_left_unchanged(env):
    d = release_dir(env, "rel-exists")
    d.mkdir(mode=0o755)
    os.chmod(d, 0o755)
    (d / "theirs.txt").write_text("keep")
    os.chmod(d / "theirs.txt", 0o644)
    with pytest.raises(FileExistsError):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-exists")
    assert mode(d) == 0o755 and mode(d / "theirs.txt") == 0o644
    assert sorted(p.name for p in d.iterdir()) == ["theirs.txt"]
    assert (d / "theirs.txt").read_text() == "keep" and leftovers(env) == []


def test_private_write_never_overwrites_or_rechmods_an_existing_file(tmp_path):
    p = tmp_path / "x.json"
    p.write_text("original")
    os.chmod(p, 0o644)
    with pytest.raises(FileExistsError):
        tracking.write_json(p, {"new": True}, mode=0o600)
    assert p.read_text() == "original" and mode(p) == 0o644


def test_existing_releases_are_not_rechmodded_by_a_new_release(env):
    rel.create_release(env, RUN_ID, KEY, release_id="rel-old")
    old = release_dir(env, "rel-old")
    os.chmod(old, 0o755)                     # as releases made before W5 are
    os.chmod(old / rel.RELEASE_FILE, 0o644)
    rel.create_release(env, RUN_ID, KEY, release_id="rel-new")
    assert mode(old) == 0o755 and mode(old / rel.RELEASE_FILE) == 0o644
    assert mode(release_dir(env, "rel-new")) == 0o700


# --- failures leave nothing that looks like a release -----------------------------------
def test_failed_file_chmod_leaves_no_release_and_no_temp_dir(env, monkeypatch):
    def fail(*a, **k):
        raise PermissionError("simulated fchmod failure")
    monkeypatch.setattr(os, "fchmod", fail)
    with pytest.raises(PermissionError):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-fchmod")
    assert not release_dir(env, "rel-fchmod").exists() and leftovers(env) == []
    assert "release_created" not in events(env)


def test_release_is_private_without_os_fchmod(env, monkeypatch):
    # Simulates the fallback branch only (e.g. Windows before Python 3.13); it is not a
    # real cross-platform run.
    monkeypatch.delattr(os, "fchmod")
    rel.create_release(env, RUN_ID, KEY, release_id="rel-nofchmod")
    d = release_dir(env, "rel-nofchmod")
    assert mode(d) == 0o700
    assert mode(d / rel.RELEASE_FILE) == 0o600 and mode(d / rel.GUARD_FILE) == 0o600
    assert leftovers(env) == [] and "release_created" in events(env)


def test_failed_directory_chmod_leaves_no_release_and_no_temp_dir(env, monkeypatch):
    def fail(*a, **k):
        raise PermissionError("simulated chmod failure")
    monkeypatch.setattr(os, "chmod", fail)
    with pytest.raises(PermissionError):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-chmod")
    assert not release_dir(env, "rel-chmod").exists() and leftovers(env) == []
    assert "release_created" not in events(env)


def test_a_file_that_ends_up_looser_blocks_the_release(env, monkeypatch):
    # E.g. a filesystem that ignores chmod: the check before the rename refuses it.
    real = rel.write_json
    monkeypatch.setattr(rel, "write_json",
                        lambda path, obj, mode=None: (real(path, obj), os.chmod(path, 0o644)))
    with pytest.raises(rel.ReleaseError, match="owner-only.*release.json"):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-loose")
    assert not release_dir(env, "rel-loose").exists() and leftovers(env) == []
    assert "release_created" not in events(env)


def test_an_unexpected_entry_blocks_the_release(env, monkeypatch):
    real = rel.write_json

    def plus_subdir(path, obj, mode=None):
        real(path, obj, mode=mode)
        (path.parent / "sub").mkdir(mode=0o700)
        os.chmod(path.parent / "sub", 0o600)
    monkeypatch.setattr(rel, "write_json", plus_subdir)
    with pytest.raises(rel.ReleaseError, match="sub"):
        rel.create_release(env, RUN_ID, KEY, release_id="rel-sub")
    assert not release_dir(env, "rel-sub").exists() and leftovers(env) == []


# --- atomic_directory: other callers unchanged ----------------------------------------------
def test_atomic_directory_without_mode_keeps_the_umask_behaviour(tmp_path):
    with Umask(0o022):
        with tracking.atomic_directory(tmp_path / "out") as tmp:
            (tmp / "f").write_text("x")
    assert mode(tmp_path / "out") == 0o755 and mode(tmp_path / "out" / "f") == 0o644


def test_atomic_directory_with_mode_still_cleans_up_on_failure(tmp_path):
    with pytest.raises(RuntimeError):
        with tracking.atomic_directory(tmp_path / "out", mode=0o700):
            raise RuntimeError("boom")
    assert list(tmp_path.iterdir()) == []
