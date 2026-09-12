from pathlib import Path

import pytest
from airfare import config


def test_project_root_prefers_source_checkout() -> None:
    assert (config.PROJECT_ROOT / "pyproject.toml").exists()


def test_project_root_falls_back_to_cwd(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        config, "__file__", str(tmp_path / "site-packages" / "airfare" / "config.py")
    )
    monkeypatch.chdir(tmp_path)
    assert config._project_root() == tmp_path
