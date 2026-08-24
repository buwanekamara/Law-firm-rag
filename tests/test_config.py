"""The configuration contract.

Every setting is read from the environment with no fallback in code, so .env
is the only complete statement of what the application needs to start. A
setting added to Settings without a matching line there would otherwise
surface as a startup failure on someone else's machine.

These tests skip when there is no .env, so a checkout without one still runs
green - the suite supplies its own environment in conftest.
"""

from __future__ import annotations

import pytest
from app.config import PROJECT_ROOT, ConfigurationError, Settings, get_settings
from pydantic import ValidationError

REQUIRED = sorted(
    name.upper() for name, field in Settings.model_fields.items() if field.is_required()
)


def test_env_supplies_every_required_setting(env_file):
    missing = [name for name in REQUIRED if name not in env_file]
    assert not missing, f".env is missing: {', '.join(missing)}"


def test_env_carries_nothing_the_application_ignores(env_file):
    """The reverse direction: a leftover line for a setting that was removed
    reads like live configuration and quietly does nothing."""
    known = {name.upper() for name in Settings.model_fields}
    stale = sorted(set(env_file) - known)
    assert not stale, f".env names settings that no longer exist: {stale}"


def test_the_folder_paths_are_derived_not_configured(env_file):
    """Deliberate exception. Requiring these would put an absolute path from one
    developer's machine into .env, which then breaks inside a container."""
    assert not Settings.model_fields["contracts_dir"].is_required()
    assert not Settings.model_fields["data_dir"].is_required()
    assert "CONTRACTS_DIR" not in env_file
    assert "DATA_DIR" not in env_file


def test_the_key_file_is_never_committed():
    """.env holds the gateway key, so git must not be able to see it."""
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in gitignore]


def test_an_incomplete_environment_names_what_is_absent(monkeypatch):
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(ValidationError) as caught:
        Settings(_env_file=None)

    reported = {str(item["loc"][0]).upper() for item in caught.value.errors()}
    assert "AI_GATEWAY_API_KEY" in reported
    assert "PROMPT_VERSION" in reported


def test_the_startup_error_reads_like_an_instruction(monkeypatch):
    """A raw validation dump tells a deployer what a computer noticed. This
    tells them what to do about it."""
    for name in REQUIRED:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(Settings, "model_config", {**Settings.model_config, "env_file": None})

    with pytest.raises(ConfigurationError) as caught:
        get_settings.__wrapped__()

    message = str(caught.value)
    assert "AI_GATEWAY_API_KEY" in message
    assert ".env" in message
