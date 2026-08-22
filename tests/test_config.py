"""The configuration contract.

Every setting is read from the environment with no fallback in code, which
makes .env.example the only complete statement of what this application needs
to start. That is a contract worth testing: a setting added to Settings without
a matching line in the template would otherwise stay invisible here and surface
as a startup failure on somebody else's machine. It happened once already.
"""

from __future__ import annotations

import pytest
from app.config import PROJECT_ROOT, ConfigurationError, Settings, get_settings
from pydantic import ValidationError

REQUIRED = sorted(
    name.upper() for name, field in Settings.model_fields.items() if field.is_required()
)


def test_template_supplies_every_required_setting(env_template):
    missing = [name for name in REQUIRED if name not in env_template]
    assert not missing, f".env.example is missing: {', '.join(missing)}"


def test_template_carries_nothing_the_application_ignores(env_template):
    """The reverse direction: a leftover line for a setting that was removed
    reads like live configuration and quietly does nothing."""
    known = {name.upper() for name in Settings.model_fields}
    stale = sorted(set(env_template) - known)
    assert not stale, f".env.example describes settings that no longer exist: {stale}"


def test_the_folder_paths_are_derived_not_configured(env_template):
    """Deliberate exception. Requiring these would put an absolute path from one
    developer's machine into .env, which then breaks inside a container."""
    assert not Settings.model_fields["contracts_dir"].is_required()
    assert not Settings.model_fields["data_dir"].is_required()
    assert "CONTRACTS_DIR" not in env_template
    assert "DATA_DIR" not in env_template


def test_the_template_is_committed_and_the_real_file_is_not():
    """.env holds the gateway key. The template beside it must never."""
    assert (PROJECT_ROOT / ".env.example").is_file()
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
    assert ".env.example" in message
