"""Path helpers for persistent deployments."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path

import click

from headroom import paths as _paths

_PROFILE_RE = re.compile(r"^[A-Za-z0-9._-]+$")

# A deployment profile directory holds secrets. `headroom install --env
# KEY=VALUE` is the supported way to give a supervised proxy a provider API key
# — supervisors (launchd, systemd, Task Scheduler, cron) start from a bare
# environment, so nothing else reaches the process — and every such value is
# persisted verbatim, both into `manifest.json` and into the generated runner
# scripts that `export` it before the exec. Those files are therefore written
# owner-only rather than at the process umask (which leaves them world-readable
# at the common 022). The supervisor always runs them as the installing user
# (user scope) or as root (system scope), and neither needs the group/other
# bits, so this is not a functional restriction.
#
# SECURITY.md documents these modes; `tests/test_packaging_extras_and_security_docs.py`
# reads the octal values back out of that document and compares them with what
# an install actually writes, so the policy and the code cannot drift apart.
SECRET_FILE_MODE = 0o600
SECRET_SCRIPT_MODE = 0o700
SECRET_DIR_MODE = 0o700


def chmod_owner_only(path: Path, mode: int) -> None:
    """Best-effort ``chmod`` to an owner-only ``mode``.

    POSIX permission bits are advisory on Windows (access is governed by ACLs)
    and some filesystems reject ``chmod`` outright; a deployment must not fail
    to install over it, so a failure is logged by the caller's context rather
    than raised.
    """
    try:
        path.chmod(mode)
    except OSError:  # pragma: no cover - platform/filesystem dependent
        pass


def validate_profile_name(profile: str) -> str:
    """Validate and normalize a deployment profile name."""

    if profile in {".", ".."} or not _PROFILE_RE.fullmatch(profile):
        raise click.ClickException(f"Invalid profile name '{profile}'")
    return profile


def deploy_root() -> Path:
    """Return the root directory for deployment state."""

    return _paths.deploy_root()


def profile_root(profile: str) -> Path:
    """Return the directory for a named deployment profile."""

    return deploy_root() / validate_profile_name(profile)


def manifest_path(profile: str) -> Path:
    """Return the manifest path for a named profile."""

    return profile_root(profile) / "manifest.json"


def log_path(profile: str) -> Path:
    """Return the log path used by persistent runner scripts."""

    return profile_root(profile) / "runner.log"


def pid_path(profile: str) -> Path:
    """Return the pid file for the raw runtime process."""

    return profile_root(profile) / "runner.pid"


def unix_run_script_path(profile: str) -> Path:
    """Return the foreground runner shell script path."""

    return profile_root(profile) / "run-headroom.sh"


def unix_ensure_script_path(profile: str) -> Path:
    """Return the watchdog shell script path."""

    return profile_root(profile) / "ensure-headroom.sh"


def windows_run_script_path(profile: str) -> Path:
    """Return the foreground runner PowerShell script path."""

    return profile_root(profile) / "run-headroom.ps1"


def windows_run_cmd_path(profile: str) -> Path:
    """Return the foreground runner CMD shim path."""

    return profile_root(profile) / "run-headroom.cmd"


def windows_ensure_script_path(profile: str) -> Path:
    """Return the watchdog PowerShell script path."""

    return profile_root(profile) / "ensure-headroom.ps1"


def windows_ensure_cmd_path(profile: str) -> Path:
    """Return the watchdog CMD shim path."""

    return profile_root(profile) / "ensure-headroom.cmd"


def unix_user_env_targets() -> list[Path]:
    """Return user shell files that can carry the persistent env block."""

    home = Path.home()
    return [home / ".bashrc", home / ".zshrc", home / ".profile"]


def unix_system_env_targets() -> list[Path]:
    """Return system shell files that can carry the persistent env block."""

    if sys.platform == "darwin":
        return [Path("/etc/profile"), Path("/etc/zprofile"), Path("/etc/bashrc")]
    return [Path("/etc/profile.d/headroom.sh")]


def claude_settings_path() -> Path:
    """Return the Claude user settings path."""

    return Path.home() / ".claude" / "settings.json"


def codex_config_path() -> Path:
    """Return the Codex config path."""

    return Path.home() / ".codex" / "config.toml"


def openclaw_config_path() -> Path:
    """Return the OpenClaw config path."""

    return Path.home() / ".openclaw" / "openclaw.json"


def opencode_config_path() -> Path:
    """Return the OpenCode config path.

    Resolves ``~/.config/opencode/opencode.json`` when ``OPENCODE_CONFIG``
    is unset; otherwise the value of that environment variable. Checks for
    ``opencode.jsonc`` as well.
    """

    env_path = os.environ.get("OPENCODE_CONFIG", "").strip()
    if env_path:
        return Path(env_path).expanduser()
    base_dir = Path.home() / ".config" / "opencode"
    jsonc_path = base_dir / "opencode.jsonc"

    if jsonc_path.exists():
        return jsonc_path

    return base_dir / "opencode.json"


def zcode_config_dir() -> Path:
    """Return the ZCode user configuration directory."""

    return Path.home() / ".zcode"
