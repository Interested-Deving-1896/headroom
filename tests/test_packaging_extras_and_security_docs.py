"""A-10: the `sandbox` extra promised isolation it never provided, and
SECURITY.md drifted four minor releases behind the code.

Both are documentation defects with a security consequence: a reader who
believes either one makes a deployment decision on a false premise. These tests
pin the rename's compatibility alias and make the stale version table — the part
that silently rots — a build failure instead of a reviewer's job.
"""

from __future__ import annotations

import re
from pathlib import Path

import tomllib

_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    with (_ROOT / "pyproject.toml").open("rb") as fh:
        return tomllib.load(fh)


def test_sandbox_alias_still_resolves_to_the_lean_profile() -> None:
    """The old name must keep working for one release.

    `sandbox` was renamed to `lean`; anything already installing the old extra
    (our own python-314-wheels CI job did) must not break on upgrade.
    """
    extras = _pyproject()["project"]["optional-dependencies"]

    assert "lean" in extras, "the renamed extra is missing"
    assert "sandbox" in extras, "the deprecated alias was removed too early"
    assert extras["sandbox"] == ["headroom-ai[lean]"], (
        "the alias must forward to `lean` rather than duplicate its dependency "
        "list, or the two will drift"
    )


def test_lean_profile_stays_torch_free() -> None:
    """The profile's whole purpose is avoiding torch; guard it by name.

    A torch-pulling extra added here would quietly defeat the reason anyone
    selects this profile.
    """
    extras = _pyproject()["project"]["optional-dependencies"]
    (spec,) = extras["lean"]
    selected = set(re.search(r"\[(.*)\]", spec).group(1).split(","))

    torch_pulling = {"ml", "memory", "evals", "image", "voice", "pytorch-mps"}
    assert not (selected & torch_pulling), (
        f"`lean` must stay torch-free; it now selects {sorted(selected & torch_pulling)}"
    )


def test_security_policy_supported_versions_match_the_shipped_version() -> None:
    """SECURITY.md's table sat at 0.27.x while the code shipped 0.38.0.

    A supported-versions table that lags is worse than none: it tells a reporter
    their release is unsupported when it is the current one.
    """
    version = _pyproject()["project"]["version"]
    series = ".".join(version.split(".")[:2]) + ".x"

    text = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert f"| {series} (latest) |" in text, (
        f"SECURITY.md does not list {series} as the supported series "
        f"(pyproject version is {version})"
    )
    assert f"| < {series} |" in text, f"SECURITY.md does not mark < {series} unsupported"


def test_security_policy_does_not_claim_blanket_no_credential_storage() -> None:
    """`headroom copilot login` writes an OAuth refresh token to disk.

    The policy used to say "We never store or log API keys", which is broader
    than the code supports. Keep the honest wording.
    """
    text = (_ROOT / "SECURITY.md").read_text(encoding="utf-8")

    assert "never store or log API keys" not in text, (
        "the blanket no-credential-storage claim is contradicted by "
        "headroom/copilot_auth.py, which persists the Copilot OAuth refresh token"
    )
    assert "copilot_auth.json" in text, "the one stored credential must stay documented"
