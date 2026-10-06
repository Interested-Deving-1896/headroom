"""The Headroom for Teams offer shows on unlicensed installs only."""

from __future__ import annotations

import pytest

from headroom.dashboard import get_dashboard_html


def test_offer_shown_without_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("HEADROOM_LICENSE", raising=False)
    html = get_dashboard_html()
    assert 'data-testid="teams-perk"' in html
    assert "headroom-perks.vercel.app" in html


def test_offer_removed_with_license(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("HEADROOM_LICENSE", "hlk_test")
    html = get_dashboard_html()
    assert 'data-testid="teams-perk"' not in html
    assert "headroom-perks.vercel.app" not in html
    assert "teams-perk:start" not in html
    # Only the offer block goes; the dashboard around it is intact.
    assert 'data-testid="session-view"' in html
    assert html.count("<main") == 1
