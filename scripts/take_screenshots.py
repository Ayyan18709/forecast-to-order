"""Capture dashboard screenshots for the README.

Starts the Streamlit app, opens it in a locally installed Edge or Chrome via
Playwright (no browser download needed) and saves PNGs to docs/images/.

Usage:
    python scripts/take_screenshots.py
"""

from __future__ import annotations

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "images"
PORT = 8599
URL = f"http://localhost:{PORT}"


def wait_for_server(timeout: float = 60) -> None:
    """Block until the Streamlit health endpoint answers."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f"{URL}/_stcore/health", timeout=2) as r:
                if r.status == 200:
                    return
        except OSError:
            time.sleep(1)
    raise TimeoutError("Streamlit did not start")


def settle(page: Page, min_charts: int) -> None:
    """Wait until Streamlit has finished running and the charts are drawn."""
    page.wait_for_function(f"document.querySelectorAll('.js-plotly-plot .main-svg').length >= {min_charts}", timeout=90_000)
    page.wait_for_selector("[data-testid='stStatusWidget']", state="detached", timeout=90_000)
    page.wait_for_timeout(2500)


def open_tab(page: Page, name: str) -> None:
    page.get_by_role("tab", name=name).click()
    page.wait_for_timeout(3000)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen([sys.executable, "-m", "streamlit", "run", "app.py", "--server.port", str(PORT),
                             "--server.headless", "true"], cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        wait_for_server()
        with sync_playwright() as pw:
            browser = None
            for channel in ("msedge", "chrome"):
                try:
                    browser = pw.chromium.launch(channel=channel)
                    break
                except Exception:  # noqa: BLE001 - try the next installed browser
                    continue
            if browser is None:
                browser = pw.chromium.launch()  # requires `playwright install chromium`
            page = browser.new_page(viewport={"width": 1440, "height": 1900}, device_scale_factor=1.5)
            page.goto(URL)
            settle(page, 2)
            page.screenshot(path=OUT / "dashboard_overview.png")
            page.screenshot(path=OUT / "dashboard_kpis.png", clip={"x": 0, "y": 0, "width": 1440, "height": 720})

            open_tab(page, "Cash vs. service trade-off")
            settle(page, 3)
            el = page.get_by_role("tabpanel").filter(has_text="Every ordering rule")
            el.screenshot(path=OUT / "dashboard_tradeoff.png")

            open_tab(page, "Model accuracy")
            settle(page, 4)
            el = page.get_by_role("tabpanel").filter(has_text="Rolling-origin backtest")
            el.screenshot(path=OUT / "dashboard_accuracy.png")
            browser.close()
    finally:
        proc.terminate()
        proc.wait(timeout=20)
    for f in sorted(OUT.glob("*.png")):
        print(f.relative_to(ROOT), f"{f.stat().st_size / 1e3:.0f} kB")


if __name__ == "__main__":
    main()
