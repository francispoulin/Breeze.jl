"""Check the standalone timeline in Chromium, including local-file loading.

Requires Playwright and its Chromium browser:
    python3 -m pip install playwright
    python3 -m playwright install chromium
    python3 -B benchmarking/distributed/test_nvtx_timeline_browser.py
"""

import argparse
import html
import json
from pathlib import Path
import re

from playwright.sync_api import sync_playwright


def check_page(path, screenshot=None):
    content = path.read_text()
    payload = re.search(r'<script id="trace-data" type="application/json">(.*?)</script>',
                        content, re.S).group(1)
    data = json.loads(payload)
    template = Path(__file__).with_name("nvtx_timeline.html").read_text()
    expected = template.replace("__TITLE__", html.escape(data["title"])).replace("__DATA__", payload)
    assert content == expected, "Generated HTML differs from the renderer template"
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        context = browser.new_context(viewport={"width": 1440, "height": 1000},
                                      device_scale_factor=2, locale="en-US", offline=True)
        page = context.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.goto(path.resolve().as_uri())
        assert not errors, errors
        assert f"{data['steps']} captured steps" in page.locator("#summary").inner_text()
        assert f"{data['kernels']:,} CUDA kernels" in page.locator("#summary").inner_text()
        assert page.locator("#legend .swatch").count() == len(data["categories"])

        def bounds():
            return [float(v.removesuffix(" ms"))
                    for v in page.locator("#window").inner_text().split("–")]

        def span():
            start, end = bounds()
            return end - start

        def canvas_box():
            return page.locator("#timeline").bounding_box()

        # Inspect canvas pixels, rather than treating a nonempty DOM as a plot.
        colors = page.locator("#timeline").evaluate("""canvas => {
            const pixels = canvas.getContext('2d').getImageData(0,0,canvas.width,canvas.height).data;
            const colors = new Set();
            for (let i=0; i<pixels.length; i+=256) colors.add(`${pixels[i]},${pixels[i+1]},${pixels[i+2]}`);
            return colors.size;
        }""")
        assert colors > 10, f"Expected colored bars, found only {colors} canvas colors"
        assert 100 < canvas_box()["height"] < 1000
        initial = span()
        page.locator("#in").click()
        assert abs(span() - initial / 2) < 0.01
        page.locator("#out").click()
        assert abs(span() - initial) < 0.01
        page.locator("#full").click()
        assert abs(span() - (data["extent"][1] - data["extent"][0])) < 0.01
        page.locator("#step").click()
        assert abs(span() - initial) < 0.01
        box = canvas_box()
        page.mouse.move(box["x"] + box["width"] / 2, box["y"] + 80)
        page.mouse.wheel(0, -120)
        page.wait_for_function("""initial => {
            const values=document.getElementById('window').textContent.split('–').map(parseFloat);
            return values[1]-values[0]<initial;
        }""", arg=initial)
        before = bounds()[0]
        page.mouse.down()
        page.mouse.move(box["x"] + box["width"] / 2 + 40, box["y"] + 80, steps=4)
        page.mouse.up()
        assert bounds()[0] < before, "Dragging did not pan the timeline"
        page.locator("#step").click()

        lane = data["lanes"].index("GPU projection · level 0")
        event = next(e for e in data["events"] if e[2] == lane)
        start, end = bounds()
        fraction = ((event[0] + event[1]) / 2 - start) / (end - start)
        box = canvas_box()
        page.mouse.move(box["x"] + 1 + 220 + fraction * (box["width"] - 2 - 240),
                        box["y"] + 1 + 48 + lane * 34 + 17)
        assert page.locator("#tooltip").is_visible()
        assert "time_step! #1" in page.locator("#tooltip").inner_text()
        assert "Duration:" in page.locator("#tooltip").inner_text()
        page.set_viewport_size({"width": 1024, "height": 900})
        assert canvas_box()["width"] < box["width"], "Canvas did not resize"
        assert not errors, errors
        if screenshot:
            screenshot.parent.mkdir(parents=True, exist_ok=True)
            page.mouse.move(0, 0)
            page.screenshot(path=str(screenshot), full_page=True)
        print(f"PASS Chromium {browser.version}: offline local file, plotted bars, buttons, "
              "wheel zoom, drag pan, hover and resize; no JavaScript errors")
        browser.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("page", type=Path, nargs="?", default=Path(__file__).resolve().parents[1] /
                        "results/rondeau/nvtx_demo_512_cubed_2026-11-10/timeline.html")
    parser.add_argument("--screenshot", type=Path)
    args = parser.parse_args()
    check_page(args.page, args.screenshot)


if __name__ == "__main__":
    main()
