"""Results page for people (animated photos, chart, agent story) and MCP server for agents.

All pictures are rendered once on the laptop by scripts/render_visuals.py and shipped in
assets/, so this app only shows files and needs no dataset or plotting library.
Agents call the same tool functions at /gradio_api/mcp/.
"""

from __future__ import annotations

import json
from pathlib import Path

import gradio as gr

from triage_agent import tools

ASSETS = Path("assets")
gr.set_static_paths(paths=[ASSETS])
CAPTIONS = json.loads((ASSETS / "captions.json").read_text(encoding="utf-8"))

CSS = """
.stats {display: flex; gap: 16px; flex-wrap: wrap; margin: 8px 0 4px;}
.card {flex: 1 1 180px; border-radius: 14px; padding: 18px; background: #1f2937; color: white;}
.card b {display: block; font-size: 2.1em; line-height: 1.1;}
.card span {opacity: .8;}
.gifs {display: grid; grid-template-columns: repeat(auto-fill, minmax(300px, 1fr)); gap: 14px;}
.gifs figure {margin: 0;}
.gifs img {width: 100%; border-radius: 10px;}
.gifs figcaption {font-size: .9em; opacity: .75; margin-top: 4px;}
.legend span {display: inline-block; margin-right: 14px;}
"""


def url(name: str) -> str:
    return f"/gradio_api/file={ASSETS / name}"


def stats_html() -> str:
    size = tools.size_breakdown()["recall_by_size"]["person"]
    cards = [
        (f"{size['small']['recall']:.0%} vs {size['large']['recall']:.0%}", "small vs large persons found"),
        (f"{size['small']['missed']} of {size['small']['num_gt']}", "small persons missed"),
        ("3 AI agents", "explain the failures, a checker verifies every claim"),
    ]
    return '<div class="stats">' + "".join(
        f'<div class="card"><b>{big}</b><span>{small}</span></div>' for big, small in cards) + "</div>"


def gallery_html() -> str:
    items = "".join(
        f'<figure><img src="{url(Path(name).stem + ".gif")}" alt="{name}">'
        f"<figcaption>{name}: {caption}</figcaption></figure>"
        for name, caption in CAPTIONS.items())
    keys = [("#3498db", "detected by YOLOv8n"), ("#2ecc71", "found"), ("#e74c3c", "missed"),
            ("#f39c12", "false alarm")]
    legend = '<p class="legend">' + "".join(
        f'<span><i style="display:inline-block;width:12px;height:12px;background:{c};'
        f'margin-right:6px"></i>{label}</span>' for c, label in keys) + "</p>"
    return legend + f'<div class="gifs">{items}</div>'


def numbers_md() -> str:
    data = tools.size_breakdown()
    rows = "\n".join(
        f"| {c} | {s['small']['recall']} | {s['medium']['recall']} | {s['large']['recall']} |"
        for c, s in data["recall_by_size"].items())
    confusions = "\n".join(f"- {c}" for c in tools.confusions()["confusions"])
    return ("| Class | Small | Medium | Large |\n|---|---:|---:|---:|\n" + rows +
            "\n\n**Mixed-up classes**\n\n" + confusions)


AGENTS = """Three AI agents (LangGraph, Llama 3.1 8B) each explain one part of the failure:
object size, mixed-up classes, hard photos. An editor writes the summary. A checker compares
every claim with the measured numbers and sends wrong parts back to the agent that wrote them.

The agents get their numbers from tools like the ones on this site: connect any MCP client to
`/gradio_api/mcp/` (tools: `size_breakdown`, `confusions`, `false_alarms`, `hard_examples`).

Code: [perception-triage-agent](https://github.com/knh4abt/perception-triage-agent)"""

with gr.Blocks(title="Where does YOLOv8n fail?") as demo:
    gr.Markdown("# Where does YOLOv8n fail?\nAn object detector on 200 street photos, "
                "and AI agents that explain where it goes wrong.")
    gr.HTML(stats_html())

    gr.Markdown("## Watch the detector at work\nEach photo: the photo, what YOLOv8n detected, "
                "then the check against the real labels.")
    gr.HTML(gallery_html())

    gr.Markdown("## The pattern: small objects get missed")
    gr.HTML(f'<img src="{url("recall_by_size.png")}" style="width:100%;max-width:820px">')

    gr.Markdown("## The agents explain it, the checker keeps them honest")
    gr.HTML(f'<img src="{url("agent-story.gif")}" style="width:100%;max-width:640px;border-radius:10px">'
            "<p>A real catch: an agent wrote that the hard photos are darker; the measured "
            "brightness says brighter, so the checker sent it back.</p>")
    gr.Markdown(AGENTS)

    with gr.Accordion("All numbers", open=False):
        gr.Markdown(numbers_md())

    # Hidden from the page, visible to agents: each function becomes one MCP tool.
    for fn in tools.METRICS_TOOLS:
        gr.api(fn, api_name=fn.__name__)


if __name__ == "__main__":
    demo.launch(mcp_server=True, css=CSS, theme=gr.themes.Soft(primary_hue="orange"))
