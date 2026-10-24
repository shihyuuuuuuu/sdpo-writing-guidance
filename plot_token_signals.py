#!/usr/bin/env python3
"""Visualize saved simple-SDPO token signals.

Example:
  python plot_token_signals.py \
    --train_steps eval/eval-online-sdpo-.../train_steps.jsonl \
    --step 1 --example 0 --out_dir /tmp/sdpo-token-plot

The HTML output is the preferred view for mixed Chinese/English text.  The
PNG is a compact heatmap overview; token ids and numeric signals remain in
the HTML title attributes and the original JSONL record.
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def load_records(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def resolve_train_steps(repo_root: Path, run_id: str | None, explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser()
    elif run_id:
        path = repo_root / "eval" / run_id / "train_steps.jsonl"
    else:
        candidates = list((repo_root / "eval").glob("*/train_steps.jsonl"))
        if not candidates:
            raise FileNotFoundError(f"No train_steps.jsonl found under {repo_root / 'eval'}")
        path = max(candidates, key=lambda item: item.stat().st_mtime)
    if not path.is_file():
        raise FileNotFoundError(f"Training log not found: {path}")
    return path


def select_debug(records: list[dict], step: int | None, example: int) -> tuple[dict, dict]:
    record = records[-1] if step is None else next(
        (r for r in records if r.get("step") == step), None
    )
    if record is None:
        available = [r.get("step") for r in records]
        raise ValueError(f"step={step} not found; available steps: {available}")

    debug = record.get("token_debug")
    if not debug:
        raise ValueError(
            "This train_steps.jsonl has no token_debug. Run a new simple_signal "
            "training job after token logging was added."
        )
    if not 0 <= example < len(debug):
        raise ValueError(f"example must be in [0, {len(debug) - 1}]")
    return record, debug[example]


def color(signal: float, scale: float) -> str:
    # Blue = positive/reinforced, red = negative/penalized, white = neutral.
    x = max(-1.0, min(1.0, signal / scale)) if scale else 0.0
    if x >= 0:
        alpha = 0.10 + 0.75 * x
        return f"rgba(30, 110, 220, {alpha:.3f})"
    alpha = 0.10 + 0.75 * (-x)
    return f"rgba(220, 60, 55, {alpha:.3f})"


def robust_scale(signals: list[float]) -> float:
    """Use P95(|signal|), so a few outliers do not wash out the plot."""
    values = sorted(abs(float(value)) for value in signals)
    if not values:
        return 1e-6
    index = int(0.95 * (len(values) - 1))
    return max(values[index], 1e-6)


def find_plot_font():
    try:
        from matplotlib import font_manager

        candidates = [
            "Noto Sans CJK TC", "Noto Sans CJK SC", "Noto Sans CJK JP",
            "Microsoft JhengHei", "WenQuanYi Zen Hei", "Arial Unicode MS",
        ]
        for name in candidates:
            path = font_manager.findfont(
                font_manager.FontProperties(family=name),
                fallback_to_default=False,
            )
            if path:
                return name
    except (ImportError, ValueError):
        pass
    return None


def write_html(record: dict, debug: dict, path: Path, example: int) -> None:
    texts = debug["token_text"]
    signals = debug["signal"]
    ids = debug["token_ids"]
    scale = robust_scale([float(x) for x in signals])
    outliers = sum(abs(float(x)) > scale for x in signals)

    spans = []
    for i, (text, signal, token_id) in enumerate(zip(texts, signals, ids)):
        shown = html.escape(text) or "∅"
        title = html.escape(f"index={i} token_id={token_id} signal={float(signal):+.6f}")
        spans.append(
            f'<span class="tok" style="background:{color(float(signal), scale)}" '
            f'title="{title}">{shown}</span>'
        )

    body = "".join(spans)
    page = f"""<!doctype html>
<meta charset="utf-8">
<title>SDPO token signals — step {record.get('step')}</title>
<style>
body {{ font-family: system-ui, "Noto Sans CJK TC", "Noto Sans CJK SC", sans-serif; margin: 2rem; line-height: 2.2; }}
.meta {{ color: #555; margin-bottom: 1rem; }}
.text {{ font-family: ui-monospace, SFMono-Regular, monospace; white-space: pre-wrap;
         border: 1px solid #ddd; padding: 1rem; line-height: 2.4; }}
.tok {{ padding: .18rem .08rem; border-radius: .2rem; cursor: help; }}
.context {{ white-space: pre-wrap; border: 1px solid #ddd; padding: 1rem; background: #fafafa; }}
.label {{ font-weight: 700; margin-top: 1.2rem; }}
pre {{ max-width: 100%; white-space: pre-wrap; overflow-wrap: anywhere; word-break: break-word; }}
.decoded {{
  font-size: 1.05rem;
  line-height: 1.8;
  white-space: pre-wrap;
  overflow-wrap: anywhere;
  word-break: break-word;
  max-width: 100%;
}}
.legend {{ margin-top: 1rem; color: #555; }}
</style>
<h2>SDPO token signals</h2>
<div class="meta">step={record.get('step')} &nbsp; example={example} &nbsp;
visual scale=±{scale:.4f} (P95 of |signal|; values beyond this are shown at maximum color intensity)
&nbsp; red=negative, blue=positive
&nbsp; outliers clipped for color={outliers}</div>
<div class="label">User prompt / context</div>
<div class="context">{html.escape(str(record.get('interactions', [{}])[example].get('p_context', '')))}</div>
<div class="label">User feedback (o)</div>
<div class="context">{html.escape(str(record.get('interactions', [{}])[example].get('user_follow_up', '')))}</div>
<div class="label">Token signal</div>
<div class="text">{body}</div>
<div class="legend">Hover a token to see its index, token id, and exact signal. Color is clipped only for readability; the exact value is in the tooltip.</div>
<h3>Decoded completion</h3>
<pre class="decoded">{html.escape(debug.get('completion_text', ''))}</pre>
"""
    path.write_text(page, encoding="utf-8")


def write_png(record: dict, debug: dict, path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.colors import TwoSlopeNorm
    except ImportError:
        print("matplotlib is not installed; skipped PNG (HTML was still written).")
        return

    signals = [float(x) for x in debug["signal"]]
    texts = debug["token_text"]
    if not signals:
        return
    scale = robust_scale(signals)
    font_name = find_plot_font()
    if font_name:
        plt.rcParams["font.sans-serif"] = [font_name, "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    fig_width = max(12.0, min(36.0, len(signals) * 0.22))
    fig, ax = plt.subplots(figsize=(fig_width, 2.8))
    norm = TwoSlopeNorm(vmin=-scale, vcenter=0.0, vmax=scale)
    ax.imshow([signals], aspect="auto", cmap="RdBu", norm=norm)
    ax.set_yticks([0])
    ax.set_yticklabels([f"step {record.get('step')}"])
    step = max(1, len(signals) // 30)
    positions = list(range(0, len(signals), step))
    ax.set_xticks(positions)
    ax.set_xticklabels([texts[i] or "∅" for i in positions], rotation=65, ha="right", fontsize=8)
    ax.set_xlabel("decoded token pieces; color scale uses P95(|signal|); exact values are in HTML")
    fig.colorbar(ax.images[0], ax=ax, label="log p(y|x,o) − log p(y|x)")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_id", default=None, help="Evaluation run directory name")
    parser.add_argument("--train_steps", type=Path, default=None, help="Optional explicit JSONL path")
    parser.add_argument("--step", type=int, default=None, help="Default: last saved step")
    parser.add_argument("--example", type=int, default=0, help="Example index inside the batch")
    parser.add_argument("--out_dir", type=Path, default=None)
    args = parser.parse_args()

    train_steps = resolve_train_steps(repo_root, args.run_id, args.train_steps)
    if args.out_dir is None:
        args.out_dir = train_steps.parent / "token_plots"
    records = load_records(train_steps)
    record, debug = select_debug(records, args.step, args.example)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"step{record['step']}_example{args.example}"
    html_path = args.out_dir / f"{stem}.html"
    png_path = args.out_dir / f"{stem}.png"
    write_html(record, debug, html_path, args.example)
    write_png(record, debug, png_path)
    print(f"wrote {html_path}")
    if png_path.is_file():
        print(f"wrote {png_path}")


if __name__ == "__main__":
    main()
