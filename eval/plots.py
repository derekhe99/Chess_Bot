"""Elo vs. training compute -- the headline chart (plan Step 11), drawn from results.csv only.

Intent: after every training run, redraw one picture showing where each run's
checkpoints land. x = total training GPU-hours (an RL run's points include the
GPU-hours of the SFT checkpoint it started from; an AlphaZero run starts from
nothing, at 0), y = Elo with its 95% CI. Each run keeps one color, in a fixed
order, so adding a run never repaints the others.

What it reads (eval.anchors.RESULTS_COLUMNS):
- model rows: one point each, grouped by ``run``, ordered by ``train_gpu_hours``. The
  run's folder prefix says its kind (run_kind): ``llm_rl/`` = RL, ``alphazero/`` =
  AlphaZero, anything else = SFT.
- reference rows whose spec is an SFT checkpoint (e.g. step_20000): drawn as a hollow
  marker on their run. Their GPU-hours aren't stored, so they're interpolated by step
  from that run's model rows.
- an RL row whose note says "from sft <run>@<step>" joins its run (dotted) to that point.
- other pin/reference rows rated near the plotted range: dashed horizontal lines.

    from eval.plots import plot_elo_vs_compute
    fig = plot_elo_vs_compute(f"{ROOT}/results/results.csv", f"{ROOT}/results/elo_vs_compute.png")
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from matplotlib.figure import Figure

# Categorical colors in fixed order (a colorblind-checked palette); run i gets COLORS[i].
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
INK, INK2, GRID, BG = "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
_FROM_SFT = re.compile(r"from sft (\S+)@(\d+)")


def _num(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def run_kind(run: str) -> str:
    """'RL', 'AlphaZero' or 'SFT', from the run's folder prefix in results.csv."""
    if run.startswith("llm_rl/"):
        return "RL"
    if run.startswith("alphazero/"):
        return "AlphaZero"
    return "SFT"


def _label(run: str, step: float) -> str:
    return f"{step / 1000:g}k" if run_kind(run) == "SFT" else f"r{step:g}"   # SFT steps vs. RL / AlphaZero rounds


def elo_points(results_path: str | Path) -> tuple[dict[str, list[dict]], dict[str, tuple], list[dict]]:
    """Read results.csv into what the chart draws.

    Returns (runs, starts, lines):
      runs   -- {run: [point]} in first-seen order; a point is {x, y, lo, hi, step, label, hollow, name}
      starts -- {RL run: (x, y)} of the SFT checkpoint it started from (when its note says so)
      lines  -- other pin/reference rows: [{name, y}]
    """
    with open(results_path, newline="") as f:
        rows = list(csv.DictReader(f))
    runs: dict[str, list[dict]] = {}
    notes: dict[str, str] = {}
    for r in rows:
        x, y, step = _num(r["train_gpu_hours"]), _num(r["rating"]), _num(r["step"])
        if r["role"] != "model" or not r["run"] or x is None or y is None:
            continue
        runs.setdefault(r["run"], []).append({"x": x, "y": y, "lo": _num(r["ci_low"]), "hi": _num(r["ci_high"]),
                                              "step": step, "label": _label(r["run"], step or 0),
                                              "hollow": False, "name": r["name"]})
        notes.setdefault(r["run"], r.get("note") or "")
    lines = []
    for r in rows:
        if r["role"] not in ("pin", "reference"):
            continue
        spec = json.loads(r["spec"]) if r["spec"] else {}
        pts = runs.get(spec.get("run", ""), [])
        known = sorted((p["step"], p["x"]) for p in pts if p["step"] is not None)
        if spec.get("kind") == "sft_checkpoint" and len(known) >= 2:
            steps, hours = zip(*known)
            pts.append({"x": float(np.interp(spec["step"], steps, hours)), "y": float(r["rating"]),
                        "lo": _num(r["ci_low"]), "hi": _num(r["ci_high"]), "step": float(spec["step"]),
                        "label": _label("", spec["step"]), "hollow": True, "name": r["name"]})
        else:
            lines.append({"name": r["name"], "y": float(r["rating"])})
    for pts in runs.values():
        pts.sort(key=lambda p: p["x"])
    starts = {}
    for run, note in notes.items():
        m = _FROM_SFT.search(note)
        if m:
            src = [p for p in runs.get(m.group(1), []) if p["step"] == float(m.group(2))]
            if src:
                starts[run] = (src[0]["x"], src[0]["y"])
    return runs, starts, lines


def plot_elo_vs_compute(results_path: str | Path, out_path: str | Path | None = None,
                        title: str = "Elo vs. training compute") -> Figure:
    """Draw (and optionally save as PNG) Elo vs. total training GPU-hours for every run in results.csv."""
    runs, starts, lines = elo_points(results_path)
    fig = Figure(figsize=(10, 5.5), dpi=150, facecolor=BG)
    ax = fig.add_subplot(facecolor=BG)
    if not runs:
        ax.text(0.5, 0.5, "no model rows in results.csv yet", ha="center", va="center", color=INK2)
    lows, highs, xmax = [], [], 1.0
    for i, (run, pts) in enumerate(runs.items()):
        color = COLORS[i % len(COLORS)]
        kind = run_kind(run)
        xs, ys = [p["x"] for p in pts], [p["y"] for p in pts]
        ax.plot(xs, ys, color=color, lw=2, label=f"{kind}: {run.split('/')[-1]}", zorder=2)
        if run in starts:
            sx, sy = starts[run]
            ax.plot([sx, xs[0]], [sy, ys[0]], color=color, lw=1, ls=":", zorder=1)
        for p in pts:
            if p["lo"] is not None and p["hi"] is not None:
                ax.errorbar(p["x"], p["y"], yerr=[[p["y"] - p["lo"]], [p["hi"] - p["y"]]], color=color,
                            capsize=4, elinewidth=1.2, lw=0, zorder=2, alpha=0.45 if p["hollow"] else 1.0)
                if not p["hollow"]:     # a frozen reference's wide CI shouldn't set the plotted range
                    lows.append(p["lo"])
                    highs.append(p["hi"])
            ax.plot(p["x"], p["y"], "o", ms=8, mew=2, color=color, mec=color if p["hollow"] else BG,
                    mfc=BG if p["hollow"] else color, zorder=3)
            top = p["hi"] if p["hi"] is not None and not p["hollow"] else p["y"]
            ax.annotate(p["label"], (p["x"], top), xytext=(0, 5 if top != p["y"] else 9),
                        textcoords="offset points", ha="center", fontsize=8, color=INK2)
            xmax = max(xmax, p["x"])
    lo = min(lows, default=600.0) - 20
    hi = max(highs, default=800.0) + 20
    shown = [l for l in lines if lo - 40 <= l["y"] <= hi + 40]
    for l in shown:
        ax.hlines(l["y"], 0, xmax * 1.03, color=INK2, lw=1, ls="--", zorder=1)
        ax.text(xmax * 1.05, l["y"], f"{l['name']} {l['y']:.0f}", va="center", fontsize=8, color=INK2)
    ys = [lo, hi] + [l["y"] for l in shown]
    ax.set_ylim(min(ys) - 10, max(ys) + 25)
    ax.set_xlim(0, xmax * 1.22)
    ax.set_xlabel("Total training GPU-hours (SFT + RL; AlphaZero from 0)", color=INK2)
    ax.set_ylabel("Elo (95% CI)", color=INK2)
    ax.set_title(title, loc="left", fontsize=12, color=INK)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(colors=INK2)
    if runs:
        ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path, facecolor=BG)
    return fig
