"""Regenerate the data-driven figures used by docs/PIPELINE.md (written to docs/img/).

Needs the nflverse data (`gridline nflverse pull`, `gridline events build --seasons
1999-2026`). The latency chart also needs the 2025 Kalshi pull (`gridline kalshi pull
--season 2025`), the calibration chart in docs/RESULTS.md needs the held-out
predictions from `gridline wp cv`, the simulator charts need `gridline sim gate
--season 2025`, and the market charts need the consolidated Kalshi tables (`gridline
kalshi consolidate --season 2025`), the replay (`gridline replay run --season 2025`) and
the benchmark (`gridline market benchmark --season 2025`); each is skipped until its
input exists.

    uv run --group docs python scripts/make_figures.py

The diagrams in docs/img/ (pipeline, timing, simulator) are hand-drawn SVG, not
generated here.
"""

from __future__ import annotations

import datetime as dt
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from gridline import config  # noqa: E402
from gridline.data import kalshi_pull, nflverse  # noqa: E402
from gridline.eval import metrics  # noqa: E402
from gridline.inventory import events_path, load_events  # noqa: E402
from gridline.models import wp  # noqa: E402
from gridline.state.events import timing_summary  # noqa: E402

OUT = config.REPO_ROOT / "docs" / "img"
ET = ZoneInfo(config.EASTERN_TZ)
UTC = dt.timezone.utc

GAME = "2025_01_BAL_BUF"  # BUF 41, BAL 40: the week-1 comeback
HOME_TICKER = "KXNFLGAME-25SEP07BALBUF-BUF"
FUMBLE, NEXT_PLAY = 4126.0, 4166.0  # Henry fumble (3:10 left), Allen 29-yd pass

# Reference palette (light surface) from the dataviz method.
SURFACE, INK, INK2, MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, AXIS, BAND = "#e1e0d9", "#c3c2b7", "#f0efec"
BLUE, ORANGE = "#2a78d6", "#eb6834"
FONT_STACK = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"


def style() -> None:
    plt.rcParams.update({
        "svg.fonttype": "none",  # keep text as text; the font stack is swapped in on save
        "svg.hashsalt": "gridline",  # stable ids, so an unchanged chart redraws byte for byte
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.75,
        "axes.labelcolor": INK2,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": False,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.75,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK2,
        "ytick.labelcolor": INK2,
        "xtick.major.size": 0,
        "ytick.major.size": 0,
        "xtick.major.pad": 6,
        "ytick.major.pad": 6,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.solid_capstyle": "round",
        "lines.solid_joinstyle": "round",
    })


def titles(fig, title: str, subtitle: str) -> None:
    fig.text(0.0, 1.0, title, ha="left", va="bottom", fontsize=12.5, color=INK,
             fontweight="bold", transform=fig.transFigure)
    fig.text(0.0, 0.985, subtitle, ha="left", va="top", fontsize=10, color=INK2,
             transform=fig.transFigure)


def save(fig, name: str) -> Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{name}.svg"
    fig.savefig(path, bbox_inches="tight", pad_inches=0.3, metadata={"Date": None})
    plt.close(fig)
    svg = re.sub(r"font-family: [^;\"]+", f"font-family: {FONT_STACK}", path.read_text())
    path.write_text(svg)
    print("wrote", path.relative_to(config.REPO_ROOT))
    return path


def note(ax, text, xy, xytext, ha="left"):
    ax.annotate(text, xy=xy, xytext=xytext, textcoords="data", ha=ha, va="center",
                fontsize=9.5, color=INK2,
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.75, shrinkA=2, shrinkB=3))


# ------------------------------------------------------------------------------- data


def game_events() -> pl.DataFrame:
    ev = load_events(2025).filter(pl.col("game_id") == GAME)
    wp = nflverse.load_pbp([2025], columns=["game_id", "play_id", "vegas_home_wp"])
    return ev.join(wp, on=["game_id", "play_id"], how="left")


def kalshi_game(game_id: str = GAME) -> tuple[pl.DataFrame, pl.DataFrame] | None:
    """(1-minute candles, per-second trade summary) for the home winner contract,
    from the Kalshi pull. None if the game hasn't been pulled yet."""
    gdir = kalshi_pull.game_dir(2025, game_id)
    if not (gdir / "candles.parquet").exists() or not (gdir / "trades.parquet").exists():
        return None
    candles = (pl.read_parquet(gdir / "candles.parquet")
               .filter(pl.col("ticker") == HOME_TICKER)
               .select("end_ts", pl.col("bid_close").alias("bid"), pl.col("ask_close").alias("ask")))
    trades = pl.read_parquet(gdir / "trades.parquet").filter(pl.col("ticker") == HOME_TICKER)
    return candles, per_second(trades)


def per_second(trades: pl.DataFrame) -> pl.DataFrame:
    """Trades -> one row per second with trades: count, contracts, VWAP, last price."""
    return (
        trades.sort("ts")
        .with_columns(sec=pl.col("ts").dt.truncate("1s"))
        .group_by("sec", maintain_order=True)
        .agg(n=pl.len(), contracts=pl.col("count").sum(),
             vwap=(pl.col("yes_price") * pl.col("count")).sum() / pl.col("count").sum(),
             last=pl.col("yes_price").last())
    )


# ---------------------------------------------------------------------------- figures


def fig_market_vs_model(ev: pl.DataFrame) -> Path | None:
    """Phase 3: Kalshi's price and gridline's own, through the example game."""
    from gridline.engine import sinks
    from gridline.eval import market

    log = sinks.run_dir("replay_2025") / "prices" / f"{GAME}.parquet"
    if not log.exists() or not kalshi_pull.consolidated_path("candles", 2025).exists():
        print("skipped market_vs_model: needs `gridline kalshi consolidate --season 2025` and "
              "`gridline replay run --season 2025`", file=sys.stderr)
        return None
    q = market.winner_minutes(2025).filter(pl.col("game_id") == GAME)
    mid_t = [t.astimezone(ET) for t in q["t"]]
    mid = (q["market"] * 100).to_list()
    prices = pl.read_parquet(log).sort("order")
    model_t = [t.astimezone(ET) for t in prices["t"]]
    model = (prices["p_home"] * 100).to_list()

    fig, ax = plt.subplots(figsize=(9.6, 4.1))
    ax.step(mid_t, mid, where="post", color=BLUE, lw=1.5, label="Kalshi price for BUF to win (1-minute mid)")
    ax.step(model_t, model, where="post", color=ORANGE, lw=1.5,
            label="gridline's price for BUF to win (after every play)")
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(dt.datetime(2025, 9, 7, 20, 0, tzinfo=ET), dt.datetime(2025, 9, 7, 23, 48, tzinfo=ET))
    ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30], tz=ET))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I:%M", tz=ET))
    ax.set_xlabel("Eastern time, Sunday Sept 7 2025 (pm)")

    def at(pid, col="t_end"):
        return ev.filter(pl.col("play_id") == pid)[col][0].astimezone(ET)

    note(ax, "Henry TD: BAL leads 40–25, 11:50 left", (at(3313.0), 5),
         (at(3313.0) - dt.timedelta(minutes=8), 40), ha="right")
    note(ax, "Henry fumbles, 3:10 left", (at(FUMBLE), 15),
         (at(FUMBLE) - dt.timedelta(minutes=9), 58), ha="right")
    note(ax, "Field goal at 0:00: BUF wins 41–40", (at(4673.0), 99),
         (at(4673.0) - dt.timedelta(minutes=9), 84), ha="right")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncols=2, handlelength=1.6)
    titles(fig, "One real game: the market's price and gridline's, side by side",
           "BAL at BUF, week 1 2025. Both start from Kalshi's price at kickoff; after that, gridline sees only "
           "the plays.")
    return save(fig, "market_vs_model")


def fig_latency(sec: pl.DataFrame, ev: pl.DataFrame) -> Path:
    def row(pid):
        return ev.filter(pl.col("play_id") == pid).row(0, named=True)

    f, nxt = row(FUMBLE), row(NEXT_PLAY)
    t0 = f["t_end"]
    x = [((s - t0).total_seconds() + 0.5) for s in sec["sec"]]
    y = (sec["vwap"] * 100).to_list()

    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    for p, label, ha in [(f, "fumble play ", "right"), (nxt, "29-yard pass", "center")]:
        a, b = (p["t_snap"] - t0).total_seconds(), (p["t_end"] - t0).total_seconds()
        ax.axvspan(a, b, color=BAND, lw=0, zorder=0)
        ax.text(a if ha == "right" else (a + b) / 2, 41, label, ha=ha, va="bottom",
                fontsize=9, color=INK2)
    ax.axvline(0, color=MUTED, lw=0.75, zorder=1)

    ax.plot(x, y, color=BLUE, lw=1.5, label="Kalshi trades, BUF to win (volume-weighted price per second)")
    wp_before, wp_after, wp_next = (ev.filter(pl.col("play_id") == p)["vegas_home_wp"][0] * 100
                                    for p in (FUMBLE, NEXT_PLAY, 4191.0))
    t_next = (nxt["t_end"] - t0).total_seconds()
    ax.step([-75, 0, t_next, 135], [wp_before, wp_after, wp_next, wp_next], where="post",
            color=ORANGE, lw=1.5, label="nflfastR win probability, updated the moment each play ends")

    jump = sec.filter(pl.col("vwap") > 0.15)["sec"][0]
    lag = (jump - t0).total_seconds()
    note(ax, f"first trades at the new price: {lag:.1f}–{lag + 1:.1f} s after the play ends",
         (lag + 0.5, 22), (14, 36))
    note(ax, "NFL end-of-play timestamp", (0, 8), (-58, 30))
    ax.set_xlim(-75, 135)
    ax.set_ylim(0, 44)
    ax.set_yticks([0, 10, 20, 30, 40], ["0¢", "10¢", "20¢", "30¢", "40¢"])
    ax.set_xticks(range(-60, 136, 30), [f"{s:+d} s" if s else "0" for s in range(-60, 136, 30)])
    ax.set_xlabel("seconds from the end of the fumble play (3:19:15 am UTC)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=1, handlelength=1.6)
    titles(fig, "The market repriced about two seconds after the play ended",
           "Henry's fumble with 3:10 left: price moves, measured against the NFL's own play timestamps.")
    return save(fig, "latency")


def fig_timestamp_cleaning(ev: pl.DataFrame) -> Path:
    raw = nflverse.load_pbp([2025], columns=["game_id", "play_id", "time_of_day"]).filter(
        pl.col("game_id") == GAME)
    rows = (ev.filter(pl.col("play_id").is_between(1665, 1833))
            .join(raw, on=["game_id", "play_id"])
            .with_columns(raw_snap=pl.col("time_of_day").str.strip_suffix("Z")
                          .str.to_datetime("%Y-%m-%dT%H:%M:%S%.f").dt.replace_time_zone("UTC"))
            .sort("play_id"))
    labels = []
    for r in rows.iter_rows(named=True):
        if r["play_type"] == "no_play" and r["desc"].startswith("Timeout"):
            team = r["desc"].split(" by ")[1].split(" ")[0]
            labels.append(f"Timeout {team}")
        else:
            clock = r["desc"].split(")")[0].lstrip("(")
            labels.append(f"{r['play_type'].replace('_', ' ').capitalize()}, {clock}")

    n = rows.height
    fig, ax = plt.subplots(figsize=(9.6, 4.4))
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", visible=True)
    for i, r in enumerate(rows.iter_rows(named=True)):
        y = n - 1 - i
        rt, ct = r["raw_snap"].astimezone(ET), r["t_snap"].astimezone(ET)
        if r["snap_out_of_order"]:
            ax.annotate("", xy=(ct, y), xytext=(rt, y),
                        arrowprops=dict(arrowstyle="-|>", color=MUTED, lw=1, shrinkA=5, shrinkB=5,
                                        mutation_scale=9))
            ax.plot([rt], [y], "o", ms=7, color=ORANGE, mec=SURFACE, mew=1.5, zorder=3)
            ax.plot([ct], [y], "o", ms=7, mfc=SURFACE, mec=INK2, mew=1.2, zorder=3)
        else:
            ax.plot([rt], [y], "o", ms=7, color=BLUE, mec=SURFACE, mew=1.5, zorder=3)
    ax.set_yticks(range(n), labels[::-1])
    ax.tick_params(axis="y", labelcolor=INK2)
    ax.set_ylim(-0.7, n - 0.3)
    ax.xaxis.set_major_locator(mdates.MinuteLocator(tz=ET))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%-I:%M", tz=ET))
    ax.set_xlabel("snap time, Eastern (pm)")
    ax.legend(handles=[
        Line2D([], [], ls="", marker="o", ms=7, color=BLUE, mec=SURFACE, label="logged snap time, in order"),
        Line2D([], [], ls="", marker="o", ms=7, color=ORANGE, mec=SURFACE,
               label="logged earlier than the row above it"),
        Line2D([], [], ls="", marker="o", ms=7, mfc=SURFACE, mec=INK2, label="where cleaning puts it"),
    ], loc="upper left", bbox_to_anchor=(0.0, -0.13), ncols=3, handletextpad=0.3, columnspacing=1.6)
    titles(fig, "Timeouts arrive out of order. Cleaning moves them later, never earlier",
           "BAL at BUF, last two minutes of the first half, rows in the order nflverse logs them.")
    return save(fig, "timestamp_cleaning")


def fig_timing_coverage() -> Path:
    seasons = [s for s in range(config.FIRST_PBP_SEASON, 2027) if events_path(s).exists()]
    if len(seasons) < 2027 - config.FIRST_PBP_SEASON:
        print(f"timing_coverage: only {len(seasons)} seasons of events built; "
              "run `gridline events build --seasons 1999-2026` for the full chart", file=sys.stderr)
    ts = timing_summary(pl.concat([load_events(s) for s in seasons], how="diagonal_relaxed"))
    x = ts["season"].to_list()
    fig, ax = plt.subplots(figsize=(9.6, 3.6))
    ax.axvspan(2024.5, 2026.5, color=BAND, lw=0, zorder=0)
    ax.text(2025.5, 50, "Kalshi NFL\nmarkets", ha="center", va="center", fontsize=9, color=INK2)
    ax.step(x, (ts["snap_raw"] * 100).to_list(), where="mid", color=BLUE, lw=1.5,
            label="snap time taken from the raw data")
    ax.step(x, (ts["end_raw"] * 100).to_list(), where="mid", color=ORANGE, lw=1.5,
            label="play-end time taken from the raw data")
    ax.set_ylim(0, 105)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xlim(1998.3, 2026.7)
    ax.set_xticks(range(2000, 2027, 5))
    ax.set_xlabel("season")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncols=2, handlelength=1.6)
    titles(fig, "Snap times exist from 2003, play-end times only from 2022",
           "Share of plays per season whose timestamp comes straight from nflverse (the rest are imputed).")
    return save(fig, "timing_coverage")


def fig_contract_prices() -> Path:
    sched = nflverse.load_schedules(list(range(1999, 2026))).filter(pl.col("result").is_not_null())
    fav3 = sched.filter(pl.col("spread_line").is_between(2.5, 3.5))["result"].to_list()
    tot45 = sched.filter(pl.col("total_line").is_between(44.5, 46.5))["total"].to_list()

    def hist(values, lo, hi):
        xs = list(range(lo, hi + 1))
        n = len(values)
        return xs, [sum(1 for v in values if v == k) / n * 100 for k in xs]

    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.6), sharey=False)
    panels = [
        (fav3, (-28, 31), lambda k: k > 0, "Winner (home)",
         "P(margin > 0) + ½·P(tie)", lambda v: sum(k > 0 for k in v) / len(v) + 0.5 * sum(k == 0 for k in v) / len(v)),
        (fav3, (-28, 31), lambda k: k > 3.5, "Spread: home wins by over 3.5",
         "P(margin > 3.5)", lambda v: sum(k > 3.5 for k in v) / len(v)),
        (tot45, (20, 75), lambda k: k > 45.5, "Total: over 45.5 points",
         "P(total > 45.5)", lambda v: sum(k > 45.5 for k in v) / len(v)),
    ]
    for ax, (vals, (lo, hi), on, name, formula, price) in zip(axes, panels):
        xs, ys = hist(vals, lo, hi)
        ax.bar(xs, ys, width=0.72, color=[BLUE if on(k) else AXIS for k in xs], lw=0)
        ax.set_title(f"{name}\n{formula} = {price(vals):.1%}", loc="left", fontsize=10, color=INK, pad=8)
        ax.set_ylim(0, 11)
        ax.set_yticks([0, 5, 10], ["0%", "5%", "10%"])
    for ax, xlabel in zip(axes, ["final margin, home − away", "final margin, home − away",
                                 "final total points"]):
        ax.set_xlabel(xlabel)
    for ax in axes[:2]:
        for k in (3, 7):
            share = sum(v == k for v in fav3) / len(fav3) * 100
            ax.text(k, share + 0.3, str(k), ha="center", va="bottom", fontsize=8.5, color=INK2)
    fig.subplots_adjust(wspace=0.28, top=0.70)
    titles(fig, "A contract's fair price is the probability on one side of its strike",
           f"Final scores of {len(fav3):,} games with the home team favoured by 2.5–3.5 (left, middle) and "
           f"{len(tot45):,} games with a 44.5–46.5 total (right), 1999–2025.")
    return save(fig, "contract_prices")


def fig_wp_calibration() -> Path | None:
    """Phase 1: actual win rate vs predicted, by quarter, for the leave-one-season-out
    predictions that `gridline wp cv` saves."""
    kinds = [("wp_spread", BLUE, "with the betting spread (wp_spread)"),
             ("wp", ORANGE, "without it (wp)")]
    if not all(wp.cv_path(k).exists() for k, _, _ in kinds):
        print("skipped wp_calibration: run `gridline wp cv` first", file=sys.stderr)
        return None
    fig, axes = plt.subplots(1, 4, figsize=(10.2, 3.3), sharey=True)
    ticks, tick_labels = [0, 0.5, 1], ["0%", "50%", "100%"]
    for q, ax in enumerate(axes, start=1):
        ax.plot([0, 1], [0, 1], color=AXIS, lw=1, zorder=1)
        ax.grid(True, axis="both")
        ax.set_aspect("equal")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(ticks, tick_labels)
        ax.set_yticks(ticks, tick_labels)
        ax.set_title(["1st", "2nd", "3rd", "4th"][q - 1] + " quarter", loc="left", fontsize=10,
                     color=INK, pad=6)
        ax.set_xlabel("predicted")
    axes[0].set_ylabel("actual win rate")
    handles = []
    for kind, color, label in kinds:
        cv = pl.read_parquet(wp.cv_path(kind))
        table = metrics.calibration_table(cv["wp"], cv["label"], cv["qtr"])
        biggest = table["n_plays"].max()
        filled = kind == "wp_spread"
        for q, ax in enumerate(axes, start=1):
            t = table.filter(pl.col("qtr") == q)
            size = (6 + 70 * t["n_plays"] / biggest).to_list()
            ax.scatter(t["bin"].to_list(), t["actual"].to_list(), s=size, zorder=3 if filled else 4,
                       color=color if filled else "none", edgecolors=color,
                       linewidths=0 if filled else 1.0, alpha=0.85 if filled else 1.0)
        handles.append(Line2D([], [], ls="", marker="o", markersize=7, label=label,
                              color=color, markerfacecolor=color if filled else "none"))
    seasons = cv["season"]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.01), ncols=2,
               handletextpad=0.4)
    fig.subplots_adjust(wspace=0.12, top=0.82, bottom=0.14)
    titles(fig, "Held-out win probabilities are calibrated in every quarter",
           f"Actual win rate against predicted win probability, {seasons.min()}–{seasons.max()}. "
           f"Each play is predicted by a model that never saw its season. Dot area ∝ plays.")
    return save(fig, "wp_calibration")


AQUA = "#1baf7a"
SIM_SEASON = 2025
SIM_GAME = "2025_01_BAL_BUF"  # the comeback the market charts in PIPELINE.md show


def fig_sim_pit(season: int = SIM_SEASON) -> Path | None:
    """Phase 2: where each final margin and total fell inside the predicted distribution."""
    from gridline.eval import distribution as dist
    from gridline.eval import sim_gate

    path = sim_gate.eval_path(season)
    if not path.exists():
        print(f"skipped sim_pit: run `gridline sim gate --season {season}` first", file=sys.stderr)
        return None
    df = pl.read_parquet(path)
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.6), sharey=True)
    x = [i / 10 + 0.05 for i in range(10)]
    for ax, what, name in ((axes[0], "margin", "Final margin"), (axes[1], "total", "Final total")):
        for dx, col, color, label in ((-0.021, f"pit_{what}", BLUE, "simulator"),
                                      (0.021, f"pit_{what}_base", ORANGE, "normal baseline")):
            p = df[col].to_numpy()
            share = [((p >= i / 10) & (p < (i + 1) / 10)).mean() * 100 for i in range(10)]
            ax.bar([v + dx for v in x], share, width=0.04, color=color, lw=0, label=label)
        ax.axhline(10, color=INK2, lw=1, ls=(0, (3, 3)), zorder=3)
        c50, c90 = dist.coverage(df[f"pit_{what}"], 0.5), dist.coverage(df[f"pit_{what}"], 0.9)
        b50, b90 = dist.coverage(df[f"pit_{what}_base"], 0.5), dist.coverage(df[f"pit_{what}_base"], 0.9)
        ax.set_title(f"{name}\nin the 50% / 90% ranges: {c50:.1%} / {c90:.1%}\n"
                     f"(normal baseline {b50:.1%} / {b90:.1%})", loc="left", fontsize=10, color=INK, pad=8)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "0.25", "0.5", "0.75", "1"])
        ax.set_xlim(-0.02, 1.02)
        ax.set_xlabel("where the outcome fell in the predicted distribution (PIT)")
    axes[0].set_ylim(0, 20)
    axes[0].set_yticks([0, 5, 10, 15, 20], ["0%", "5%", "10%", "15%", "20%"])
    axes[0].set_ylabel("share of states")
    axes[0].legend(loc="upper left", bbox_to_anchor=(0.0, -0.22), ncols=2, handlelength=1.2)
    fig.subplots_adjust(wspace=0.08, top=0.66)
    titles(fig, f"Predicted score distributions are calibrated on {season}",
           f"{df.height:,} states of {df['game_id'].n_unique()} games, models trained through {season - 1}. "
           "A calibrated forecast puts 10% of outcomes in each bin (dashed line).")
    return save(fig, "sim_pit")


def fig_sim_game(season: int = SIM_SEASON, game_id: str = SIM_GAME) -> Path | None:
    """Phase 2: three contracts priced from the same simulations, through one game."""
    from gridline.eval import sim_gate

    path = sim_gate.eval_path(season)
    if not path.exists():
        print(f"skipped sim_game: run `gridline sim gate --season {season}` first", file=sys.stderr)
        return None
    df = pl.read_parquet(path).filter(pl.col("game_id") == game_id).sort("seq")
    ev = load_events(season).filter(pl.col("game_id") == game_id).sort("seq")
    home, away = ev["home_team"][0], ev["away_team"][0]
    spread, total = df["spread_line"][0], df["total_line"][0]
    minutes = ((3600 - df["game_seconds_remaining"]) / 60).to_list()
    end = 60.0
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    scores = ev.filter((pl.col("home_score_post") != pl.col("home_score_pre"))
                       | (pl.col("away_score_post") != pl.col("away_score_pre")))
    for r in scores.iter_rows(named=True):
        ax.axvline((3600 - r["game_seconds_remaining"]) / 60, color=GRID, lw=1, zorder=0)
    series = [  # the home line is minus the spread (home favoured by 3 -> home -3)
        ("wp_sim_home", BLUE, f"{home} wins"),
        ("p_home_cover", ORANGE, f"{home} covers {-spread:+g}"),
        ("p_over", AQUA, f"total over {total:g}"),
    ]
    finals = {"wp_sim_home": 1.0 if df["final_margin"][0] > 0 else 0.0,
              "p_home_cover": float(df["final_margin"][0] > spread),
              "p_over": float(df["final_total"][0] > total)}
    for col, color, label in series:
        y = [v * 100 for v in df[col].to_list()]
        ax.step(minutes + [end], y + [finals[col] * 100], where="post", color=color, lw=1.5, label=label)
    last = df.filter(pl.col("kind") == "snap").tail(1).row(0, named=True)
    if game_id == "2025_01_BAL_BUF":  # the snap before the winning 32-yard field goal
        x_last = (3600 - last["game_seconds_remaining"]) / 60
        note(ax, f"4th & goal from the 14, 0:03 left, BUF down 2:\nsimulator {last['wp_sim_home']:.0%}, "
                 f"Phase 1 model {last['wp_direct_home']:.0%}", (x_last, last["wp_sim_home"] * 100),
             (55.5, 60), ha="right")
    x_lab = 26.0
    y_lab = df.filter(((3600 - pl.col("game_seconds_remaining")) / 60) <= x_lab)["p_over"][-1] * 100
    ax.text(x_lab, y_lab + 4, f"total over {total:g}", ha="center", va="bottom", fontsize=9.5, color=INK2)
    ax.set_xlim(0, end)
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(0, 61, 15), ["kickoff", "Q2", "half", "Q4", "end"])
    ax.set_xlabel("game clock (vertical lines: scoring plays)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=3, handlelength=1.6)
    fm, ft = int(df["final_margin"][0]), int(df["final_total"][0])
    hs, as_ = (ft + fm) // 2, (ft - fm) // 2
    titles(fig, "One set of simulations prices every contract, all through the game",
           f"{away} at {home}, {season} season: final {away} {as_}, {home} {hs}. Line: {home} {-spread:+g}, "
           f"total {total:g}. Fair prices from 2,000 simulated games at every snap.")
    return save(fig, "sim_game")


def _benchmark(season: int = 2025) -> tuple[dict, pl.DataFrame] | None:
    import json

    from gridline.eval import benchmark

    path = benchmark.out_dir() / f"benchmark_{season}.json"
    curve = benchmark.out_dir() / f"event_curve_{season}.parquet"
    if not path.exists() or not curve.exists():
        print(f"skipped benchmark charts: run `gridline market benchmark --season {season}` first", file=sys.stderr)
        return None
    return json.loads(path.read_text()), pl.read_parquet(curve)


def fig_lead_lag(season: int = 2025) -> Path | None:
    """Phase 3: how fast the market prices the plays that move the model."""
    b = _benchmark(season)
    if b is None:
        return None
    res, curve = b
    e = res["event_study"]
    c = curve.filter(pl.col("tau").is_between(-30, 120))
    x = c["tau"].to_list()
    fig, ax = plt.subplots(figsize=(9.6, 4.0))
    ax.fill_between(x, (c["p05"] * 100).to_list(), (c["p95"] * 100).to_list(), color=BLUE, alpha=0.15, lw=0,
                    step="post")
    ax.step(x, (c["share"] * 100).to_list(), where="post", color=BLUE, lw=1.5,
            label="Kalshi: share of gridline's move in the last trade price (average over plays; band: 90% interval)")
    ax.step([-30, 0, 120], [0, 100, 100], where="post", color=ORANGE, lw=1.5,
            label="gridline with no feed delay (δ = 0): the whole move when the play ends")
    for d in (5, 10, 20):
        y = e["by_delta"][str(d)]["share"] * 100
        ax.plot([d], [y], "o", ms=5, color=INK2, zorder=4)
        ax.text(d + 2.2, y - 1.5, f"{y:.0f}% after {d} s", fontsize=9, color=INK2, ha="left", va="top")
    plateau = e["share_at_120s"] * 100
    ax.text(118, plateau - 4, f"the market's move settles at about {plateau:.0f}% of gridline's", fontsize=9,
            color=INK2, ha="right", va="top")
    ax.set_xlim(-30, 120)
    ax.set_ylim(-5, 115)
    ax.set_yticks([0, 25, 50, 75, 100], ["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(-30, 121, 15), [f"{t:+d} s" if t else "0" for t in range(-30, 121, 15)])
    ax.set_xlabel("seconds from the end of the play (NFL timestamp)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.17), ncols=1, handlelength=1.6)
    titles(fig, f"Kalshi makes half of a big play's move within {e['half_move_s']:.0f} seconds of the NFL's timestamp",
           f"{e['moves']:,} plays in {e['games']} games of {season} that moved gridline's win price by 5 points or more "
           f"(median {e['median_abs_move'] * 100:.0f}).")
    return save(fig, "lead_lag")


def fig_benchmark(season: int = 2025) -> Path | None:
    """Phase 3: Brier score of gridline minus Kalshi's, by stage of the game, for two delays."""
    b = _benchmark(season)
    if b is None:
        return None
    res, _ = b
    stages = ["Q1", "Q2", "Q3", "Q4", "last 5 min"]
    labels = ["Q1", "Q2", "Q3", "Q4 to 5:00", "last 5 minutes"]
    fig, ax = plt.subplots(figsize=(9.6, 3.9))
    ax.axhline(0, color=INK2, lw=1, zorder=1)
    for dx, d, color in ((-0.12, "0", ORANGE), (0.12, "20", BLUE)):
        w = res["winner"][d]["by_stage"]
        xs = [i + dx for i, st in enumerate(stages) if st in w]
        mid = [w[st]["brier_corrected_diff"] * 1000 for st in stages if st in w]
        lo = [w[st]["brier_corrected_diff_p05"] * 1000 for st in stages if st in w]
        hi = [w[st]["brier_corrected_diff_p95"] * 1000 for st in stages if st in w]
        ax.vlines(xs, lo, hi, color=color, lw=2, alpha=0.6)
        ax.plot(xs, mid, "o", ms=6, color=color, label=f"gridline learns each play {d} s after it ends")
    ax.set_xticks(range(len(stages)), labels)
    ax.set_ylabel("Brier score difference (× 1000)")
    ax.set_xlabel("stage of the game (below zero: gridline more accurate than the market)")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, -0.2), ncols=2, handlelength=1.2)
    titles(fig, "As accurate as the market, until the last five minutes",
           f"Brier score of gridline's win price minus Kalshi's, {season}, every in-game minute, "
           "with 90% intervals from resampling games.")
    return save(fig, "benchmark")


def main(argv: list[str] | None = None) -> None:
    style()
    ev = game_events()
    fig_timestamp_cleaning(ev)
    fig_timing_coverage()
    fig_contract_prices()
    fig_wp_calibration()
    fig_sim_pit()
    fig_sim_game()
    fig_market_vs_model(ev)
    fig_lead_lag()
    fig_benchmark()
    k = kalshi_game()
    if k is None:
        print(f"skipped latency: no raw Kalshi pull for {GAME} here "
              "(run `gridline kalshi pull --season 2025`)", file=sys.stderr)
        return
    _, sec = k
    fig_latency(sec, ev)


if __name__ == "__main__":
    main()
