"""One eval panel for every headline model (plan v5 Step 12f).

Intent: ratings only compare when models faced the same opponents. The reference
members' own ratings are uncertain (+/- ~100); that error is shared, so it cancels
between two models rated against the same members, and doesn't between models rated
against different ones. So every headline model is rated against one fixed **panel**
(``configs/eval.yaml``): ``random``, ``greedy``, ``step_20000``, ``sf_1320``, plus two
frozen model **rungs** in the empty stretch between 719 and 1320, AlphaZero r109 and
SFT 500k @62.5k.

Main pieces:
- load_panel      -- the panel's member names, in order (configs/eval.yaml)
- coverage        -- per model: saved eval games against each panel member
- panel_ratings   -- ratings that compare: panel members + models fully rated on the panel
- calibrate_rungs -- stage 1: the new rungs play the existing members and each other, are
                     fitted together with the existing members held fixed, then frozen
- extend          -- stage 2: a model plays only the panel members its saved games lack,
                     then is refitted on old + new games together

Why old games stay valid: a model's rating uses only its own games against frozen
opponents; a frozen opponent's rating never changes, so a game against it means the same
next week (same checkpoint, same engine, same time limit).
"""
from __future__ import annotations

from pathlib import Path

import yaml

from eval.elo import RatingEstimate, fit_with_ci
from eval.match import play_match, records_from_rows, summarize, to_results

PANEL_NOTE = "rated on the panel"
RUNG_NOTE = "panel rung (Step 12f)"


def load_panel(config_path: str | Path = "configs/eval.yaml") -> list[str]:
    """The panel members' names, in order (configs/eval.yaml ``panel``)."""
    with open(config_path) as f:
        return list(yaml.safe_load(f)["panel"])


def _opponent(record, name: str) -> str:
    return record.black if record.white == name else record.white


def coverage(results, panel: list[str]) -> dict[str, dict[str, int]]:
    """{model name: {panel member: saved eval games against it}} for every model row."""
    out = {n: {p: 0 for p in panel} for n, r in results.rows.items() if r["role"] == "model"}
    for g in results.games():                       # kind "eval": the games behind each model row
        counts = out.get(g["subject"])
        if counts is None:
            continue
        opp = g["black"] if g["white"] == g["subject"] else g["white"]
        if opp in counts:
            counts[opp] += 1
    return out


def panel_ratings(results, panel: list[str], games_per_opponent: int = 20) -> dict[str, float]:
    """Ratings that compare with each other: the panel members, plus every model row with at
    least ``games_per_opponent`` saved games against every panel member. Rows rated any other
    way are left out (use this for charts and the consistency check)."""
    cov = coverage(results, panel)
    out = {}
    for name, row in results.rows.items():
        if row["rating"] in ("", None):
            continue
        if name in panel and row["role"] in ("pin", "reference"):
            out[name] = float(row["rating"])
        elif row["role"] == "model" and all(n >= games_per_opponent for n in cov.get(name, {}).values()):
            out[name] = float(row["rating"])
    return out


def reference_spec(spec: dict) -> dict:
    """A load_player spec -> the ``spec`` cell of its reference row (how to rebuild it as an opponent)."""
    if spec["kind"] == "sft":
        return {"kind": "sft_checkpoint", "run": spec["run"], "step": int(spec["step"])}
    if spec["kind"] == "alphazero":
        return {"kind": "alphazero_checkpoint", "run": spec["run"], "step": int(spec["step"]),
                "simulations": int(spec.get("simulations", 50))}
    raise ValueError(f"no rung kind for {spec['kind']!r} (sft and alphazero only)")


def calibrate_rungs(rungs: list[dict], results, artifact_root, existing: dict, *, games_per_pair: int = 40,
                    batch_games: int = 10, n_boot: int = 1000, load=None,
                    verbose: bool = True) -> dict[str, RatingEstimate]:
    """Stage 1: measure the new rungs, then freeze them. Returns their estimates ({} if already frozen).

    rungs: load_player specs ({"name", "kind": "sft" | "alphazero", "run", "step",
    "simulations"}), each with optional "fields": row columns for its frozen row (run, step,
    train_gpu_hours, label_cpu_hours, dataset_size). existing: {name: agent} for members
    already frozen (e.g. random, greedy, step_20000, sf_1320).

    Each rung plays every existing member and every other rung, ``games_per_pair`` games
    per pairing (color-swapped pairs, every game saved to games.csv as kind "calib",
    subject "calib|a|b", after every batch: resumable, a rerun plays only what is missing).
    Then all rungs are fitted together with the existing members held fixed, and each is
    frozen as a reference row (a model row of the same name is replaced; its note keeps
    the old rating, and its checkpoint facts carry over). Refuses to freeze a rating
    flagged off the scale.
    """
    from eval.agents import CARRIED_FIELDS, load_player
    names = [s["name"] for s in rungs]
    frozen = [n for n in names if results.role(n) in ("pin", "reference")]
    if len(frozen) == len(names):
        if verbose:
            print("rungs already frozen:", names)
        return {}
    if frozen:
        raise ValueError(f"{frozen} are frozen but {sorted(set(names) - set(frozen))} aren't; calibrate rungs together")
    if batch_games % 2 or games_per_pair % 2:
        raise ValueError("game counts must be even (games come in color-swapped pairs)")
    fixed = results.fixed()
    missing = [n for n in existing if n not in fixed]
    if missing:
        raise ValueError(f"{missing} have no frozen rating")

    pairs = [(r, o) for r in names for o in existing] + [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    agents = dict(existing)
    for s in rungs:
        agents[s["name"]] = (load or load_player)(s, artifact_root)
    records = []
    for k, (a, b) in enumerate(pairs):
        subject = f"calib|{a}|{b}"
        saved = records_from_rows(results.games(subject, kind="calib"))
        while len(saved) < games_per_pair:
            n = min(batch_games, games_per_pair - len(saved))
            games = play_match(agents[a], agents[b], n, seed=k + 1000 * (len(saved) // batch_games))
            saved = saved + games
            results.log_games(saved, "calib", subject)
            results.save()
            if verbose:
                s_ = summarize(saved, a)
                print(f"  {a} vs {b}: {len(saved)}/{games_per_pair} games, {s_['wins']}W {s_['draws']}D {s_['losses']}L for {a}")
        records += saved

    est = fit_with_ci(to_results(records), {n: fixed[n] for n in existing}, n_boot=n_boot)
    off = {n: est[n].at_bound for n in names if est[n].at_bound}
    if off:
        raise RuntimeError(f"not freezing ratings flagged off the scale: {off} -- add games or a closer member")
    for s in rungs:
        e, name = est[s["name"]], s["name"]
        old = results.rows.get(name)
        fields = dict(s.get("fields") or {})
        note = f"{RUNG_NOTE}: {e.games} calibration games vs {', '.join(existing)} and the other rung(s)"
        if old and old["role"] == "model":
            for k_ in (*CARRIED_FIELDS, "sec_per_move", "gpu"):
                if old.get(k_) not in ("", None):
                    fields.setdefault(k_, old[k_])
            note += f"; was model {old['rating']} [{old['ci_low']}, {old['ci_high']}], {old['games']} games"
        results.add_reference(name, e.rating, e.ci_low, e.ci_high, e.games, spec=reference_spec(s), note=note, **fields)
        if verbose:
            print(f"frozen {name}: {e.rating:.0f} [{e.ci_low:.0f}, {e.ci_high:.0f}], {e.games} games")
    results.save()
    return est


def extend(spec: dict, results, artifact_root, opponents: dict, *, games_per_opponent: int = 20,
           batch_games: int = 10, fields: dict | None = None, gpu: str = "", n_boot: int = 1000, load=None,
           verbose: bool = True) -> RatingEstimate | None:
    """Stage 2: rate a model on the full panel, playing only what its saved games lack.

    spec: a load_player spec; ``name`` is its results row (existing or new).
    opponents: {name: agent} for the whole panel, in panel order (each must be frozen).
    For each member with fewer than ``games_per_opponent`` saved games, plays the rest in
    batches (seed = the member's panel position + 1000 x batch number), and after every
    batch refits on all saved + new games and overwrites the row (Elo, CI, W/D/L,
    sec/move): a stop keeps every finished batch, and a rerun continues. A row with
    nothing missing is skipped (returns None). An existing row's checkpoint facts carry
    over and its note keeps the old rating once; ``fields`` adds or overrides columns
    (e.g. for a new row).
    """
    from eval.agents import CARRIED_FIELDS, load_player
    name = spec["name"]
    if results.role(name) in ("pin", "reference"):
        raise ValueError(f"{name!r} is a {results.role(name)} row; panel members aren't rated against the panel")
    if batch_games % 2 or games_per_opponent % 2:
        raise ValueError("game counts must be even (games come in color-swapped pairs)")
    fixed = results.fixed()
    missing = [o for o in opponents if o not in fixed]
    if missing:
        raise ValueError(f"{missing} have no frozen rating -- calibrate the rungs first")
    panel = list(opponents)
    saved = [r for r in records_from_rows(results.games(name)) if _opponent(r, name) in fixed]
    played = {o: sum(_opponent(r, name) == o for r in saved) for o in panel}
    if all(n >= games_per_opponent for n in played.values()):
        if verbose:
            print(f"{name}: already rated on the full panel -- skipped")
        return None

    row_fields: dict = {}
    old = results.rows.get(name)
    if old:
        row_fields = {k: old[k] for k in CARRIED_FIELDS if old.get(k) not in ("", None)}
        note = old.get("note", "") or ""
        if PANEL_NOTE not in note:
            note = (f"{note}; " if note else "") + (f"{PANEL_NOTE} (was {old['rating']} [{old['ci_low']}, "
                                                    f"{old['ci_high']}], {old['games']} games)")
        row_fields["note"] = note
    row_fields.update(fields or {})
    if verbose:
        todo = {o: games_per_opponent - n for o, n in played.items() if n < games_per_opponent}
        print(f"{name}: kept {len(saved)} saved games; to play {todo}")

    agent = (load or load_player)(spec, artifact_root)
    new, est = [], None
    try:
        while any(played[o] < games_per_opponent for o in panel):
            for i, o in enumerate(panel):
                if played[o] >= games_per_opponent:
                    continue
                n = min(batch_games, games_per_opponent - played[o])
                games = play_match(agent, opponents[o], n, seed=i + 1000 * (played[o] // batch_games))
                new += games
                played[o] += n
                records = saved + new
                vs = {_opponent(r, name) for r in records}
                est = fit_with_ci(to_results(records), {v: fixed[v] for v in vs}, n_boot=n_boot)[name]
                results.log_model(est, records, gpu=gpu, **row_fields)
                results.save()
                if verbose:
                    s_ = summarize(games, name)
                    print(f"  vs {o}: +{n} -> {s_['wins']}W {s_['draws']}D {s_['losses']}L | "
                          f"now {est.rating:.0f} [{est.ci_low:.0f}, {est.ci_high:.0f}], {est.games} games")
    finally:
        del agent
    return est
