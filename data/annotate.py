"""Labels each sampled position with Stockfish's best move and a win probability.

Intent: this is what makes the data "supervised" -- every position gets a
known answer from the oracle. Labeling is also real compute, and the plan
charges it to the SFT and warm-start budgets (Sec 0, Compute accounting), so
the Stockfish CPU time is measured and saved alongside the labels.

Main pieces:
- label_chunk -- one worker: start Stockfish, label a list of positions, report CPU used
- annotate    -- label everything in parallel, saving each finished chunk to disk;
                 re-running skips chunks already saved (survives Colab disconnects)
- load_labels -- read the saved labels back as one table

Each label is from the point of view of the side to move: ``win_prob`` is that
side's expected score (the value target decided in the design doc), and
``best_move`` is Stockfish's choice at a fixed search depth, one thread, with
its memory cleared before every position -- so the same position always gets
the same label, no matter the order or the machine.
"""
from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import chess
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from engine.stockfish import SF_VERSION, WIN_PROB_MODEL, Stockfish, find_stockfish

LABEL_COLUMNS = ("best_move", "cp", "mate", "win_prob", "depth", "nodes")


def label_chunk(job: tuple[list[dict], dict]) -> tuple[list[dict], float]:
    """Label one chunk of positions in a fresh Stockfish process.

    Input:  (records, settings) -- records carry at least "fen"; settings has
            "depth", "nodes", and "path" (Stockfish binary).
    Output: (records with the LABEL_COLUMNS added, CPU seconds Stockfish used).
            The CPU figure includes loading the engine's network -- a real cost
            of labeling, so it's counted.
    """
    records, settings = job
    out = []
    with Stockfish(path=settings["path"], threads=1) as sf:
        for rec in records:
            a = sf.analyse(chess.Board(rec["fen"]), depth=settings["depth"], nodes=settings["nodes"], fresh=True)
            out.append({**rec, "best_move": a.best_move, "cp": a.cp, "mate": a.mate,
                        "win_prob": a.win_prob, "depth": a.depth, "nodes": a.nodes})
    return out, sf.cpu_seconds


def _parts(out_dir: Path) -> list[Path]:
    return sorted((out_dir / "parts").glob("part-*.parquet"))


def _write_part(path: Path, rows: list[dict], cpu_seconds: float) -> None:
    """Save one labeled chunk; its Stockfish CPU time rides along in the file's metadata."""
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}),
                                           b"stockfish_cpu_seconds": str(cpu_seconds).encode()})
    tmp = path.with_name(path.name + ".tmp")
    pq.write_table(table, tmp)
    tmp.replace(path)  # only complete chunks get a real name


def _part_cpu(path: Path) -> float:
    return float(pq.read_schema(path).metadata[b"stockfish_cpu_seconds"])


def annotate(records: list[dict], out_dir: str | Path, *, depth: int | None = 12, nodes: int | None = None,
             workers: int | None = None, chunk_size: int = 250, stockfish_path: str | None = None,
             verbose: bool = True) -> dict:
    """Label every record with Stockfish, in parallel, resumably. Returns the manifest.

    Inputs:
      records    -- from sample_positions (each has a unique "fen")
      out_dir    -- where labels go, e.g. <Drive>/data/processed/<dataset name>
      depth / nodes -- the fixed search limit per label (give one or both)
      workers    -- parallel Stockfish processes (default: one per CPU core)
      chunk_size -- positions per saved chunk; a dropped session loses at most one chunk per worker
    Output: manifest dict (also written to <out_dir>/manifest.json): row counts per
            phase, total Stockfish CPU seconds, wall time, and the label settings.
            The combined labels are written to <out_dir>/labels.parquet.

    Core logic: read the chunks already saved and drop those positions from the
    to-do list. Split the rest into chunks, label them in a pool of worker
    processes (each runs its own single-threaded Stockfish), and save each chunk
    the moment it finishes. When nothing is left, combine all chunks into one file.
    """
    if depth is None and nodes is None:
        raise ValueError("give depth= and/or nodes= -- labels need a fixed search limit")
    out_dir = Path(out_dir)
    (out_dir / "parts").mkdir(parents=True, exist_ok=True)
    settings = {"depth": depth, "nodes": nodes, "path": find_stockfish(stockfish_path)}

    done = {fen for part in _parts(out_dir) for fen in pq.read_table(part, columns=["fen"])["fen"].to_pylist()}
    todo = [r for r in records if r["fen"] not in done]
    chunks = [todo[i:i + chunk_size] for i in range(0, len(todo), chunk_size)]
    next_index = len(_parts(out_dir))
    if verbose:
        print(f"{len(done)} positions already labeled; {len(todo)} to go in {len(chunks)} chunks")

    start = time.perf_counter()
    if chunks:
        # "spawn" starts each worker clean, so it never inherits the parent's
        # engine-communication threads (forking a process with threads can hang).
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            futures = [pool.submit(label_chunk, (c, settings)) for c in chunks]
            for k, fut in enumerate(as_completed(futures)):  # save each chunk as soon as it's done
                rows, cpu = fut.result()
                _write_part(out_dir / "parts" / f"part-{next_index + k:05d}.parquet", rows, cpu)
                if verbose:
                    print(f"  chunk {k + 1}/{len(chunks)}: {len(rows)} positions, {cpu:.0f} CPU s")
    wall = time.perf_counter() - start

    labels = load_labels(out_dir, from_parts=True)
    labels.to_parquet(out_dir / "labels.parquet", index=False)
    manifest = {
        "rows": len(labels),
        "rows_per_phase": labels["phase"].value_counts().to_dict() if "phase" in labels else {},
        "stockfish": SF_VERSION, "label_depth": depth, "label_nodes": nodes, "win_prob_model": WIN_PROB_MODEL,
        "stockfish_cpu_seconds": sum(_part_cpu(p) for p in _parts(out_dir)),
        "labeled_this_run": len(todo), "wall_seconds_this_run": wall,
    }
    with open(out_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def load_labels(out_dir: str | Path, *, from_parts: bool = False) -> pd.DataFrame:
    """The labeled positions as one table (from labels.parquet, or straight from the chunks)."""
    out_dir = Path(out_dir)
    if not from_parts and (out_dir / "labels.parquet").exists():
        return pd.read_parquet(out_dir / "labels.parquet")
    parts = _parts(out_dir)
    if not parts:
        raise FileNotFoundError(f"no labeled chunks in {out_dir / 'parts'}")
    return pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
