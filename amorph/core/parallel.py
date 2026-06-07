"""
parallel.py — process-pool helpers for multi-core analysis.

Two granularities are used in the package:
  * temperature-level (sweep): each worker loads its own dump and computes all
    observables — minimal data transfer, ideal for a folder of per-T dumps.
  * frame-level (single structure): the per-frame kernel is mapped over frames.

Workers run in separate processes (true parallelism, no GIL). Callables passed
to :func:`pmap` must be importable/picklable — use module-level functions or
``functools.partial`` of them.
"""
from __future__ import annotations
import os
from concurrent.futures import ProcessPoolExecutor


def resolve_jobs(jobs):
    """Normalise a jobs request to a positive int. jobs<=0 -> all CPUs."""
    if jobs is None:
        return 1
    if jobs <= 0:
        return os.cpu_count() or 1
    return max(1, int(jobs))


def pmap(func, items, jobs=1, chunksize=1, ordered=True, on_done=None):
    """Map ``func`` over ``items`` across ``jobs`` processes.

    Falls back to a serial list comprehension when jobs==1 or there is nothing
    to gain. Results are returned in input order. ``on_done(n_completed)`` is
    called after each item finishes (for progress reporting).
    """
    items = list(items)
    jobs = resolve_jobs(jobs)
    if jobs == 1 or len(items) <= 1:
        out = []
        for x in items:
            out.append(func(x))
            if on_done:
                on_done(len(out))
        return out

    jobs = min(jobs, len(items))
    results = [None] * len(items)
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        futures = {ex.submit(func, x): i for i, x in enumerate(items)}
        done = 0
        from concurrent.futures import as_completed
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            if on_done:
                on_done(done)
    return results
