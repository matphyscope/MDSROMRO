"""
average.py — Trajectory time-averaging with block-averaged error bars.

The whole package favours *accuracy over speed*: scalar/curve observables are
averaged over many production frames, and the statistical uncertainty is the
standard deviation between contiguous blocks of frames (block averaging, which
accounts for temporal correlation far better than a naive per-frame std).
"""
from __future__ import annotations
import numpy as np


def select_frames(traj, frame_range=None, stride=1):
    """Pick a production window and decorrelate by stride.

    frame_range : (lo, hi) | None   slice of frame indices (hi exclusive)
    stride : int
    """
    lo, hi = (0, len(traj)) if frame_range is None else frame_range
    return traj[lo:hi:stride]


def block_average(per_frame_values, n_blocks=5):
    """Mean and block-std of a per-frame quantity.

    Parameters
    ----------
    per_frame_values : sequence of array-like (or scalars)
        One entry per frame; all entries must share a shape.
    n_blocks : int
        Number of contiguous blocks. Effective value is capped at the number of
        frames.

    Returns
    -------
    mean : ndarray   per-frame mean (nan-aware)
    err  : ndarray   1σ standard deviation between block means (ddof=1);
                     zeros if only one block.
    """
    arr = np.asarray(per_frame_values, dtype=float)
    nframes = arr.shape[0]
    if nframes == 0:
        raise ValueError("No frames to average — check frame_range/stride.")
    nb = max(1, min(int(n_blocks), nframes))
    block_ids = np.array_split(np.arange(nframes), nb)
    block_means = np.array([np.nanmean(arr[b], axis=0) for b in block_ids])
    mean = np.nanmean(block_means, axis=0)
    err = (np.nanstd(block_means, axis=0, ddof=1)
           if nb > 1 else np.zeros_like(mean))
    return mean, err


def time_average(traj, per_frame_fn, n_blocks=5):
    """Apply ``per_frame_fn(frame) -> array`` to every frame, then block-average.

    Returns (mean, err) with the shape of the per-frame output.
    """
    vals = [np.asarray(per_frame_fn(fr), dtype=float) for fr in traj]
    return block_average(vals, n_blocks=n_blocks)
