"""
report.py — uniform output saving for every analysis run.

Every figure is saved as a PNG *and* the data behind it as a .txt; every
numeric result table is saved as .txt; everything printed is also collected
into report.txt. One :class:`Reporter` per output directory.

    rep = Reporter("results/T0300")
    rep.print("density =", 2.73)
    rep.save_curve("gr_SiN", {"r": r, "g": g})          # → gr_SiN.txt
    rep.save_fig(fig, "gr_partials", data={"r": r, ...}) # → gr_partials.png + .txt
    rep.flush()                                          # → report.txt
"""
from __future__ import annotations
from pathlib import Path
from datetime import datetime
import numpy as np


class Reporter:
    def __init__(self, outdir, echo=True, title=None):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.echo = echo
        self._log = []
        if title:
            self.print("=" * 72)
            self.print(title)
            self.print(f"generated {datetime.now():%Y-%m-%d %H:%M:%S}   ->  {self.dir}")
            self.print("=" * 72)

    # ---- text log ----------------------------------------------------------
    def print(self, *args):
        s = " ".join(str(a) for a in args)
        self._log.append(s)
        if self.echo:
            print(s, flush=True)   # flush so progress shows in real time

    def flush(self, name="report.txt"):
        path = self.dir / name
        path.write_text("\n".join(self._log) + "\n")
        if self.echo:
            print(f"[saved] {path}")
        return path

    # ---- numeric data ------------------------------------------------------
    def save_curve(self, name, columns, headers=None, fmt="%.6g"):
        """Save named columns to <name>.txt. `columns` is a dict {col: 1d-array}
        (all same length) or a 2d array (with optional `headers`)."""
        path = self.dir / f"{name}.txt"
        if isinstance(columns, dict):
            headers = list(columns.keys())
            arr = np.column_stack([np.asarray(columns[h], float) for h in headers])
        else:
            arr = np.atleast_2d(np.asarray(columns, float))
            if arr.shape[0] < arr.shape[1] and headers and len(headers) == arr.shape[0]:
                arr = arr.T
        hdr = "  ".join(headers) if headers else ""
        np.savetxt(path, arr, header=hdr, fmt=fmt)
        self._note(path)
        return path

    def save_table(self, name, text):
        """Save a preformatted text table to <name>.txt."""
        path = self.dir / f"{name}.txt"
        path.write_text(text if text.endswith("\n") else text + "\n")
        self._note(path)
        return path

    # ---- figures -----------------------------------------------------------
    def save_fig(self, fig, name, data=None, headers=None, dpi=150):
        """Save a matplotlib figure to <name>.png, and (if `data` given) the
        underlying numbers to <name>.txt."""
        png = self.dir / f"{name}.png"
        fig.savefig(png, dpi=dpi, bbox_inches="tight")
        self._note(png)
        if data is not None:
            self.save_curve(name, data, headers)
        return png

    def _note(self, path):
        self._log.append(f"[saved] {path}")
        if self.echo:
            print(f"[saved] {path}")
