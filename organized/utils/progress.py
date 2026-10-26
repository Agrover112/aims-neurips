"""tqdm-style progress bars that survive ``modal app logs``.

tqdm redraws with carriage returns, which is fine on a TTY but turns a pulled
Modal log into one unreadable line. These emit a **newline-terminated** line per
update instead, each one self-describing so it still makes sense when grepped
out of a interleaved multi-container log.

    from progress import track

    for bench in track(benchmarks, desc=f"seed={seed} benchmarks"):
        ...

    # nested / manual
    bar = Bar(total=len(kinds), desc=f"seed={seed} {bench} models")
    for kind in kinds:
        ...
        bar.step(f"kind={kind}")

Shipped into Modal containers by the job scripts via
``image.add_local_file(... , "/root/progress.py")``; import it inside the
remote function, after ``sys.path.insert(0, "/root")``.
"""

from __future__ import annotations

import sys
import time

WIDTH = 24


def _fmt(seconds: float) -> str:
    if seconds != seconds or seconds in (float("inf"), float("-inf")):
        return "--:--"
    seconds = int(seconds)
    if seconds >= 3600:
        return f"{seconds // 3600}h{(seconds % 3600) // 60:02d}m"
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


class Bar:
    """A progress bar that prints one standalone line per update.

    ``min_interval`` throttles output so a fast inner loop cannot flood the log;
    the first and last updates always print regardless.
    """

    def __init__(self, total=None, desc="", out=None, min_interval=0.0, every=1):
        self.total = total
        self.desc = desc
        self.out = out or sys.stdout
        self.min_interval = min_interval
        self.every = max(1, every)
        self.n = 0
        self._start = time.monotonic()
        self._last_emit = 0.0
        self.emit(note="start")

    def emit(self, note=""):
        elapsed = time.monotonic() - self._start
        if self.total:
            frac = min(self.n / self.total, 1.0)
            filled = int(round(WIDTH * frac))
            bar = "#" * filled + "." * (WIDTH - filled)
            eta = (elapsed / self.n) * (self.total - self.n) if self.n else float("nan")
            head = (f"{self.desc} [{bar}] {frac * 100:5.1f}% "
                    f"{self.n}/{self.total} elapsed={_fmt(elapsed)} eta={_fmt(eta)}")
        else:
            head = f"{self.desc} {self.n} done elapsed={_fmt(elapsed)}"
        print(f"{head}  {note}".rstrip(), file=self.out, flush=True)
        self._last_emit = time.monotonic()

    def step(self, note="", n=1):
        self.n += n
        done = self.total is not None and self.n >= self.total
        due = (self.n % self.every == 0
               and time.monotonic() - self._last_emit >= self.min_interval)
        if done or due:
            self.emit(note)

    def close(self, note="done"):
        if self.total is None or self.n < self.total:
            self.emit(note)


def track(iterable, desc="", total=None, out=None, min_interval=0.0, every=1):
    """Wrap an iterable, emitting a progress line after each item.

    The line is emitted *after* the body runs, so the reported count is work
    actually completed, not work dispatched.
    """
    if total is None:
        try:
            total = len(iterable)
        except TypeError:
            total = None
    bar = Bar(total=total, desc=desc, out=out, min_interval=min_interval, every=every)
    for item in iterable:
        yield item
        bar.step()
    bar.close()
