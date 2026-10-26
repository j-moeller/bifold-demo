"""Background work that leaves the main thread its frames: a worker thread rests now and then, on purpose.

Python runs one thread at a time (the GIL). A worker thread busy with Python code or many small numpy operations
(making a behaviour again for another avatar, fitting its timing in, planning its camera play: half a second or so)
takes the GIL for up to 5 ms every time the main thread lets go of it, which the main thread does many times a frame
(each larger numpy operation, the GL calls, the sleep until the next frame). Measured on the fair, the main thread's
frames then take two to three times as long, and at 30 frames a second every other one comes late.

So such work runs :func:`paced`: the long computations call :func:`pause` between their steps (a few milliseconds
apart), and in a paced thread a pause rests after each ``burst`` seconds of work, long enough that the work takes only
``share`` of the time. While it rests, the main thread has the GIL to itself. The work takes longer (four times as
long with a share of a quarter), the frames hardly notice. In any thread that is not paced (the main thread, tests),
:func:`pause` does nothing.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

SHARE = 0.25  # the share of the time paced work takes
BURST = 0.003  # seconds of work between rests

_local = threading.local()


class Pacer:
    """The pacing of one thread's work (see the module docs). ``rested``: seconds it slept so far; ``longest``: the
    longest stretch of work between two pauses (a pause point is missing where that is much more than ``burst``)."""

    def __init__(self, share: float = SHARE, burst: float = BURST):
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share must be in (0, 1], not {share}")
        self.share, self.burst = float(share), float(burst)
        self.rested = self.longest = 0.0
        self._since = time.perf_counter()

    def pause(self) -> None:
        now = time.perf_counter()
        ran = now - self._since
        self.longest = max(self.longest, ran)
        if ran < self.burst:
            return
        if self.share < 1.0:
            rest = ran * (1.0 - self.share) / self.share
            time.sleep(rest)
            self.rested += rest
        self._since = time.perf_counter()


def pause() -> None:
    """A point between two steps of a long computation where a paced thread may rest (else nothing happens)."""
    pacer = getattr(_local, "pacer", None)
    if pacer is not None:
        pacer.pause()


@contextmanager
def paced(share: float = SHARE, burst: float = BURST):
    """Pace the current thread's work inside the ``with`` block (see the module docs); yields its :class:`Pacer`."""
    previous = getattr(_local, "pacer", None)
    pacer = _local.pacer = Pacer(share, burst)
    try:
        yield pacer
    finally:
        _local.pacer = previous


def run_paced(work, *args, share: float = SHARE, burst: float = BURST, **options):
    """``work(*args, **options)``, paced: for a thread pool's ``submit``."""
    with paced(share, burst):
        return work(*args, **options)
