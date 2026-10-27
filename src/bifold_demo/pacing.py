"""Background work that leaves the main thread its frames: a worker thread rests now and then, on purpose.

Python runs one thread at a time (the GIL). A worker thread busy with Python code or many small numpy operations
(making a behaviour again for another avatar, fitting its timing in, planning its camera play: half a second or so)
takes the GIL for up to 5 ms every time the main thread lets go of it, which the main thread does many times a frame
(each larger numpy operation, the GL calls, the sleep until the next frame). Measured on the fair, the main thread's
frames then take two to three times as long, and at 30 frames a second every other one comes late.

So such work runs :func:`paced`: the long computations call :func:`pause` between their steps (a few milliseconds
apart), and in a paced thread a pause rests after each ``burst`` seconds of work, long enough that the work takes only
``share`` of the time. While it rests, the main thread has the GIL to itself. The work takes longer (four times as
long with a share of a quarter, alone), the frames hardly notice. Work is counted in wall-clock time (``CLOCK``), the
time the worker waits for the GIL included: so the busier the main thread, the more the worker holds back (beside the
demo's 30 frames a second its jobs take four to six times their CPU time; counting CPU time instead would keep them
near four times, but wake the worker, and take the GIL from the frames, more often; how many frames came late did not
differ measurably either way). In any thread that is not paced (the main thread,
tests), :func:`pause` does nothing. A :class:`Control` lets another thread tell paced work to hurry (rest no more:
someone waits for it) or to stop at its next pause (:class:`Cancelled`).
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager

SHARE = 0.25  # the share of the time paced work takes
BURST = 0.003  # seconds of work between rests
CLOCK = time.perf_counter  # what work is counted in (see the module docs)

_local = threading.local()


class Cancelled(Exception):
    """Raised at a :func:`pause` of paced work that was told to stop (:meth:`Control.cancel`)."""


class Control:
    """What another thread can tell paced work (see the module docs): :meth:`hurry` or :meth:`cancel`."""

    def __init__(self):
        self.hurried = self.cancelled = False
        self._wake = threading.Event()

    def hurry(self) -> None:
        """Rest no more (someone waits for the result), from now on."""
        self.hurried = True
        self._wake.set()

    def cancel(self) -> None:
        """Stop at the next pause: it raises :class:`Cancelled` there."""
        self.cancelled = True
        self._wake.set()

    def rest(self, seconds: float) -> None:
        self._wake.wait(seconds)


class Pacer:
    """The pacing of one thread's work (see the module docs). ``rested``: seconds it slept so far; ``longest``: the
    longest stretch of work between two pauses (or after the last one; a pause point is missing where that is much more
    than ``burst``); ``control``: a :class:`Control`, or ``None``."""

    def __init__(self, share: float = SHARE, burst: float = BURST, control: Control | None = None):
        if not 0.0 < share <= 1.0:
            raise ValueError(f"share must be in (0, 1], not {share}")
        self.share, self.burst, self.control = float(share), float(burst), control
        self.rested = self.longest = 0.0
        self._since = CLOCK()

    def pause(self) -> None:
        control = self.control
        if control is not None and control.cancelled:
            raise Cancelled()
        ran = self._stretch()
        if ran < self.burst:
            return
        if self.share < 1.0 and not (control is not None and control.hurried):
            rest = ran * (1.0 - self.share) / self.share
            start = time.perf_counter()
            if control is not None:
                control.rest(rest)
                if control.cancelled:
                    raise Cancelled()
            else:
                time.sleep(rest)
            self.rested += time.perf_counter() - start
        self._since = CLOCK()

    def _stretch(self) -> float:
        """The work since the last rest (and ``longest`` updated)."""
        ran = CLOCK() - self._since
        self.longest = max(self.longest, ran)
        return ran


def pause() -> None:
    """A point between two steps of a long computation where a paced thread may rest (else nothing happens)."""
    pacer = getattr(_local, "pacer", None)
    if pacer is not None:
        pacer.pause()


@contextmanager
def paced(share: float = SHARE, burst: float = BURST, control: Control | None = None):
    """Pace the current thread's work inside the ``with`` block (see the module docs); yields its :class:`Pacer`."""
    previous = getattr(_local, "pacer", None)
    pacer = _local.pacer = Pacer(share, burst, control)
    try:
        yield pacer
    finally:
        pacer._stretch()  # the work after the last pause counts too
        _local.pacer = previous


def run_paced(work, *args, share: float = SHARE, burst: float = BURST, control: Control | None = None, **options):
    """``work(*args, **options)``, paced: for a thread pool's ``submit``."""
    with paced(share, burst, control):
        return work(*args, **options)
