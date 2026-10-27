"""Paced background work: a worker thread rests at its pause points, so the main thread keeps the GIL."""

import threading
import time

import pytest

from bifold_demo import pacing


def _busy(seconds: float) -> None:
    """Work for ``seconds`` (pure Python, holding the GIL; rests not counted), with a pause point every half
    millisecond."""
    worked = 0.0
    while worked < seconds:
        start = stop = time.perf_counter()
        while stop - start < 0.0005:
            stop = time.perf_counter()
        worked += stop - start
        pacing.pause()


def test_pause_does_nothing_unless_paced():
    assert getattr(pacing._local, "pacer", None) is None
    start = time.perf_counter()
    for _ in range(10000):
        pacing.pause()
    assert time.perf_counter() - start < 0.05


def test_paced_work_rests_for_the_rest_of_the_time():
    with pacing.paced(share=0.25, burst=0.002) as pacer:
        start = time.perf_counter()
        _busy(0.03)
        took = time.perf_counter() - start
    assert pacer.rested == pytest.approx(3 * 0.03, rel=0.35)  # three times as long resting as working
    assert took >= 0.03 + 0.8 * 3 * 0.03
    assert pacer.longest < 0.02
    assert getattr(pacing._local, "pacer", None) is None  # the thread is not paced after the block


def test_pacing_is_per_thread_and_nests():
    seen = {}
    with pacing.paced(share=0.5) as outer:
        thread = threading.Thread(target=lambda: seen.update(other=getattr(pacing._local, "pacer", None)))
        thread.start()
        thread.join()
        with pacing.paced(share=0.1) as inner:
            assert pacing._local.pacer is inner
        assert pacing._local.pacer is outer
    assert seen == {"other": None}
    assert pacing.run_paced(lambda x, y=0: (x + y, pacing._local.pacer.share), 1, y=2, share=0.3) == (3, 0.3)
    with pytest.raises(ValueError):
        pacing.Pacer(share=0.0)


def test_a_paced_worker_leaves_the_main_thread_its_time():
    """The main thread does short steps of Python work; a busy worker thread slows them down much less when paced."""

    def main_loop(seconds: float) -> float:
        steps, end = 0, time.perf_counter() + seconds
        while time.perf_counter() < end:
            sum(i * i for i in range(300))
            time.sleep(0.0005)  # lets go of the GIL, as a frame does many times
            steps += 1
        return steps / seconds

    def run(paced: bool) -> float:
        done = threading.Event()

        def work():
            if paced:
                with pacing.paced(share=0.25):
                    while not done.is_set():
                        _busy(0.01)
            else:
                while not done.is_set():
                    _busy(0.01)

        worker = threading.Thread(target=work)
        worker.start()
        try:
            return main_loop(0.4)
        finally:
            done.set()
            worker.join()

    alone, unpaced, paced = main_loop(0.4), run(False), run(True)
    # measured: about 1500 steps a second alone, 170 beside the busy worker (each sleep waits for the GIL's 5 ms
    # switch), 1250 beside the paced one
    assert paced > 3 * unpaced and paced > 0.5 * alone, (alone, unpaced, paced)


def test_hurried_work_rests_no_more():
    control = pacing.Control()
    with pacing.paced(share=0.25, burst=0.002, control=control) as pacer:
        _busy(0.01)
        rested = pacer.rested
        assert rested > 0.015
        control.hurry()  # someone waits for it now
        start = time.perf_counter()
        _busy(0.02)
        assert pacer.rested == rested and time.perf_counter() - start < 0.05


def test_cancelled_work_stops_at_its_next_pause_even_while_it_rests():
    control, stopped = pacing.Control(), {}

    def work():
        try:
            with pacing.paced(share=0.01, burst=0.002, control=control):  # rests 99 % of the time
                _busy(1.0)
        except pacing.Cancelled:
            stopped["at"] = time.perf_counter()

    thread = threading.Thread(target=work)
    thread.start()
    time.sleep(0.05)
    start = time.perf_counter()
    control.cancel()
    thread.join(timeout=1.0)
    assert not thread.is_alive() and stopped["at"] - start < 0.05


def test_the_longest_stretch_counts_the_work_after_the_last_pause():
    with pacing.paced() as pacer:
        pacing.pause()
        start = time.perf_counter()
        while time.perf_counter() - start < 0.02:  # no pause point
            pass
    assert pacer.longest >= 0.02
