"""Yoga clips on demand, for the interactive demo.

Solving a pose for an avatar takes up to about three seconds (the floor poses), which would stall the video if it
ran in the demo's process, even on another thread. :class:`YogaClips` therefore builds clips in a worker process (one
at a time, most urgent first), keeps them in memory and caches them on disk in ``<characters folder>/yoga/``, so each
avatar's pose is solved once (``bifold-characters yoga`` solves them all ahead of time). The cache is keyed by the
avatar's skeleton and mesh and by the source of :mod:`.yoga`: changing a pose rebuilds its clips.

:func:`worker_main` is the worker: it reads pickled jobs on stdin and writes pickled results to stdout.
"""

from __future__ import annotations

import hashlib
import itertools
import os
import pickle
import re
import signal
import subprocess
import sys
import threading
from dataclasses import replace
from pathlib import Path

from . import yoga
from .library import default_folder
from .model import AnimationClip, CharacterModel

URGENT, LATER = 0, 1  # priorities: a pose someone asked for, poses solved ahead of time


_SOURCE = Path(yoga.__file__).read_bytes()


def model_key(model: CharacterModel) -> str:
    """A fingerprint of what a yoga clip depends on: the skeleton, the skinned mesh and the pose code."""
    h = hashlib.blake2b(_SOURCE, digest_size=10)
    sk = model.skeleton
    h.update("\n".join(sk.names).encode())
    for array in (sk.parents, sk.rest_local, *(a for p in model.primitives for a in (p.positions, p.joints, p.weights))):
        h.update(array.tobytes())
    return h.hexdigest()


def cache_path(folder: Path, model: CharacterModel, pose: str) -> Path:
    return Path(folder) / f"{re.sub(r'[^A-Za-z0-9_.-]', '_', model.name)}-{pose}.npz"


def load_or_build(model: CharacterModel, pose: str, path: Path | None, key: str) -> AnimationClip:
    """The clip of ``pose`` for ``model``: from ``path`` if it holds one for ``key``, else solved (and saved there)."""
    from .rocketbox import read_clip, save_clip

    if path is not None and Path(path).is_file():
        try:
            clip = read_clip(path)
            if clip.metadata.get("cache_key") == key:
                return clip
        except Exception:  # noqa: BLE001 - a damaged or old file: solve again
            pass
    clip = yoga.yoga_clip(model, pose)
    clip.metadata["cache_key"] = key
    if path is not None:
        try:
            save_clip(clip, None, path=path)
        except OSError:
            pass  # a read-only folder: keep it in memory only
    return clip


def _light(model: CharacterModel) -> CharacterModel:
    """``model`` without its textures (all the worker needs, and cheap to send)."""
    return replace(model, primitives=[replace(p, texture=None) for p in model.primitives])


class YogaClips:
    """Clips of the poses in :data:`yoga.POSES`, built in the background.

    Args:
        cache_dir: the characters folder whose ``yoga/`` subfolder caches the clips (default: :func:`default_folder`);
            ``False`` caches in memory only.
        process: build in a worker process; ``False`` builds on a thread of this process (tests, or where Python
            cannot start itself).
    """

    def __init__(self, cache_dir=None, process: bool = True):
        if cache_dir is False:
            self.folder = None
        else:
            self.folder = (Path(cache_dir).expanduser() if cache_dir is not None else default_folder()) / "yoga"
        self.process = process
        self._lock = threading.Condition()
        self._ready: dict[tuple, AnimationClip] = {}
        self._errors: dict[tuple, str] = {}
        self._queue: dict[tuple, tuple] = {}  # (key, pose) -> (priority, order, model)
        self._running: tuple | None = None
        self._order = itertools.count()
        self._keys: dict[int, tuple[CharacterModel, str]] = {}
        self._worker = None  # the current thread's worker process (for close)
        self._thread = None
        self._generation = self._thread_generation = 0  # close() starts a new generation

    def key(self, model: CharacterModel) -> str:
        cached = self._keys.get(id(model))
        if cached is None or cached[0] is not model:
            cached = self._keys[id(model)] = (model, model_key(model))
        return cached[1]

    def get(self, model: CharacterModel, pose: str) -> AnimationClip | None:
        """The clip if it is ready (without asking for it)."""
        with self._lock:
            return self._ready.get((self.key(model), pose))

    def error(self, model: CharacterModel, pose: str) -> str | None:
        """Why the clip could not be built, if it failed."""
        with self._lock:
            return self._errors.get((self.key(model), pose))

    def pending(self, model: CharacterModel, pose: str) -> bool:
        job = (self.key(model), pose)
        with self._lock:
            return job == self._running or job in self._queue

    def request(self, model: CharacterModel, pose: str, priority: int = URGENT) -> AnimationClip | None:
        """The clip if it is ready; else start building it (``priority`` 0 first) and return ``None``."""
        if pose not in yoga.POSES:
            raise KeyError(f"unknown yoga pose {pose!r}; known: {', '.join(yoga.POSES)}")
        job = (self.key(model), pose)
        with self._lock:
            if job in self._ready:
                return self._ready[job]
            if job in self._errors and priority > URGENT:
                return None  # failed before: only try again when someone asks for it
            self._errors.pop(job, None)
            if job != self._running:
                known = self._queue.get(job)
                if known is None or priority < known[0]:
                    order = next(self._order) if priority == URGENT or known is None else known[1]
                    self._queue[job] = (priority, -order if priority == URGENT else order, model)
            self._start()
            self._lock.notify_all()
        return None

    def prepare(self, models, poses=None) -> None:
        """Build ``poses`` (default: all) for ``models`` in the background, after anything asked for; replaces the
        models prepared before (their clips that have not started yet are dropped)."""
        poses = list(yoga.POSES) if poses is None else list(poses)
        models = [m for m in models if yoga.supports(m)]
        with self._lock:
            for job in [j for j, (priority, *_rest) in self._queue.items() if priority == LATER]:
                del self._queue[job]
        for model in models:
            for pose in poses:
                self.request(model, pose, LATER)

    def close(self) -> None:
        """Stop the worker; clips being built are dropped (asking for a clip again starts a new worker)."""
        with self._lock:
            self._generation += 1
            self._queue.clear()
            self._running = None
            self._lock.notify_all()
            worker, self._worker = self._worker, None
        if worker is not None:
            _stop(worker)

    # -- background ---------------------------------------------------------------------------------------------

    def _start(self) -> None:
        """Make sure a thread works through the queue (called with the lock held)."""
        if self._thread is None or not self._thread.is_alive() or self._thread_generation != self._generation:
            self._thread_generation = self._generation
            self._thread = threading.Thread(target=self._loop, args=(self._generation,), name="yoga-clips",
                                            daemon=True)
            self._thread.start()

    def _next(self, generation: int):
        with self._lock:
            while not self._queue and self._generation == generation:
                self._lock.wait()
            if self._generation != generation:
                return None
            job = min(self._queue, key=lambda j: self._queue[j][:2])
            model = self._queue.pop(job)[2]
            self._running = job
            return job, model

    def _loop(self, generation: int) -> None:
        """Build the queued clips one after the other until :meth:`close` (each thread has its own worker)."""
        worker = None
        try:
            while True:
                item = self._next(generation)
                if item is None:
                    return
                (key, pose), model = item
                path = cache_path(self.folder, model, pose) if self.folder is not None else None
                clip, error = None, None
                try:
                    if self.process:
                        try:
                            worker = self._ensure_worker(worker, generation)
                            clip = _in_worker(worker, _light(model), pose, path, key)
                        except _WorkerGone as exc:
                            if self._generation != generation:
                                return  # closed meanwhile
                            print(f"[yoga] the worker process stopped ({exc}); building clips in the demo's process "
                                  "instead", file=sys.stderr)
                            self.process = False
                    if clip is None:
                        clip = load_or_build(model, pose, path, key)
                except Exception as exc:  # noqa: BLE001 - reported to whoever asks for the clip
                    error = f"{type(exc).__name__}: {exc}"
                with self._lock:
                    if self._generation != generation:
                        return
                    self._running = None
                    if clip is not None:
                        self._ready[(key, pose)] = clip
                    else:
                        self._errors[(key, pose)] = error
        finally:
            if worker is not None and worker.poll() is None:
                _stop(worker)

    def _ensure_worker(self, worker, generation: int) -> subprocess.Popen:
        if worker is not None and worker.poll() is None:
            return worker
        with self._lock:
            if self._generation != generation:
                raise _WorkerGone("closed")
            self._worker = _spawn()
            return self._worker


def _in_worker(worker: subprocess.Popen, model, pose, path, key) -> AnimationClip:
    """Have ``worker`` build a clip (see :func:`_serve`)."""
    try:
        pickle.dump((model, pose, None if path is None else str(path), key), worker.stdin,
                    protocol=pickle.HIGHEST_PROTOCOL)
        worker.stdin.flush()
        clip, error = pickle.load(worker.stdout)
    except (OSError, EOFError, ValueError, pickle.UnpicklingError, AttributeError) as exc:
        raise _WorkerGone(f"{type(exc).__name__}: {exc}") from None
    if error is not None:
        raise RuntimeError(error)
    return clip


class _WorkerGone(Exception):
    pass


_WORKER = "from bifold_demo.characters.yoga_clips import worker_main; worker_main()"


def _spawn() -> subprocess.Popen:
    """Start the worker with the same Python and the same ``bifold_demo`` package as this process."""
    env = dict(os.environ)
    root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = root + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    try:
        return subprocess.Popen([sys.executable, "-c", _WORKER], stdin=subprocess.PIPE, stdout=subprocess.PIPE, env=env)
    except OSError as exc:
        raise _WorkerGone(f"cannot start {sys.executable}: {exc}") from None


def _stop(worker: subprocess.Popen) -> None:
    try:
        worker.stdin.close()
        worker.wait(timeout=2.0)
    except (OSError, subprocess.TimeoutExpired):
        worker.kill()


def _serve(inp, out) -> None:
    """The worker loop: ``(model, pose, path, key)`` jobs in, ``(clip, error)`` results out."""
    while True:
        try:
            model, pose, path, key = pickle.load(inp)
        except EOFError:
            return
        try:
            result = (load_or_build(model, pose, None if path is None else Path(path), key), None)
        except Exception as exc:  # noqa: BLE001 - sent back to the demo
            result = (None, f"{type(exc).__name__}: {exc}")
        pickle.dump(result, out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()


def worker_main() -> None:
    """The worker process (see the module docs)."""
    if hasattr(os, "nice"):
        os.nice(10)  # the demo comes first
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl+C is for the demo, which then closes our stdin
    results = os.fdopen(os.dup(sys.stdout.fileno()), "wb")
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())  # anything printed goes to stderr, not into the results
    _serve(sys.stdin.buffer, results)


if __name__ == "__main__":
    worker_main()
