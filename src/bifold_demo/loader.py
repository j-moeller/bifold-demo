"""Loading scenes from Python files or modules, and reloading them when the file changes."""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import os
import sys
import traceback
from pathlib import Path

from .scene import Scene

DEFAULT_SCENE = "bifold_demo.scenes.fair"
_counter = itertools.count()


def _split(spec: str) -> tuple[str, str | None, bool]:
    """``"target[:function]"`` -> (target, function, target is a file path)."""
    target, func = spec, None
    head, sep, tail = spec.rpartition(":")
    if sep and head and tail.isidentifier():
        target, func = head, tail
    is_file = target.endswith(".py") or os.sep in target or Path(target).is_file()
    return target, func, is_file


def load_scene(spec: str | None = None) -> tuple[Scene, Path | None]:
    """Build a scene from ``spec`` and return it with the file it came from.

    ``spec`` is ``"path/to/scene.py"`` or a module name (``"bifold_demo.scenes.basic"``),
    optionally followed by ``":function"``. Without a function name the module's
    ``build_scene()`` is called, or its module-level ``scene`` is used.
    """
    target, func, is_file = _split(spec or DEFAULT_SCENE)
    if is_file:
        path = Path(target).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"scene file not found: {path}")
        module = _import_file(path)
    else:
        module = importlib.import_module(target)
        path = Path(module.__file__).resolve() if getattr(module, "__file__", None) else None
    return _scene_from(module, func), path


def _import_file(path: Path):
    name = f"_bifold_scene_{next(_counter)}"  # fresh module on every (re)load
    module_spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(module_spec)
    sys.path.insert(0, str(path.parent))  # let scene files import helpers next to them
    try:
        module_spec.loader.exec_module(module)
    finally:
        sys.path.remove(str(path.parent))
    return module


def _scene_from(module, func: str | None) -> Scene:
    if func:
        factory = getattr(module, func)
    elif hasattr(module, "build_scene"):
        factory = module.build_scene
    elif hasattr(module, "scene"):
        factory = module.scene
    else:
        raise AttributeError(f"{module.__name__} defines neither build_scene() nor a 'scene' variable")
    scene = factory if isinstance(factory, Scene) else factory()
    if not isinstance(scene, Scene):
        raise TypeError(f"expected a Scene, got {type(scene).__name__}")
    return scene


class SceneReloader:
    """Rebuilds the scene when its source file is saved."""

    def __init__(self, spec: str | None, path: Path | None):
        self.spec, self.path = spec, path
        self._mtime = self._stat()

    def _stat(self):
        try:
            return self.path.stat().st_mtime if self.path else None
        except OSError:
            return None

    def check(self) -> Scene | None:
        """Return a freshly built scene if the file changed (and builds), else ``None``."""
        mtime = self._stat()
        if mtime is None or mtime == self._mtime:
            return None
        self._mtime = mtime
        try:
            target, _func, is_file = _split(self.spec or DEFAULT_SCENE)
            if not is_file and target in sys.modules:
                importlib.reload(sys.modules[target])
            scene, _ = load_scene(self.spec)
        except Exception:  # noqa: BLE001 - keep running the previous scene
            print(f"[scene] reloading {self.path.name} failed; keeping the previous scene:")
            traceback.print_exc()
            return None
        print(f"[scene] reloaded {self.path.name}")
        return scene
