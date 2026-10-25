import os
import textwrap
import time
from pathlib import Path

import pytest

from bifold_demo.loader import SceneReloader, load_scene
from bifold_demo.scene import Scene

EXAMPLES = sorted((Path(__file__).parent.parent / "examples").glob("*.py"))


def _write(path, body):
    path.write_text(textwrap.dedent(body))


def test_builtin_scenes(monkeypatch):
    import bifold_demo.scenes.fair as fair

    monkeypatch.setattr(fair, "people_for_scene", lambda folder: [])
    scene, path = load_scene(None)  # the default: the trade-fair hall
    assert isinstance(scene, Scene) and path.name == "fair.py" and "stage_screen" in scene
    scene, path = load_scene("bifold_demo.scenes.basic")
    assert {"poster", "cube", "ball", "floor"} <= set(scene.objects)


def test_scene_file_variants(tmp_path):
    f = tmp_path / "my_scene.py"
    _write(f, """
        from bifold_demo import Scene, Box
        def build_scene():
            s = Scene(); s.add(Box(), name="a"); return s
        def other():
            s = Scene(); s.add(Box(), name="b"); return s
    """)
    assert "a" in load_scene(str(f))[0]
    assert "b" in load_scene(f"{f}:other")[0]
    g = tmp_path / "var_scene.py"
    _write(g, """
        from bifold_demo import Scene, Box
        scene = Scene()
        scene.add(Box(), name="c")
    """)
    assert "c" in load_scene(str(g))[0]


def test_bad_scene_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_scene(str(tmp_path / "missing.py"))
    f = tmp_path / "empty.py"
    f.write_text("x = 1\n")
    with pytest.raises(AttributeError):
        load_scene(str(f))


def test_reloader_picks_up_changes_and_survives_errors(tmp_path, capsys):
    f = tmp_path / "live.py"
    _write(f, """
        from bifold_demo import Scene, Box
        def build_scene():
            s = Scene(); s.add(Box(), name="first"); return s
    """)
    scene, path = load_scene(str(f))
    reloader = SceneReloader(str(f), path)
    assert reloader.check() is None

    def touch():
        stamp = time.time() + touch.offset
        touch.offset += 5
        os.utime(f, (stamp, stamp))

    touch.offset = 5
    f.write_text("this is not python\n")
    touch()
    assert reloader.check() is None
    assert "failed" in capsys.readouterr().out

    _write(f, """
        from bifold_demo import Scene, Box
        def build_scene():
            s = Scene(); s.add(Box(), name="second"); return s
    """)
    touch()
    assert "second" in reloader.check()


@pytest.mark.parametrize("example", EXAMPLES, ids=[p.stem for p in EXAMPLES])
def test_examples_build_and_update(example):
    scene, _ = load_scene(str(example))
    for i in range(30):
        scene.update(i / 10, 0.1 if i else 0.0)
    assert len(scene) > 0
