import os
import time

import cv2
import numpy as np
import pytest

from bifold_demo.scene import Box, Keyframes, Quad, Scene, Sphere


def test_names_and_lookup():
    scene = Scene()
    a = scene.add(Box())
    b = scene.add(Box())
    c = scene.add(Sphere(), name="ball")
    assert (a.name, b.name, c.name) == ("box1", "box2", "ball")
    assert scene["ball"] is c and "ball" in scene and len(scene) == 3
    with pytest.raises(ValueError):
        scene.add(Box(), name="ball")
    with pytest.raises(KeyError, match="box1"):
        scene["nope"]
    scene.remove("ball")
    assert "ball" not in scene


def test_set_texture_validates_immediately(two_tone_png):
    scene = Scene()
    scene.add(Box(), name="crate")
    scene.set_texture("crate", two_tone_png)
    scene.update(0, 0)
    assert scene["crate"].current_texture.name == "two_tone.png"
    with pytest.raises(FileNotFoundError):
        scene["crate"].texture = "missing.png"


def test_every_property_can_be_a_function_of_time():
    scene = Scene()
    box = scene.add(Box(position=lambda t: (t, 0, 0), rotation=lambda t: (0, 10 * t, 0), scale=lambda t: 1 + t,
                        color=Keyframes({0: (0, 0, 0), 1: (255, 255, 255)}), visible=lambda t: t < 1,
                        texture=lambda t: "#ff0000" if t < 0.5 else "#00ff00"))
    scene.update(0.5, 0.5)
    np.testing.assert_allclose(box.current_position, (0.5, 0, 0))
    np.testing.assert_allclose(box.current_rotation, (0, 5, 0))
    np.testing.assert_allclose(box.current_scale, (1.5, 1.5, 1.5))
    assert box.current_color == (128, 128, 128)
    assert box.current_visible
    assert box.current_texture.rgba[0, 0].tolist() == [0, 255, 0, 255]
    scene.update(1.0, 0.5)
    assert not box.current_visible


def test_broken_dynamic_texture_keeps_previous_one(capsys):
    scene = Scene()
    box = scene.add(Box(texture=lambda t: "#ff0000" if t < 1 else "missing.png"))
    scene.update(0, 0)
    good = box.current_texture
    scene.update(1, 1)
    assert box.current_texture is good
    assert "missing.png" in capsys.readouterr().out


def test_quad_width_follows_texture_aspect(two_tone_png):
    scene = Scene()
    quad = scene.add(Quad(texture=two_tone_png, height=1.0))
    scene.update(0, 0)
    width = np.ptp(quad.mesh.positions[:, 0])
    assert width == pytest.approx(2.0)
    quad.texture = "#ffffff"  # 1x1 texture: square
    scene.update(0.1, 0.1)
    assert np.ptp(quad.mesh.positions[:, 0]) == pytest.approx(1.0)
    quad.height = 3.0
    assert np.ptp(quad.mesh.positions[:, 1]) == pytest.approx(3.0)


def test_changing_dimensions_rebuilds_mesh():
    box = Box(size=1)
    first = box.mesh
    box.size = (2, 1, 1)
    assert box.mesh is not first
    assert np.ptp(box.mesh.positions[:, 0]) == pytest.approx(2)


def test_face_camera_turns_towards_camera():
    scene = Scene()
    quad = scene.add(Quad(size=1, position=(0, 0, 0), face_camera=True))
    scene.camera.position = (5, 0, 0)
    scene.update(0, 0)
    normal = quad.model_matrix(scene.camera.current_position)[:3, :3] @ np.array([0, 0, 1.0])
    np.testing.assert_allclose(normal, (1, 0, 0), atol=1e-9)


def test_callbacks_receive_time():
    scene = Scene()
    seen = []
    scene.on_update(lambda s, t, dt: seen.append(("scene", t, dt)))
    scene.add(Box(on_update=lambda o, t, dt: seen.append((o.name, t, dt))))
    scene.update(1.0, 0.5)
    assert seen == [("scene", 1.0, 0.5), ("box1", 1.0, 0.5)]


def test_upload_folder_applies_replaces_and_restores(tmp_path, two_tone_png, monkeypatch):
    from bifold_demo.scene import textures

    monkeypatch.setattr(textures._FileCache, "check_interval", 0.0)
    scene = Scene()
    poster = scene.add(Quad(texture="#123456"), name="Poster")
    uploads = tmp_path / "uploads"
    watcher = scene.watch_folder(uploads)
    assert scene.watch_folder(uploads) is watcher  # no duplicate watchers
    assert uploads.is_dir()

    target = uploads / "poster.png"  # case-insensitive match
    target.write_bytes(two_tone_png.read_bytes())
    watcher.poll(force=True)
    scene.update(0, 0)
    assert poster.current_texture.rgba[0, 0].tolist() == [255, 0, 0, 255]

    cv2.imwrite(str(target), np.full((4, 4, 3), (0, 255, 0), np.uint8))
    stamp = time.time() + 5
    os.utime(target, (stamp, stamp))
    watcher.poll(force=True)
    scene.update(0.1, 0.1)
    assert poster.current_texture.rgba[0, 0].tolist() == [0, 255, 0, 255]

    target.unlink()
    watcher.poll(force=True)
    assert poster.texture == "#123456"


def test_upload_folder_ignores_unknown_names(tmp_path, two_tone_png, capsys):
    scene = Scene()
    scene.add(Box(), name="crate")
    watcher = scene.watch_folder(tmp_path)
    (tmp_path / "other.png").write_bytes(two_tone_png.read_bytes())
    watcher.poll(force=True)
    watcher.poll(force=True)
    out = capsys.readouterr().out
    assert out.count("no object named 'other'") == 1  # warned once
