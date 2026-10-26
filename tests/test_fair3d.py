import pytest

from bifold_demo.scenes import fair, fair3d


@pytest.mark.parametrize("heading", [0.0, 10.0, -25.0, 39.0, 90.0, 180.0])
def test_nobody_faces_the_camera(heading):
    position = (0.0, 0.0, -4.0)  # straight in front of the camera; heading 0 looks right at it
    turned = fair3d.away_from_camera(position, heading)
    to_camera = 0.0
    off = abs((turned - to_camera + 180) % 360 - 180)
    assert off >= fair3d.MIN_ANGLE_TO_CAMERA - 1e-9
    if abs((heading + 180) % 360 - 180) >= fair3d.MIN_ANGLE_TO_CAMERA:
        assert turned == heading  # already looking elsewhere: unchanged


def test_without_prepared_people_the_hall_is_empty(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("BIFOLD_CHARACTERS", str(tmp_path))
    scene = fair3d.build_scene()
    assert "booth1_wall" in scene and not any(hasattr(o, "skinned_model") for o in scene)
    assert "bifold-characters fetch" in capsys.readouterr().out


def test_fair_hall_can_be_built_without_2d_people():
    scene = fair.build_scene(people=[])
    assert "visitor1" not in scene and "stage_screen" in scene
    assert len(fair.booth_layout()) == 2 * len(fair.BOOTH_ROWS)
