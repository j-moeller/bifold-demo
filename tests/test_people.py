"""The people component. The text-to-image model itself is replaced by fakes here: these tests check
the plumbing around it (prompting, cutting out, quality checks, storage, CLI), not image quality."""

import json

import cv2
import numpy as np
import pytest
from PIL import Image

from bifold_demo.people import PeopleLibrary, people_for_scene, random_spec
from bifold_demo.people import cli
from bifold_demo.people.cutout import Cutout, cut_out, refine_mask
from helpers import silhouette, studio_photo

from bifold_demo.people.generator import (
    PRESETS,
    ImageGenerator,
    PeopleGenerator,
    check_cutout,
    get_preset,
    preset_with,
)


def test_specs_are_reproducible_and_describe_a_full_body():
    a, b = random_spec(5), random_spec(5)
    assert a == b and a.prompt() == b.prompt()
    walker = random_spec(9, pose="walking")
    assert walker.facing == "left" and "side" in walker.prompt()
    standing = random_spec(9, pose="standing")
    assert standing.facing == "front" and "Full-body" in standing.prompt()
    assert 1.4 < standing.height < 2.0
    assert len({random_spec(s).prompt() for s in range(30)}) == 30
    with pytest.raises(ValueError):
        random_spec(1, pose="sitting")


def test_cut_out_flags_people_touching_the_border():
    image, mask = studio_photo()
    inside = cut_out(image, mask)
    assert inside.full_body and not inside.touches
    assert inside.rgba.shape[2] == 4 and inside.rgba[0, 0, 3] == 0  # transparent corner
    image, mask = studio_photo(person_scale=0.95, top=60)  # feet run off the bottom
    assert not cut_out(image, mask).full_body


def test_refine_mask_snaps_a_rough_mask_to_the_person():
    image, truth = studio_photo()
    rough = cv2.dilate(truth, np.ones((15, 15), np.uint8))  # a sloppy segmentation
    refined = refine_mask(image, rough)
    iou = (refined & truth).sum() / (refined | truth).sum()
    assert iou > 0.95


def _cutout(height, touches=(), confidence=0.9, area_scale=1.0):
    rgba = silhouette(height, max(4, int(height * 0.4 * area_scale)))
    return Cutout(rgba, (0, 0, rgba.shape[1], height), confidence, set(touches))


def test_quality_check():
    assert check_cutout([], 1000) == (None, "no person found")
    assert check_cutout([_cutout(900)], 1000)[1] == "ok"
    assert check_cutout([_cutout(900, ["bottom"])], 1000)[1] == "head or feet cut off"
    assert check_cutout([_cutout(900, ["left"])], 1000)[1] == "cut off at the side"
    assert check_cutout([_cutout(300)], 1000)[1] == "person too small"
    assert check_cutout([_cutout(900, confidence=0.3)], 1000)[1].startswith("unsure")
    assert check_cutout([_cutout(900), _cutout(800)], 1000)[1] == "more than one person"


def test_library_round_trip(tmp_path):
    lib = PeopleLibrary(tmp_path / "people")
    assert lib.people() == []
    person = lib.add(silhouette(), name="walking 7", height=1.81, pose="walking", facing="left", seed=7)
    again = lib.add(silhouette(), name="walking 7")
    assert person.name == "walking_7" and again.name == "walking_7_2"
    meta = json.loads(person.path.with_suffix(".json").read_text())
    assert meta["height"] == 1.81 and meta["facing"] == "left" and "path" not in meta

    loaded = {p.name: p for p in lib.people()}
    assert loaded["walking_7"].height == 1.81 and loaded["walking_7"].pose == "walking"
    tex = loaded["walking_7"].texture()
    assert tex.rgba.shape == (200, 80, 4)
    flipped = loaded["walking_7"].facing_texture("right")
    np.testing.assert_array_equal(flipped.rgba, tex.rgba[:, ::-1])
    assert loaded["walking_7"].facing_texture("left") is tex

    cv2.imwrite(str(lib.folder / "dropped_in.png"), cv2.cvtColor(silhouette(), cv2.COLOR_RGBA2BGRA))
    dropped = {p.name: p for p in lib.people()}["dropped_in"]
    assert dropped.full_body and dropped.pose == "standing"  # PNGs without metadata get defaults
    assert lib.contact_sheet().shape[2] == 3


def test_people_for_scene_prefers_the_library(tmp_path, monkeypatch):
    import bifold_demo.people as people

    monkeypatch.setattr(people, "default_people", lambda: ["default"])
    assert people_for_scene(tmp_path / "empty") == ["default"]
    lib = PeopleLibrary(tmp_path / "mine")
    lib.add(silhouette(), name="me")
    assert [p.name for p in people_for_scene(lib.folder)] == ["me"]


def test_presets():
    assert get_preset("flux2-klein").pipeline == "Flux2KleinPipeline"
    assert get_preset("someone/some-model").pipeline == "AutoPipelineForText2Image"
    with pytest.raises(ValueError):
        get_preset("nonsense")
    changed = preset_with(PRESETS["sd-turbo"], steps=4, size=(500, 770))
    assert (changed.steps, changed.width, changed.height) == (4, 496, 768)


class FakePipeline:
    """Stands in for a diffusers pipeline: draws a figure, records how it was called."""

    def __init__(self, bad_first=0):
        self.calls, self.bad_first = [], bad_first

    def __call__(self, prompt, width, height, num_inference_steps, guidance_scale, generator):
        self.calls.append(dict(prompt=prompt, size=(width, height), steps=num_inference_steps,
                               guidance=guidance_scale, seed=generator.initial_seed()))
        top = height // 3 if len(self.calls) <= self.bad_first else None  # feet cut off -> rejected
        image, _ = studio_photo(height, width, top=top)
        return type("Output", (), {"images": [Image.fromarray(image[..., ::-1])]})()


def test_image_generator_passes_the_preset_settings():
    fake = FakePipeline()
    images = ImageGenerator(PRESETS["flux2-klein"], pipeline=fake)
    out = images("a person", seed=11)
    assert out.shape == (1344, 768, 3) and tuple(out[0, 0]) == (205, 205, 205)  # BGR
    assert fake.calls == [dict(prompt="a person", size=(768, 1344), steps=4, guidance=1.0, seed=11)]


class MaskSegmenter:
    """Segments the fake studio photos by colour instead of running YOLO."""

    def cutouts(self, image):
        mask = (image[..., 0] < 128).astype(np.uint8)
        cutout = cut_out(image, mask, 0.9)
        return [cutout] if cutout else []


def test_people_generator_retries_and_stores_people(tmp_path):
    fake = FakePipeline(bad_first=1)
    lib = PeopleLibrary(tmp_path / "people")
    generator = PeopleGenerator(ImageGenerator(PRESETS["sd-turbo"], pipeline=fake), MaskSegmenter(), lib,
                                raw_folder=tmp_path / "raw")
    made = generator.generate(2, seed=100, pose="walking")
    assert len(made) == 2 and len(fake.calls) == 3  # the first image was rejected and retried
    assert fake.calls[0]["seed"] != fake.calls[1]["seed"]
    person = lib.people()[0]
    assert person.pose == "walking" and person.facing == "left" and person.source.endswith("sd-turbo")
    assert "Full-body" in person.prompt
    assert len(list((tmp_path / "raw").glob("*.jpg"))) == 3


def test_cli_without_a_model(tmp_path, capsys, monkeypatch):
    out = str(tmp_path / "people")
    assert cli.main(["models"]) == 0 and "FLUX.2-klein-4B" in capsys.readouterr().out
    assert cli.main(["list", "--out", out]) == 0 and "no people yet" in capsys.readouterr().out
    assert cli.main(["generate", "--model", "nonsense", "--out", out]) == 2

    import bifold_demo.people.generator as generator

    def no_diffusers(*args, **kwargs):
        raise ImportError("No module named 'diffusers'")

    monkeypatch.setattr(generator.ImageGenerator, "__init__", no_diffusers)
    assert cli.main(["generate", "-n", "1", "--out", out]) == 1
    assert 'pip install -e ".[people]"' in capsys.readouterr().err


@pytest.mark.yolo
def test_cutting_people_out_of_the_sample_photos(tmp_path):
    pytest.importorskip("ultralytics")
    from bifold_demo.people import default_people
    from bifold_demo.scene import sample_image

    if sample_image("bus.jpg") is None:
        pytest.skip("ultralytics sample images not installed")
    try:
        people = default_people(cache=tmp_path / "cache", models_dir=tmp_path / "models")
    except Exception as exc:  # noqa: BLE001 - offline machine
        pytest.skip(f"cannot load YOLO segmentation weights: {exc}")
    full = [p for p in people if p.full_body]
    busts = [p for p in people if not p.full_body]
    assert len(full) >= 2 and len(busts) >= 1
    for person in people:
        rgba = person.texture().rgba
        assert rgba[..., 3].min() == 0 and rgba[..., 3].max() == 255
