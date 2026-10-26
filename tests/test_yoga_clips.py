"""Building yoga clips in the background: priorities, a worker that dies, the disk cache and bifold-characters yoga."""

import time
from dataclasses import replace

import pytest
from test_yoga import synthetic_biped

from bifold_demo.characters import cli, rocketbox, yoga_clips
from bifold_demo.characters.yoga_clips import YogaClips, model_key


@pytest.fixture(scope="module")
def biped():
    return synthetic_biped()


def _wait(clips, model, pose, seconds=120.0):
    deadline = time.time() + seconds
    while time.time() < deadline:
        if clips.get(model, pose) is not None or clips.error(model, pose) is not None:
            return
        time.sleep(0.02)
    raise AssertionError(f"{pose} not built in {seconds} s")


def test_the_newest_request_goes_first_and_solving_ahead_comes_last(biped):
    clips = YogaClips(cache_dir=False, process=False)
    try:
        with clips._lock:  # the building thread waits until the queue is filled
            clips.prepare([biped], ["cow", "warrior_1"])
            for pose in ("yogi_squat", "warrior_2", "yogi_squat"):  # yogi squat asked for again, last
                clips.request(biped, pose)
            order = [pose for _key, pose in sorted(clips._queue, key=lambda job: clips._queue[job][:2])]
        assert order == ["yogi_squat", "warrior_2", "cow", "warrior_1"]
    finally:
        clips.close()


def test_a_worker_that_dies_while_solving_is_replaced(biped, monkeypatch):
    real, workers = yoga_clips._in_worker, []

    def dying(worker, *args):
        workers.append(worker.pid)
        if len(workers) == 1:  # the first worker dies while it solves (out of memory, killed, ...)
            worker.kill()
            worker.wait()
        return real(worker, *args)

    monkeypatch.setattr(yoga_clips, "_in_worker", dying)
    clips = YogaClips(cache_dir=False)
    try:
        clips.request(biped, "warrior_2")
        _wait(clips, biped, "warrior_2")
        assert "worker process stopped" in clips.error(biped, "warrior_2")
        assert clips.process  # the next clip gets a new worker, it is not solved in the demo's process
        clips.request(biped, "warrior_2")  # asked again
        _wait(clips, biped, "warrior_2")
        assert clips.get(biped, "warrior_2") is not None and len(set(workers)) == 2
    finally:
        clips.close()


def test_the_cache_key_covers_the_skinning(biped):
    moved = replace(biped, inverse_bind=biped.inverse_bind.copy())
    moved.inverse_bind[0, 1, 3] += 0.03
    assert model_key(moved) != model_key(biped) and model_key(replace(biped)) == model_key(biped)


def test_files_are_written_whole_or_not_at_all(tmp_path):
    def broken(f):
        f.write(b"half a file")
        raise OSError("disk full")

    with pytest.raises(OSError):
        rocketbox._write_atomic(tmp_path / "clip.npz", broken)
    assert list(tmp_path.iterdir()) == []  # neither the file nor a temporary one


def test_bifold_characters_yoga_solves_ahead_and_reports_what_it_cannot_save(biped, tmp_path, capsys):
    good, blocked = tmp_path / "good", tmp_path / "blocked"
    for folder in (good, blocked):
        rocketbox.save_character(biped, folder, {"sex": "female", "age_group": "adult"})
    assert cli.main(["yoga", "--dest", str(good), "--poses", "warrior_2", "--jobs", "1"]) == 0
    assert "1 clip ready" in capsys.readouterr().out
    path = yoga_clips.cache_path(good / "yoga", biped, "warrior_2")
    assert rocketbox.read_clip(path).metadata["cache_key"] == model_key(biped)
    (blocked / "yoga").write_text("not a folder")
    assert cli.main(["yoga", "--dest", str(blocked), "--poses", "warrior_2", "--jobs", "1"]) == 1
    assert "1 failed" in capsys.readouterr().out
