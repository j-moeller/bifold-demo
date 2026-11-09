"""The head's shape fitted to a person, on synthetic data (no avatars, no ICT-FaceKit, no MediaPipe, no OpenGL).

A synthetic avatar (a sphere head with a nose, two eyeballs on their eye bones, an open neck, the Rocketbox face bones
the fit uses, hair cards and a body) and a tiny identity space made from the head itself (scaled and moved, so the
alignment has a similarity to find; three smooth identity modes and two expressions) stand in for a Rocketbox avatar
and ICT-FaceKit; the photos' landmarks are the true landmarks projected by known cameras.
"""

import dataclasses
import math

import numpy as np
import pytest

from bifold_demo.characters.headshape import fit as shapefit
from bifold_demo.characters.headshape import head as s0
from bifold_demo.characters.headshape import mesh, rig
from bifold_demo.characters.headshape import space as identity
from bifold_demo.characters.headtex import canonical as cf
from bifold_demo.characters.headtex import raycast
from bifold_demo.characters.headtex.camera import PhotoCamera
from bifold_demo.characters.headtex.landmarks import HeadLandmarks
from bifold_demo.characters.model import CharacterModel, Skeleton, SkinnedPrimitive

CENTRE = np.array([0.0, 1.6, 0.0])
RADIUS = 0.09
EYES = np.array([[-0.032, 1.62, 0.066], [0.032, 1.62, 0.066]])  # the eyeballs' fronts just under the skin
EYE_RADIUS = 0.012
SCALE = 1.1  # the avatar's head is this much larger than the tiny space's
SHIFT = np.array([0.01, -1.55, 0.02])  # ... and the space sits here


def _sphere(centre, radius, rows, cols, nose=False, open_bottom=0):
    v, u = np.meshgrid(np.linspace(0.0, 1.0, rows + 1), np.linspace(0.0, 1.0, cols + 1), indexing="ij")
    theta, phi = v * math.pi, (u - 0.5) * 2 * math.pi
    d = np.stack([np.sin(theta) * np.sin(phi), np.cos(theta), np.sin(theta) * np.cos(phi)], axis=-1).reshape(-1, 3)
    p = d * radius
    if nose:
        bump = np.exp(-((p[:, 0] / 0.012) ** 2 + ((p[:, 1] + 0.012) / 0.016) ** 2)) * (p[:, 2] > 0)
        p = p + d * (0.022 * bump)[:, None]
    tri = []
    for r in range(rows - open_bottom):
        for c in range(cols):
            a, b = r * (cols + 1) + c, r * (cols + 1) + c + 1
            e, f = a + cols + 1, b + cols + 1
            tri += [(a, e, b), (b, e, f)]
    tri = np.array(tri)
    corners = p[tri]
    keep = np.linalg.norm(np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]), axis=1) > 1e-12
    tri = tri[keep]
    used = np.unique(tri)
    remap = np.full(len(p), -1)
    remap[used] = np.arange(len(used))
    p, uv = p[used] + centre, np.c_[u.ravel(), 1.0 - v.ravel()][used]
    tri = remap[tri]
    normals = mesh.vertex_normals(p, tri)
    return p, normals, uv, tri


def synthetic_avatar() -> CharacterModel:
    """See the module docs."""
    bones = [("Bip01", None, (0.0, 0.0, 0.0)), ("Bip01 Spine2", "Bip01", (0.0, 1.3, 0.0)),
             ("Bip01 Neck", "Bip01 Spine2", (0.0, 1.45, 0.0)), ("Bip01 Head", "Bip01 Neck", (0.0, 1.55, 0.0)),
             ("Bip01 REye", "Bip01 Head", tuple(EYES[0])), ("Bip01 LEye", "Bip01 Head", tuple(EYES[1])),
             ("Bip01 RMouthCorner", "Bip01 Head", (-0.024, 1.556, 0.075)),
             ("Bip01 LMouthCorner", "Bip01 Head", (0.024, 1.556, 0.075)),
             ("Bip01 MNose", "Bip01 Head", (0.0, 1.59, 0.09)), ("Bip01 MJaw", "Bip01 Head", (0.0, 1.57, 0.02))]
    names = [b[0] for b in bones]
    parents = [names.index(p) if p else -1 for _, p, _ in bones]
    position = np.array([b[2] for b in bones], float)
    rest = np.repeat(np.eye(4)[None], len(bones), axis=0)
    for k, p in enumerate(parents):
        rest[k, :3, 3] = position[k] - (position[p] if p >= 0 else 0)
    skeleton = Skeleton(names, parents, rest)
    skin, skin_n, skin_uv, skin_tri = _sphere(CENTRE, RADIUS, 24, 40, nose=True, open_bottom=4)
    parts = [(skin, skin_n, skin_uv, skin_tri, None)]
    for e, bone in ((0, "Bip01 REye"), (1, "Bip01 LEye")):
        parts.append((*_sphere(EYES[e], EYE_RADIUS, 6, 10), bone))
    points, normals, uvs, tris, joints = [], [], [], [], []
    start = 0
    for p, n, uv, t, bone in parts:
        points.append(p)
        normals.append(n)
        uvs.append(uv)
        tris.append(t + start)
        j = np.zeros((len(p), 4), np.int64)
        if bone is None:  # the skin: the head, the neck's lowest rows on the neck and spine
            j[:, 0] = np.where(p[:, 1] < CENTRE[1] - 0.075, names.index("Bip01 Spine2"), names.index("Bip01 Head"))
        else:
            j[:, 0] = names.index(bone)
        joints.append(j)
        start += len(p)
    points, normals, uvs = np.concatenate(points), np.concatenate(normals), np.concatenate(uvs)
    joints = np.concatenate(joints)
    weights = np.zeros((len(points), 4), np.float32)
    weights[:, 0] = 1.0
    from bifold_demo.characters.tangents import vertex_tangents

    tri = np.concatenate(tris)
    head = SkinnedPrimitive("s_head", points.astype(np.float32), normals.astype(np.float32), uvs.astype(np.float32),
                            tri.astype(np.uint32), joints, weights, np.full((4, 4, 4), 200, np.uint8),
                            tangents=vertex_tangents(points, normals, uvs, tri))
    cards = np.array([[-0.05, 1.68, 0.03], [0.05, 1.68, 0.03], [0.05, 1.70, -0.05], [-0.05, 1.70, -0.05]])
    hair = SkinnedPrimitive("s_opacity", cards.astype(np.float32), np.tile([0, 1, 0], (4, 1)).astype(np.float32),
                            np.zeros((4, 2), np.float32), np.array([[0, 1, 2], [0, 2, 3]], np.uint32),
                            np.full((4, 4), names.index("Bip01 Head")), np.tile([1, 0, 0, 0], (4, 1)).astype(
                                np.float32), np.full((2, 2, 4), 90, np.uint8))
    b, bn, buv, bt = _sphere(np.array([0.0, 1.2, 0.0]), 0.2, 8, 12)
    body = SkinnedPrimitive("s_body", b.astype(np.float32), bn.astype(np.float32), buv.astype(np.float32),
                            bt.astype(np.uint32), np.full((len(b), 4), names.index("Bip01 Spine2")),
                            np.tile([1, 0, 0, 0], (len(b), 1)).astype(np.float32), np.full((4, 4, 4), 50, np.uint8))
    return CharacterModel("Synthetic", skeleton, [body, head, hair], np.linalg.inv(skeleton.rest_global),
                          {"skin vertices": len(skin)})


def true_marks(model):
    """478 landmarks on the front of the head (cast from straight ahead), as stage 0 would find them."""
    head = model.primitives[1]
    corners = raycast.corners_of(head.positions, head.indices)
    skin = (np.asarray(head.indices, np.int64).reshape(-1, 3) < model.metadata["skin vertices"]).all(axis=1)
    rng = np.random.default_rng(7)
    r = 0.06 * np.sqrt(rng.random(cf.COUNT))
    a = rng.random(cf.COUNT) * 2 * math.pi
    xy = np.c_[r * np.cos(a), CENTRE[1] - 0.005 + 1.1 * r * np.sin(a)]
    xy[cf.RIGHT_IRIS[0]], xy[cf.LEFT_IRIS[0]] = EYES[0, :2], EYES[1, :2]  # the irises where the eyes are
    origins = np.c_[xy, np.full(cf.COUNT, 1.0)]
    _d, tri, bary = raycast.intersect(origins, np.tile([0.0, 0.0, -1.0], (cf.COUNT, 1)), corners[skin])
    assert (tri >= 0).all()
    tri = np.nonzero(skin)[0][tri]
    points = raycast.point_on(corners, tri, bary)
    return HeadLandmarks(points, tri, bary, np.ones(cf.COUNT), 1, "mediapipe", 5, "synthetic")


def tiny_space(model):
    """An identity space made from the head's skin (see the module docs) and its landmarks."""
    head = model.primitives[1]
    skin_count = model.metadata["skin vertices"]  # (the eyeballs' vertices come after the skin's)
    p = np.asarray(head.positions, np.float64)[:skin_count]
    tri = np.asarray(head.indices, np.int64).reshape(-1, 3)
    tri = tri[(tri < skin_count).all(axis=1)]
    local = (p - CENTRE) / SCALE + CENTRE + SHIFT  # the space's own place and size
    front = np.exp(-((local[:, 2] - (CENTRE + SHIFT)[2] - 0.08) / 0.05) ** 2)

    def bump(x, y, z, width):
        return np.exp(-(((p[:, 0] - x) ** 2 + (p[:, 1] - y) ** 2 + (p[:, 2] - z) ** 2) / width ** 2))

    modes = np.zeros((3, len(p), 3))
    cheeks = bump(0.05, 1.58, 0.06, 0.02) - bump(-0.05, 1.58, 0.06, 0.02)  # fuller cheeks (outwards, both sides)
    modes[0, :, 0] = 0.004 * cheeks
    modes[1, :, 2] = 0.003 * bump(0.0, 1.588, 0.1, 0.02)  # the nose's depth
    modes[2, :, 2] = 0.003 * bump(0.0, 1.63, 0.08, 0.035)  # the brows and the eyes further forward
    expressions = np.zeros((2, len(p), 3))
    expressions[0, :, 1] = -0.006 * np.clip((1.57 - p[:, 1]) / 0.05, 0, 1) * front  # jawOpen
    space = identity.IdentitySpace(local.astype(np.float32), (modes / SCALE).astype(np.float32),
                                   (expressions / SCALE).astype(np.float32), ("jawOpen", "eyeBlink_L"),
                                   tri.astype(np.int32), "tiny", "MIT")
    marks = true_marks(model)
    corners = p[tri]
    _d, t, b = raycast.closest(marks.points, corners)
    landmarks = identity.SpaceLandmarks(tri[t], b, np.ones(cf.COUNT))
    return space, landmarks, marks


@pytest.fixture(scope="module")
def setup():
    model = synthetic_avatar()
    space, landmarks, marks = tiny_space(model)
    return model, space, landmarks, marks


@pytest.fixture
def made(setup, monkeypatch):
    """The head space of the synthetic avatar with the tiny space (not cached)."""
    model, space, landmarks, marks = setup
    monkeypatch.setattr(identity, "space_landmarks", lambda: landmarks)
    return s0.make_head_space(model, marks, space)


def test_the_parts_of_the_head_are_found(made, setup):
    model = setup[0]
    eyes = len(model.primitives[1].positions) - model.metadata["skin vertices"]
    assert (made.parts[made.eye > 0] == s0.EYE).all() and (made.eye > 0).sum() == eyes
    assert (made.parts == s0.SKIN).sum() > 800
    assert made.fade.min() == 0.0 and made.fade.max() == 1.0  # the open neck fades out
    assert np.allclose(made.eye_centre, EYES, atol=1e-3) and np.allclose(made.eye_radius, EYE_RADIUS, atol=1e-3)
    assert list(made.others) == [2]  # the hair cards follow the skin; the body does not


def test_the_space_is_laid_on_the_head_with_its_scale(made, setup):
    """The head is the space's neutral head scaled by 1.1 and moved: the alignment finds the similarity, every skin
    vertex finds its own point, the avatar's coefficients are about 0 and the modes come back 1.1 times larger."""
    _model, space, _landmarks, _marks = setup
    assert made.scale == pytest.approx(SCALE, rel=0.01)
    skin = (made.parts == s0.SKIN) & (made.triangle >= 0)
    assert skin.mean() > 0.5
    assert np.abs(made.alpha).max() < 0.1
    modes = made.modes(space)
    truth = np.zeros_like(modes)
    count = len(space.neutral)
    truth[:, :count] = space.identity * SCALE
    error = np.linalg.norm(modes - truth, axis=2)[:, skin & (made.fade > 0.99)]
    assert np.median(error) < 1e-4 and error.max() < 1.5e-3  # metres: the right points, the right sign and size
    size = np.linalg.norm(truth, axis=2)[:, skin & (made.fade > 0.99)]
    moved = size > 0.001  # where the modes move the head: their size relative to the truth's
    assert np.median(error[moved] / size[moved]) < 0.03
    eyes = made.eye > 0  # an eyeball moves with its rim, as one
    assert np.ptp(modes[0][made.eye == 1], axis=0).max() < 1e-9
    assert np.linalg.norm(modes[0][eyes], axis=1).max() > 0


def test_the_fit_recovers_known_coefficients_and_cameras(made, setup):
    """Exact landmarks of a head with known coefficients, seen by four known cameras, started from wrong ones: the fit
    finds the coefficients and the cameras."""
    model, space, _landmarks, marks = setup
    beta = np.array([0.8, -0.6, 0.5])
    modes = made.modes(space).astype(np.float64)
    target = np.asarray(model.primitives[1].positions, np.float64) + np.einsum("k,knd->nd", beta - made.alpha, modes)
    tri = np.asarray(model.primitives[1].indices, np.int64).reshape(-1, 3)[marks.triangle]
    points = np.einsum("lc,lcd->ld", marks.barycentric, target[tri])
    rng = np.random.default_rng(1)
    photos, truths = [], []
    for k, (yaw, pitch) in enumerate(((0, 0), (35, 0), (-35, 0), (0, 15))):
        truth = PhotoCamera.around(CENTRE, 0.8, yaw, pitch, fov=35.0, size=(640, 480))
        start = PhotoCamera.around(CENTRE + rng.normal(0, 0.004, 3), 0.8 + rng.normal(0, 0.02), yaw + 4, pitch - 3,
                                   fov=35.0, size=(640, 480))
        photos.append(shapefit.Photo(f"photo {k}", (640, 480), truth.project(points), start, True))
        truths.append(truth)
    options = shapefit.FitOptions(expressions=False, iris=False, contour=False, outline=False, free=False,
                                  prior=1e-3, prior_high=1e-3, centre="population", trust=False)
    fit = shapefit.fit_person(photos, model, marks, made, space, options)
    assert fit.usable, fit.reason
    assert np.allclose(fit.beta, beta, atol=0.08)
    for camera, truth in zip(fit.cameras, truths):
        assert np.allclose(camera.angles(), truth.angles(), atol=0.5)
    assert fit.error_after < 0.1 < fit.error_before
    moved = np.linalg.norm(fit.offsets - (target - model.primitives[1].positions), axis=1)
    assert np.median(moved[made.parts == s0.SKIN]) < 3e-4


def test_one_photo_or_no_change_keeps_the_avatars_head(made, setup):
    model, space, _landmarks, marks = setup
    camera = PhotoCamera.around(CENTRE, 0.8, 0, 0, fov=35.0, size=(640, 480))
    photo = shapefit.Photo("front", (640, 480), camera.project(marks.points), camera, True)
    one = shapefit.fit_person([photo], model, marks, made, space)
    assert not one.usable and "only 1 photo" in one.reason and "needs 2" in one.reason
    none = shapefit.fit_person([], model, marks, made, space)
    assert not none.usable and "no photo" in none.reason
    turned = PhotoCamera.around(CENTRE, 0.8, 30, 0, fov=35.0, size=(640, 480))
    same = [photo, shapefit.Photo("turned", (640, 480), turned.project(marks.points), turned, True)]
    options = shapefit.FitOptions(expressions=False, iris=False, contour=False, outline=False, trust=False)
    kept = shapefit.fit_person(same, model, marks, made, space, options)  # the avatar itself: nothing to gain
    assert not kept.usable and "no better than the avatar's own" in kept.reason
    assert "keeps the avatar's shape" in kept.summary()


def test_the_size_comes_from_the_irises_and_is_held(made, setup):
    """Photos of the avatar whose irises say the person's head is 10 % smaller than the avatar's: the landmarks
    cannot tell sizes apart (a larger head farther away looks the same), the irises and the avatar's size as the
    prior can; the fitted head is uniformly scaled to the size between them (its shape the avatar's), and without
    irises it keeps the avatar's size."""
    model, space, _landmarks, marks = setup
    rims = list(cf.RIGHT_IRIS[1:] + cf.LEFT_IRIS[1:])
    reliability = marks.reliability.copy()
    reliability[rims] = 0.0  # (stage 0 has no rims on this head: only the photos' count, as the ruler)
    triangles = np.asarray(model.primitives[1].indices, np.int64).reshape(-1, 3)
    reliability[made.fade[triangles[marks.triangle]].min(axis=1) < 0.999] = 0.0  # (nor points where the neck
    marks = dataclasses.replace(marks, reliability=reliability)  # fades: the head does not scale there; no face does)
    pupils = np.linalg.norm(marks.points[cf.LEFT_IRIS[0]] - marks.points[cf.RIGHT_IRIS[0]])
    photos = []
    for k, (yaw, pitch) in enumerate(((0, 0), (0, 10), (0, -10), (35, 0), (-35, 0))):
        camera = PhotoCamera.around(CENTRE, 0.8, yaw, pitch, fov=35.0, size=(640, 480))
        xy = camera.project(marks.points)
        radius = shapefit.IRIS_RADIUS / (0.9 * pupils) * np.linalg.norm(xy[cf.LEFT_IRIS[0]] - xy[cf.RIGHT_IRIS[0]])
        for iris in (cf.RIGHT_IRIS, cf.LEFT_IRIS):  # the rim: across, up, across, down
            xy[list(iris[1:])] = xy[iris[0]] + radius * np.array([[1, 0], [0, -1], [-1, 0], [0, 1]])
        photos.append(shapefit.Photo(f"photo {k}", (640, 480), xy, camera, True))
    options = {"expressions": False, "contour": False, "outline": False, "trust": False}
    eyes = [made.eye == 1, made.eye == 2]
    head = np.asarray(model.primitives[1].positions, np.float64)

    def eye_distance(offsets):
        moved = head + offsets
        return float(np.linalg.norm(moved[eyes[1]].mean(axis=0) - moved[eyes[0]].mean(axis=0)))

    own = eye_distance(0.0)
    fit = shapefit.fit_person(photos, model, marks, made, space, shapefit.FitOptions(**options))
    assert fit.scale_from.startswith("the irises (3 photos"), fit.scale_from
    spread = math.hypot(shapefit.IRIS_SPREAD / shapefit.IRIS_RADIUS, shapefit.IRIS_NOISE / math.sqrt(3))
    weight, prior = spread ** -2, shapefit.SIZE_SPREAD ** -2
    expected = math.exp(weight * math.log(0.9) / (weight + prior))  # about 0.95: about as sure as each other
    assert fit.scale == pytest.approx(expected, abs=0.003)
    assert eye_distance(fit.offsets) == pytest.approx(own * expected, abs=3e-4)
    assert np.abs(fit.beta - made.alpha).max() < 0.1  # the shape is the avatar's
    assert fit.iris_radius == pytest.approx(shapefit.IRIS_RADIUS * expected / 0.9, rel=0.02)
    plain = shapefit.fit_person(photos, model, marks, made, space, shapefit.FitOptions(iris=False, **options))
    assert plain.scale_from == "the avatar's own size" and math.isnan(plain.iris_radius)
    assert plain.scale == pytest.approx(1.0, abs=0.003)
    assert eye_distance(plain.offsets) == pytest.approx(own, abs=3e-4)


def test_the_free_form_fallback_without_the_space(setup, monkeypatch):
    """Without an identity space the free-form correction alone moves the face where the photos agree, capped."""
    model, _space, _landmarks, marks = setup
    hs = s0.make_head_space(model, marks, None)
    assert hs.alpha.size == 0 and (hs.triangle < 0).all() and list(hs.others) == [2]
    head = np.asarray(model.primitives[1].positions, np.float64)
    tri = np.asarray(model.primitives[1].indices, np.int64).reshape(-1, 3)[marks.triangle]
    cheek = np.exp(-np.sum((head - [0.045, 1.58, 0.07]) ** 2, axis=1) / 0.02 ** 2)
    shift = cheek[:, None] * [0.003, 0.0, 0.0]  # the left cheek 3 mm fuller (no turn of the head explains that)
    points = np.einsum("lc,lcd->ld", marks.barycentric, (head + shift)[tri])
    photos = [shapefit.Photo(f"p{yaw}", (640, 480), PhotoCamera.around(CENTRE, 0.8, yaw, 0, fov=35.0,
                                                                        size=(640, 480)).project(points),
                             PhotoCamera.around(CENTRE, 0.8, yaw, 0, fov=35.0, size=(640, 480)), True)
              for yaw in (0, 30, -30)]
    fit = shapefit.fit_person(photos, model, marks, hs, None, shapefit.FitOptions(iris=False, contour=False,
                                                                                   trust=False))
    assert fit.method == "free-form" and fit.usable
    full = (cheek > 0.5) & (hs.parts == s0.SKIN)
    assert np.median(fit.offsets[full, 0]) > 0.0003  # fuller (smoothly: a part of the 3 mm)
    far = (head[:, 0] < -0.02) & (hs.parts == s0.SKIN)
    assert np.abs(fit.offsets[far]).max() < 0.5 * np.abs(fit.offsets[full]).max()  # the other side less
    assert np.linalg.norm(fit.offsets, axis=1).max() <= shapefit.FREE_CAP + 1e-9


def _shape(made, model, setup, beta=(1.0, -1.0, 1.0)):
    space = setup[1]
    modes = made.modes(space).astype(np.float64)
    offsets = np.einsum("k,knd->nd", np.asarray(beta) - made.alpha, modes)
    return rig.make_shape(model, made, offsets)


def test_the_fitted_head_on_the_rig(made, setup):
    """The topology, UVs and skin weights are kept (shared); the eyeballs move rigidly with their sockets; the face
    bones' rest positions move and the inverse binds follow, so the rest pose shows the new head; normals turn with
    the surface and tangents are made again; the hair follows the scalp; the neck's edge stays."""
    model = setup[0]
    shape = _shape(made, model, setup)
    fitted = rig.apply_shape(model, shape)
    old, new = model.primitives[1], fitted.primitives[1]
    assert new is not old and fitted.primitives[0] is model.primitives[0]  # the body is shared
    assert new.indices is old.indices and new.uvs is old.uvs and new.joints is old.joints
    assert new.weights is old.weights and new.texture is old.texture
    assert np.allclose(new.positions, old.positions + shape.offsets, atol=1e-6)
    eyes = made.eye == 1
    moved = np.asarray(new.positions)[eyes] - np.asarray(old.positions)[eyes]
    centre_move = moved.mean(axis=0)
    assert np.linalg.norm(centre_move) > 1e-4  # the eyeball went with its socket ...
    eye = model.skeleton.index("Bip01 REye")
    shift = fitted.skeleton.rest_global[eye, :3, 3] - model.skeleton.rest_global[eye, :3, 3]
    assert np.allclose(shift, shape.bones["Bip01 REye"], atol=1e-6) and np.linalg.norm(shift) > 1e-4  # ... its bone
    nose = model.skeleton.index("Bip01 MNose")
    assert np.linalg.norm(fitted.skeleton.rest_global[nose, :3, 3] - model.skeleton.rest_global[nose, :3, 3]) > 1e-5
    palette = fitted.skinning_matrices(fitted.skeleton.rest_global)  # the rest pose shows the fitted head
    from bifold_demo.characters.character import skin_vertices

    assert np.allclose(skin_vertices(fitted, palette)[1], new.positions, atol=1e-5)
    from bifold_demo.characters.tangents import vertex_tangents

    assert np.allclose(new.tangents, vertex_tangents(new.positions, new.normals, new.uvs, new.indices), atol=1e-5)
    assert not np.allclose(new.tangents, old.tangents, atol=1e-4)
    geometric = mesh.vertex_normals(new.positions, new.indices, mesh.weld(new.positions)[0])
    before = mesh.vertex_normals(old.positions, old.indices, mesh.weld(old.positions)[0])
    skin = made.parts == s0.SKIN
    turned = skin & (np.einsum("nk,nk->n", geometric, before) < np.cos(np.radians(2.0)))  # the surface turned there
    assert turned.sum() > 20
    assert np.median(np.einsum("nk,nk->n", geometric[turned], new.normals[turned])) > \
        np.median(np.einsum("nk,nk->n", geometric[turned], np.asarray(old.normals)[turned])) + 1e-4
    assert "s_opacity" in shape.others and np.abs(shape.others["s_opacity"]).max() > 1e-5  # the hair moved along
    assert np.abs(shape.offsets[made.fade == 0]).max() < 1e-9  # the neck's edge stays where the body is
    check = rig.rig_checks(model, fitted, made)
    assert "gap" not in " ".join(check.problems)


def test_the_shape_file_round_trips_and_refuses_what_is_not_one(made, setup, tmp_path):
    model = setup[0]
    shape = _shape(made, model, setup)
    path = shape.save(tmp_path / "person")
    assert path.name == rig.FILE and path.stat().st_size < 60_000
    again = rig.Shape.load(tmp_path / "person")
    assert np.array_equal(again.offsets, shape.offsets) and again.others.keys() == shape.others.keys()
    assert again.bones.keys() == shape.bones.keys() and again.base == rig.base_key(model)
    assert not again.problems(model)
    values = dict(np.load(path, allow_pickle=False))
    values["offsets"] = np.asarray(shape.offsets, np.float64).astype(object)  # a pickled array, else a good shape
    np.savez(tmp_path / "bad.npz", **values)
    with pytest.raises(ValueError):
        rig.Shape.load(tmp_path / "bad.npz")  # (an object array needs pickles: never loaded)
    other = synthetic_avatar()
    other.name = "Another"
    assert any("made for Synthetic" in p for p in again.problems(other))
    with pytest.raises(ValueError, match="does not fit"):
        rig.apply_shape(other, again)
    moved = synthetic_avatar()
    moved.primitives[1].positions = moved.primitives[1].positions + np.float32(0.001)  # prepared again since
    assert any("another version" in p for p in again.problems(moved))


def test_folds_are_smoothed_away(made, setup):
    model = setup[0]
    head = model.primitives[1]
    points = np.asarray(head.positions, np.float64)
    offsets = np.zeros_like(points)
    tri = np.asarray(head.indices, np.int64).reshape(-1, 3)
    skin_tri = tri[(made.parts[tri] == s0.SKIN).all(axis=1)]
    victim = skin_tri[len(skin_tri) // 3]
    offsets[victim[0]] = (points[victim[1]] - points[victim[0]]) * 2.2  # pushed across its triangle: a fold
    skin = made.parts == s0.SKIN
    out = rig.untangle(points, head.indices, skin, offsets)
    moved = points + out
    before = np.cross(points[skin_tri[:, 1]] - points[skin_tri[:, 0]], points[skin_tri[:, 2]] - points[skin_tri[:, 0]])
    after = np.cross(moved[skin_tri[:, 1]] - moved[skin_tri[:, 0]], moved[skin_tri[:, 2]] - moved[skin_tri[:, 0]])
    assert (np.einsum("tk,tk->t", before, after) > 0).all()
    far = np.linalg.norm(points - points[victim[0]], axis=1) > 0.03
    assert np.abs(out[far]).max() == 0.0  # elsewhere nothing changes


def test_mediapipes_blendshapes_name_icts_expressions():
    names = ["jawOpen", "mouthSmile_L", "mouthSmile_R", "browInnerUp_L", "cheekPuff_R", "eyeBlink_L"]
    shapes = {"jawOpen": 0.5, "mouthSmileLeft": 0.7, "mouthSmileRight": 0.2, "browInnerUp": 0.3, "cheekPuff": 0.1,
              "eyeBlinkRight": 0.9}
    assert shapefit._mediapipe_expressions(shapes, names).tolist() == [0.5, 0.7, 0.2, 0.3, 0.1, 0.0]
