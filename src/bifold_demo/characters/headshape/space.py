"""The identity space: ICT-FaceKit Light's neutral head with its 100 identity modes and 53 expressions.

**The data.** ICT-FaceKit (USC Institute for Creative Technologies, MIT licence; ``ict_files.json`` names the
repository and the commit) is a morphable model of the whole head built from light-stage scans: a neutral mesh of
26,719 vertices (the face, the head and neck, the mouth socket, the eyes' sockets, gums, tongue, teeth, eyeballs,
lashes) and the same mesh deformed by each of 100 identity modes (principal components with unit variance: a person
is the neutral mesh plus ``beta @ modes`` with ``beta`` about standard normal) and by 53 expression shapes named like
Apple's ARKit blendshapes (``jawOpen``, ``mouthSmile_L``, ...; ``_L`` is the person's left). Each is a 2.6 MB OBJ
file, 395 MB in all.

**Fetching it** (:func:`fetch`, ``bifold-characters identity-space``): every file is downloaded from the pinned
commit, checked against its SHA-256 (``ict_files.json``), and the parts the shape fit uses are kept in one float32
``.npz`` (:data:`FILE`, 21 MB, no pickles) with the licence beside it; the OBJ files are deleted. Kept: the face
(vertices 0-9408) and the head and neck (9409-11247) of the neutral mesh, their triangles (the quads split in two),
and the offsets of those vertices in every identity mode and expression, in metres (ICT's centimetres times 0.01).
ICT's axes are the avatars' character space: +Y up, the face looking along +Z, +X the person's left.

**Where it is kept.** ``$BIFOLD_IDENTITY_SPACE`` (a folder) when set, else ``identity/`` in the prepared characters'
folder (:func:`~bifold_demo.characters.library.default_folder`), downloaded like the avatars rather than committed.
Without it the shape fit falls back to a free-form deformation (:mod:`.fit`).

**Landmarks on ICT's head.** The shape fit needs MediaPipe's 478 landmarks on ICT's neutral face, as stage 0 of the
head textures has them on each avatar (:mod:`~bifold_demo.characters.headtex.landmarks`). They were found the same
way, once, on shaded renders of the neutral head (:func:`find_landmarks`), and ship with the code
(``ict_landmarks.json``: per landmark three ICT vertices, the barycentric weights and how reliable it was); ICT's own
68 Multi-PIE landmarks (:data:`MULTI_PIE`) check them.
"""

from __future__ import annotations

import json
import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np

FILE = "ict-facekit-light.npz"
LICENSE_FILE = "LICENSE-ICT-FaceKit.txt"
FORMAT = 1
ENV = "BIFOLD_IDENTITY_SPACE"
VERTICES = 11248  # kept: the face (0-9408) and the head and neck (9409-11247)
FACE_VERTICES = 9409
NARROW_FACE = 6706  # ICT's "narrow face area": vertices 0-6705
ALL_VERTICES = 26719  # vertices in every ICT OBJ file
CM = 0.01  # metres per ICT unit
RAW = "https://raw.githubusercontent.com/ICT-VGL/ICT-FaceKit/{commit}/{path}"
# ICT's 68 Multi-PIE landmarks (vertex indices, 0-based; the ICT-FaceKit README), in the usual order: jaw 0-16,
# brows 17-26, nose 27-35, eyes 36-47, mouth 48-67
MULTI_PIE = (1225, 1888, 1052, 367, 1719, 1722, 2199, 1447, 966, 3661, 4390, 3927, 3924, 2608, 3272, 4088, 3443, 268,
             493, 1914, 2044, 1401, 3615, 4240, 4114, 2734, 2509, 978, 4527, 4942, 4857, 1140, 2075, 1147, 4269, 3360,
             1507, 1542, 1537, 1528, 1518, 1511, 3742, 3751, 3756, 3721, 3725, 3732, 5708, 5695, 2081, 0, 4275, 6200,
             6213, 6346, 6461, 5518, 5957, 5841, 5702, 5711, 5533, 6216, 6207, 6470, 5517, 5966)
_HERE = Path(__file__).resolve().parent


def files() -> dict:
    """The pinned files: ``{"repository", "commit", "licence", "files": [[path, sha256, size], ...]}``."""
    return json.loads((_HERE / "ict_files.json").read_text(encoding="utf-8"))


def expression_names() -> tuple[str, ...]:
    """ICT's 53 expression shapes, in the order of the pinned files (and of the stored space)."""
    return tuple(Path(p).stem for p, _sha, _size in files()["files"]
                 if p.startswith("FaceXModel/") and not Path(p).stem.startswith(("identity", "generic_")))


def folder(path=None) -> Path:
    """Where the identity space is kept: ``path``, else ``$BIFOLD_IDENTITY_SPACE``, else ``identity/`` in the
    prepared characters' folder."""
    if path is not None:
        return Path(path).expanduser()
    if os.environ.get(ENV):
        return Path(os.environ[ENV]).expanduser()
    from ..library import default_folder

    return default_folder() / "identity"


def space_file(path=None) -> Path:
    """The stored space's file (in :func:`folder`)."""
    return folder(path) / FILE


def available(path=None) -> bool:
    return space_file(path).is_file()


FETCH_HINT = "bifold-characters identity-space"


@dataclass
class IdentitySpace:
    """ICT-FaceKit's head (see the module docs), in metres. ``identity``: ``(K, N, 3)`` offsets per unit of each
    identity coefficient; ``expressions``: ``(E, N, 3)`` offsets of each expression at full strength."""

    neutral: np.ndarray  # (N, 3) float32
    identity: np.ndarray  # (K, N, 3) float32
    expressions: np.ndarray  # (E, N, 3) float32
    expression_names: tuple
    triangles: np.ndarray  # (T, 3) int32: the face and the head and neck
    commit: str = ""
    license: str = ""

    @property
    def vertices(self) -> int:
        return len(self.neutral)

    @property
    def modes(self) -> int:
        return len(self.identity)

    def shape(self, beta=None, expression=None) -> np.ndarray:
        """``(N, 3)`` float64: the neutral head plus ``beta`` times the identity modes (the first ``len(beta)``)
        plus ``expression`` (``(E,)`` weights, or ``{name: weight}``) times the expressions."""
        out = self.neutral.astype(np.float64)
        if beta is not None:
            beta = np.asarray(beta, np.float64)
            out = out + np.einsum("k,knd->nd", beta, self.identity[:len(beta)])
        if expression is not None:
            if isinstance(expression, dict):
                weights = np.zeros(len(self.expression_names))
                for name, value in expression.items():
                    weights[self.expression_names.index(name)] = value
                expression = weights
            out = out + np.einsum("e,end->nd", np.asarray(expression, np.float64), self.expressions)
        return out

    def corners(self, points=None) -> np.ndarray:
        """``(T, 3, 3)``: the triangles' corners (of ``points``, default the neutral head)."""
        points = self.neutral if points is None else points
        return np.asarray(points, np.float64)[self.triangles]

    @property
    def face(self) -> np.ndarray:
        """bool per vertex: ICT's face area (the rest is the head and neck)."""
        mask = np.zeros(self.vertices, bool)
        mask[:FACE_VERTICES] = True
        return mask

    def save(self, path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp.npz")
        np.savez(tmp, format=np.array(FORMAT), neutral=self.neutral.astype(np.float32),
                 identity=self.identity.astype(np.float32), expressions=self.expressions.astype(np.float32),
                 expression_names=np.array(self.expression_names, dtype=str),
                 triangles=self.triangles.astype(np.int32), commit=np.array(self.commit),
                 license=np.array(self.license))
        os.replace(tmp, path)
        return path

    @classmethod
    def read(cls, path) -> IdentitySpace:
        """The space stored at ``path``. Raises ``OSError`` or ``ValueError`` for a file that is not one."""
        with np.load(path, allow_pickle=False) as data:
            if int(data["format"]) != FORMAT:
                raise ValueError(f"{path}: an identity space of format {int(data['format'])}, not {FORMAT}")
            space = cls(np.asarray(data["neutral"], np.float32), np.asarray(data["identity"], np.float32),
                        np.asarray(data["expressions"], np.float32), tuple(str(n) for n in data["expression_names"]),
                        np.asarray(data["triangles"], np.int32), str(data["commit"]), str(data["license"]))
        n = space.vertices
        if space.identity.shape[1:] != (n, 3) or space.expressions.shape[1:] != (n, 3) or \
                len(space.expression_names) != len(space.expressions) or space.triangles.max() >= n:
            raise ValueError(f"{path}: the identity space's arrays do not fit together")
        return space


_loaded: dict[str, tuple] = {}
_load_lock = threading.Lock()


def load(path=None) -> IdentitySpace:
    """The identity space from :func:`folder` (read once per process while the file is unchanged). Raises
    ``FileNotFoundError`` naming the command that fetches it."""
    file = space_file(path)
    try:
        stamp = file.stat().st_mtime_ns
    except OSError:
        raise FileNotFoundError(f"no identity space at {file} (run: {FETCH_HINT}"
                                + (f" --dest {folder(path).parent}" if path is not None else "") + ")") from None
    key = str(file.resolve())
    with _load_lock:
        hit = _loaded.get(key)
        if hit is not None and hit[0] == stamp:
            return hit[1]
        space = IdentitySpace.read(file)
        _loaded[key] = (stamp, space)
        return space


# -- fetching and converting ---------------------------------------------------------------------------------------


def obj_vertices(data: bytes) -> np.ndarray:
    """``(V, 3)`` float64: the ``v`` lines of an OBJ file."""
    rows = [line[2:] for line in data.split(b"\n") if line.startswith(b"v ")]
    values = np.array(b" ".join(rows).split(), dtype=np.float64)
    if len(values) % 3:
        raise ValueError("an OBJ vertex without three coordinates")
    return values.reshape(-1, 3)


def obj_faces(data: bytes) -> list[list[int]]:
    """The ``f`` lines of an OBJ file: 0-based vertex indices per polygon (texture and normal indices dropped)."""
    out = []
    for line in data.split(b"\n"):
        if line.startswith(b"f "):
            out.append([int(item.split(b"/")[0]) - 1 for item in line.split()[1:]])
    return out


def triangulate(polygons, vertices: int) -> np.ndarray:
    """``(T, 3)`` int32: the polygons whose vertices are all below ``vertices``, as fans of triangles (a quad
    ``a b c d`` gives ``a b c`` and ``a c d``; the order, counter-clockwise from outside, is kept)."""
    out = []
    for poly in polygons:
        if len(poly) >= 3 and max(poly) < vertices:
            out += [(poly[0], poly[k], poly[k + 1]) for k in range(1, len(poly) - 1)]
    return np.asarray(out, np.int32)


def convert(raw, target=None, progress=print) -> IdentitySpace:
    """The identity space from ICT's files in the folder ``raw`` (its ``LICENSE`` and the ``FaceXModel`` OBJs, by
    their file names), written to ``target`` when given."""
    raw = Path(raw)
    table = files()
    neutral_all = obj_vertices((raw / "generic_neutral_mesh.obj").read_bytes())
    if len(neutral_all) != ALL_VERTICES:
        raise ValueError(f"the neutral mesh has {len(neutral_all)} vertices, not {ALL_VERTICES}")
    polygons = obj_faces((raw / "generic_neutral_mesh.obj").read_bytes())
    triangles = triangulate(polygons, VERTICES)

    def offsets(name):
        points = obj_vertices((raw / f"{name}.obj").read_bytes())
        if len(points) != ALL_VERTICES:
            raise ValueError(f"{name}.obj has {len(points)} vertices, not {ALL_VERTICES}")
        return ((points[:VERTICES] - neutral_all[:VERTICES]) * CM).astype(np.float32)

    modes = sorted(Path(rel).stem for rel, _sha, _size in table["files"] if Path(rel).stem.startswith("identity"))
    identity = np.stack([offsets(name) for name in modes])
    if progress:
        progress(f"[identity] read the {len(modes)} identity modes")
    names = expression_names()
    expressions = np.stack([offsets(name) for name in names])
    if progress:
        progress(f"[identity] read the {len(names)} expressions")
    license_text = (raw / "LICENSE").read_text(encoding="utf-8") if (raw / "LICENSE").is_file() else ""
    space = IdentitySpace((neutral_all[:VERTICES] * CM).astype(np.float32), identity, expressions, names, triangles,
                          table["commit"], license_text)
    if target is not None:
        space.save(target)
    return space


def fetch(path=None, *, progress=print, jobs: int = 8, keep_raw: bool = False, force: bool = False) -> Path:
    """Download ICT-FaceKit Light's files (395 MB, each checked by its SHA-256), convert them to the identity space
    (:data:`FILE`, 21 MB) in :func:`folder` (``path``) with ICT's licence beside it, and delete the downloaded files
    (unless ``keep_raw``). Files already downloaded and intact are not downloaded again. Returns the space's path;
    raises :class:`~bifold_demo.downloads.DownloadError` when a file cannot be had."""
    from ... import downloads

    dest = folder(path)
    target = dest / FILE
    if target.is_file() and not force:
        if progress:
            progress(f"[identity] {target} is there already")
        return target
    table = files()
    raw = dest / "raw"
    wanted = [downloads.ModelFile(Path(rel).name, RAW.format(commit=table["commit"], path=rel), sha, size)
              for rel, sha, size in table["files"]]
    total = sum(f.size for f in wanted)
    if progress:
        progress(f"[identity] ICT-FaceKit Light ({table['licence']} licence, {table['repository']}): "
                 f"{len(wanted)} files, {total / 1e6:.0f} MB")
    errors = []

    def one(model):
        try:
            downloads.ensure(model, raw, progress=None, tag="identity")
        except OSError as exc:
            errors.append(f"{model.name}: {exc}")

    done = [0]
    lock = threading.Lock()

    def counted(model):
        one(model)
        with lock:
            done[0] += 1
            if progress and (done[0] % 20 == 0 or done[0] == len(wanted)):
                progress(f"[identity] {done[0]}/{len(wanted)} files")

    with ThreadPoolExecutor(max(1, jobs)) as pool:
        list(pool.map(counted, wanted))
    if errors:
        raise downloads.DownloadError(f"cannot download {len(errors)} of ICT-FaceKit's files: " + "; ".join(errors[:3]),
                                      {e.split(':')[0]: [e] for e in errors})
    convert(raw, target, progress=progress)
    (dest / LICENSE_FILE).write_text((raw / "LICENSE").read_text(encoding="utf-8"), encoding="utf-8")
    if not keep_raw:
        shutil.rmtree(raw, ignore_errors=True)
    if progress:
        progress(f"[identity] {target} ({target.stat().st_size / 1e6:.0f} MB)")
    return target


# -- MediaPipe's landmarks on ICT's head ---------------------------------------------------------------------------


@dataclass
class SpaceLandmarks:
    """MediaPipe's 478 landmarks on ICT's neutral head: each a weighted sum of three vertices."""

    vertices: np.ndarray  # (478, 3) int
    weights: np.ndarray  # (478, 3) barycentric
    reliability: np.ndarray  # (478,) 0-1 (0: not found)

    def points(self, shape: np.ndarray) -> np.ndarray:
        """``(478, 3)``: the landmarks on the head ``shape`` (``(N, 3)``, e.g. :meth:`IdentitySpace.shape`)."""
        return np.einsum("lc,lcd->ld", self.weights, np.asarray(shape, np.float64)[self.vertices])

    def offsets(self, modes: np.ndarray) -> np.ndarray:
        """``(K, 478, 3)``: the landmarks' offsets in each of ``modes`` (``(K, N, 3)``)."""
        return np.einsum("lc,klcd->kld", self.weights, np.asarray(modes)[:, self.vertices])

    def save(self, path) -> None:
        data = {"note": "MediaPipe's 478 face landmarks on ICT-FaceKit's neutral head (headshape.space."
                        "find_landmarks): three ICT vertices, their barycentric weights and the reliability",
                "vertices": self.vertices.tolist(), "weights": np.round(self.weights, 5).tolist(),
                "reliability": np.round(self.reliability, 3).tolist()}
        Path(path).write_text(json.dumps(data, separators=(",", ":")) + "\n", encoding="utf-8")

    @classmethod
    def read(cls, path=None) -> SpaceLandmarks:
        data = json.loads(Path(path or _HERE / "ict_landmarks.json").read_text(encoding="utf-8"))
        return cls(np.asarray(data["vertices"], np.int64), np.asarray(data["weights"], np.float64),
                   np.asarray(data["reliability"], np.float64))


def space_landmarks() -> SpaceLandmarks:
    """The landmarks that ship with the code (see the module docs)."""
    return SpaceLandmarks.read()


RENDER_SIZE = 640
EYEBALLS = ((21451, 23021), (23021, 24591))  # ICT's left and right eyeball vertices (README): aim the renders
RENDER_FOV = 26.0  # degrees
RENDER_DISTANCE = 0.62  # metres from the face's centre
RENDER_VIEWS = ((0.0, 0.0), (22.0, 0.0), (-22.0, 0.0), (0.0, 12.0), (0.0, -12.0))
LIGHT = np.array([0.25, 0.45, 1.0])  # towards the light, ICT's space


def shaded_render(points, triangles, camera, light=LIGHT, ambient: float = 0.3, albedo=None) -> np.ndarray:
    """A grey BGR picture of the mesh (``points``, ``triangles``) seen by ``camera`` (a
    :class:`~bifold_demo.characters.headtex.camera.PhotoCamera`): Lambert shading with the vertex normals
    interpolated, times ``albedo`` (per vertex, 0-1; default 0.8), on a mid-grey background (numpy only)."""
    from ..headtex import raster

    points = np.asarray(points, np.float64)
    corners = points[triangles]
    r = raster.rasterise(camera, corners, cull=True, barycentric=True)
    normals = np.zeros_like(points)
    face_n = np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0])
    for k in range(3):
        np.add.at(normals, triangles[:, k], face_n)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    light = np.asarray(light, np.float64) / np.linalg.norm(light)
    shade = np.clip(normals @ light, 0.0, 1.0)
    h, w = r.triangle.shape
    image = np.full((h, w), 0.45)
    hit = r.triangle >= 0
    tri = triangles[r.triangle[hit]]
    albedo = np.full(len(points), 0.8) if albedo is None else np.asarray(albedo, np.float64)
    lit = (ambient + (1.0 - ambient) * shade) * albedo
    image[hit] = np.einsum("nc,nc->n", r.barycentric[hit].astype(np.float64), lit[tri])
    grey = np.clip(np.power(image, 1 / 2.2) * 255.0, 0, 255).astype(np.uint8)
    return np.repeat(grey[..., None], 3, axis=2)


def find_landmarks(space_obj_points, triangles, landmarker, face_triangles=None, albedo=None) -> SpaceLandmarks:
    """MediaPipe's landmarks on ICT's neutral head (how ``ict_landmarks.json`` was made): the head (ICT's mesh with
    the eyeballs, ``space_obj_points`` ``(V, 3)`` metres and ``triangles``; ``albedo`` per vertex, the irises dark)
    shaded and rendered from five views as stage 0 renders an avatar, ``landmarker`` (a
    :class:`~bifold_demo.characters.headtex.detect.Landmarker`) run on each, the landmarks lifted onto the mesh and
    combined like stage 0's (:func:`~bifold_demo.characters.headtex.landmarks.combine`), then snapped to
    ``face_triangles`` (default: those of the kept vertices)."""
    from ..headtex import landmarks as stage0
    from ..headtex import raycast
    from ..headtex.camera import PhotoCamera

    points = np.asarray(space_obj_points, np.float64)
    corners = points[triangles]
    normals = raycast.triangle_normals(corners)
    eyes = np.array([points[a:b].mean(axis=0) for a, b in EYEBALLS])
    target = np.array([0.0, eyes[:, 1].mean() - 0.025, eyes[:, 2].mean()])
    candidates, facing = [], []
    for yaw, pitch in RENDER_VIEWS:
        camera = PhotoCamera.around(target, RENDER_DISTANCE, yaw, pitch, fov=RENDER_FOV,
                                    size=(RENDER_SIZE, RENDER_SIZE))
        image = shaded_render(points, triangles, camera, albedo=albedo)
        faces = landmarker.detect(image)
        if not faces:
            continue
        xyz, _tri, _bary, face = stage0.lift(faces[0].points, camera, corners, normals)
        candidates.append(xyz)
        facing.append(face)
    if not candidates:
        raise ValueError("MediaPipe finds no face on the renders of ICT's head")
    combined, reliability = stage0.combine(np.array(candidates), np.array(facing))
    keep = face_triangles if face_triangles is not None else triangles[(triangles < VERTICES).all(axis=1)]
    _d, tri, bary = raycast.closest(combined, points[keep])
    return SpaceLandmarks(keep[tri].astype(np.int64), bary, reliability)


def distances_mm(a, b) -> np.ndarray:
    """Distances between corresponding points in millimetres."""
    return np.linalg.norm(np.asarray(a, np.float64) - np.asarray(b, np.float64), axis=-1) * 1000.0


__all__ = [
    "FETCH_HINT",
    "FILE",
    "MULTI_PIE",
    "IdentitySpace",
    "SpaceLandmarks",
    "available",
    "convert",
    "expression_names",
    "fetch",
    "find_landmarks",
    "folder",
    "load",
    "space_file",
    "space_landmarks",
]
