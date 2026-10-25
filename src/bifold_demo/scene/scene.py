"""The scene: a camera, a light and a set of named, textured objects."""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from .animation import resolve
from .geometry import Mesh, box_mesh, cylinder_mesh, plane_mesh, quad_mesh, sphere_mesh
from .physics import Body
from .textures import IMAGE_EXTENSIONS, VIDEO_EXTENSIONS, Texture, as_texture, parse_color
from .transforms import euler_rotation, look_at, perspective, rotation_y, scaling, translation, vec3

MAX_PHYSICS_STEP = 1.0 / 120.0


class SceneObject:
    """A textured mesh placed in the scene.

    Every property below can be a constant or a function of simulation time ``t``
    (see :mod:`bifold_demo.scene.animation`).

    Args:
        mesh: the geometry (the shape subclasses build it for you).
        name: unique name used to look the object up (``scene["name"]``) and to
            match uploaded texture files; generated when omitted.
        texture: image path, video path, array, PIL image, colour or :class:`Texture`.
        color: ``(r, g, b)`` tint multiplied with the texture (white = unchanged).
        position: ``(x, y, z)`` of the object's centre. Y points up, the default camera looks towards -Z.
        rotation: Euler angles ``(x, y, z)`` in degrees, applied in X, Y, Z order.
        scale: a number or ``(sx, sy, sz)``.
        lit: shade with the scene light; ``False`` shows the texture exactly as is.
        visible: whether to draw the object.
        face_camera: keep turning around the vertical axis to face the camera (billboard).
        body: a :class:`Body` for simple physics (then the body owns position and rotation).
        on_update: ``fn(obj, t, dt)`` called every frame before the object is evaluated.
    """

    _geometry_params: tuple[str, ...] = ()

    def __init__(
        self,
        mesh: Mesh | None = None,
        *,
        name: str | None = None,
        texture=None,
        color=(255, 255, 255),
        position=(0.0, 0.0, 0.0),
        rotation=(0.0, 0.0, 0.0),
        scale=1.0,
        lit: bool = True,
        visible=True,
        face_camera: bool = False,
        body: Body | None = None,
        on_update=None,
    ):
        self._mesh = mesh
        self.name = name
        self.texture = texture
        self.color = color
        self.position = position
        self.rotation = rotation
        self.scale = scale
        self.lit = lit
        self.visible = visible
        self.face_camera = face_camera
        self.on_update = on_update
        # state evaluated for the current frame (read by the renderer)
        self.current_position = vec3(resolve(position, 0.0), "position")
        self.current_rotation = vec3(resolve(rotation, 0.0), "rotation")
        self.current_scale = vec3(resolve(scale, 0.0), "scale")
        self.current_color = parse_color(resolve(color, 0.0))
        self.current_visible = bool(resolve(visible, 0.0))
        self.current_texture: Texture | None = None
        self._texture_error: str | None = None
        self.body = body

    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if name in self._geometry_params:
            super().__setattr__("_mesh", None)  # rebuild lazily with the new dimensions

    # -- properties -------------------------------------------------------------------

    @property
    def mesh(self) -> Mesh:
        if self._mesh is None:
            self._mesh = self._build_mesh()
        return self._mesh

    @mesh.setter
    def mesh(self, mesh: Mesh) -> None:
        self._mesh = mesh

    def _build_mesh(self) -> Mesh:
        raise ValueError(f"{type(self).__name__} needs a mesh")

    @property
    def texture(self):
        return self._texture

    @texture.setter
    def texture(self, spec) -> None:
        if not callable(spec):
            as_texture(spec)  # fail early on missing files or unsupported values
        self._texture = spec
        self._texture_error = None

    def set_texture(self, texture) -> "SceneObject":
        """Assign a texture and return the object, so calls can be chained."""
        self.texture = texture
        return self

    @property
    def body(self) -> Body | None:
        return self._body

    @body.setter
    def body(self, body: Body | None) -> None:
        self._body = body
        if body is not None:
            body.reset(self.current_position, self.current_rotation)

    # -- per-frame evaluation -----------------------------------------------------------

    def update(self, t: float) -> None:
        """Evaluate all properties at simulation time ``t``."""
        self.current_visible = bool(resolve(self.visible, t))
        self.current_color = parse_color(resolve(self.color, t))
        self.current_scale = vec3(resolve(self.scale, t), "scale")
        if self._body is None:
            self.current_position = vec3(resolve(self.position, t), "position")
            self.current_rotation = vec3(resolve(self.rotation, t), "rotation")
        else:
            self.current_position = self._body.position.copy()
            self.current_rotation = self._body.rotation.copy()
        try:
            texture = as_texture(resolve(self._texture, t))
            self.current_texture = texture.at_time(t) if texture is not None else None
        except Exception as exc:  # noqa: BLE001 - keep the last good texture instead of stopping the demo
            message = f"{type(exc).__name__}: {exc}"
            if message != self._texture_error:
                print(f"[scene] texture of {self.name!r} failed ({message}); keeping the previous one")
                self._texture_error = message

    def model_matrix(self, camera_position=None) -> np.ndarray:
        m = translation(self.current_position)
        if self.face_camera and camera_position is not None:
            d = np.asarray(camera_position) - self.current_position
            m = m @ rotation_y(math.degrees(math.atan2(d[0], d[2])))
        return m @ euler_rotation(self.current_rotation) @ scaling(self.current_scale)

    def extents(self, rotation=None, scale=None) -> tuple[np.ndarray, np.ndarray]:
        """Bounding-box offsets ``(low, high)`` of the rotated, scaled mesh relative to its centre."""
        rotation = self.current_rotation if rotation is None else rotation
        scale = self.current_scale if scale is None else vec3(scale)
        linear = euler_rotation(rotation)[:3, :3] * scale[None, :]
        points = self.mesh.positions.astype(np.float64) @ linear.T
        return points.min(axis=0), points.max(axis=0)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class Box(SceneObject):
    """A box; ``size`` is a number (cube) or ``(width, height, depth)``. Every face shows the whole texture."""

    _geometry_params = ("size",)

    def __init__(self, size=1.0, **kwargs):
        self.size = size
        super().__init__(**kwargs)

    def _build_mesh(self) -> Mesh:
        return box_mesh(*vec3(self.size, "size"))


class Plane(SceneObject):
    """A horizontal rectangle (floor, table top); ``uv_repeat`` tiles the texture."""

    _geometry_params = ("size", "uv_repeat")

    def __init__(self, size=(10.0, 10.0), uv_repeat=1.0, **kwargs):
        self.size = size
        self.uv_repeat = uv_repeat
        super().__init__(**kwargs)

    def _build_mesh(self) -> Mesh:
        width, depth = (self.size, self.size) if np.isscalar(self.size) else self.size
        return plane_mesh(width, depth, self.uv_repeat)


class Quad(SceneObject):
    """A vertical picture facing +Z (towards the default camera).

    With ``size=None`` (default) the width follows the texture's aspect ratio for
    the given ``height``, also when the texture is swapped later.
    """

    _geometry_params = ("size", "height")

    def __init__(self, size=None, *, height: float = 1.0, **kwargs):
        self.size = size
        self.height = height
        self._built_aspect = None
        super().__init__(**kwargs)

    def _aspect(self) -> float:
        texture = self.current_texture
        if texture is None and not callable(self.texture):
            texture = as_texture(self.texture)
        return texture.aspect if texture is not None and texture.width > 1 else 1.0

    def _build_mesh(self) -> Mesh:
        if self.size is not None:
            width, height = (self.size, self.size) if np.isscalar(self.size) else self.size
        else:
            self._built_aspect = self._aspect()
            width, height = self.height * self._built_aspect, self.height
        return quad_mesh(width, height)

    def update(self, t: float) -> None:
        super().update(t)
        if self.size is None and self._built_aspect is not None and abs(self._aspect() - self._built_aspect) > 1e-3:
            self._mesh = None


class Sphere(SceneObject):
    """A UV sphere; the texture is wrapped around it like a world map."""

    _geometry_params = ("radius", "segments")

    def __init__(self, radius: float = 0.5, segments: int = 48, **kwargs):
        self.radius = radius
        self.segments = segments
        super().__init__(**kwargs)

    def _build_mesh(self) -> Mesh:
        return sphere_mesh(self.radius, self.segments, max(8, self.segments // 2))


class Cylinder(SceneObject):
    """An upright cylinder; the texture wraps once around the side."""

    _geometry_params = ("radius", "height", "segments")

    def __init__(self, radius: float = 0.5, height: float = 1.0, segments: int = 48, **kwargs):
        self.radius = radius
        self.height = height
        self.segments = segments
        super().__init__(**kwargs)

    def _build_mesh(self) -> Mesh:
        return cylinder_mesh(self.radius, self.height, self.segments)


class Camera:
    """A perspective camera; ``position``, ``target`` and ``fov`` may be functions of time.

    Args:
        position: where the camera is.
        target: the point it looks at.
        fov: vertical field of view in degrees.
    """

    def __init__(self, position=(0.0, 1.6, 6.0), target=(0.0, 1.0, 0.0), fov=55.0, up=(0.0, 1.0, 0.0),
                 near: float = 0.05, far: float = 200.0):
        self.position, self.target, self.fov, self.up = position, target, fov, up
        self.near, self.far = near, far
        self.update(0.0)

    def update(self, t: float) -> None:
        self.current_position = vec3(resolve(self.position, t), "camera position")
        self.current_target = vec3(resolve(self.target, t), "camera target")
        self.current_fov = float(resolve(self.fov, t))

    def view_matrix(self) -> np.ndarray:
        return look_at(self.current_position, self.current_target, self.up)

    def projection_matrix(self, aspect: float) -> np.ndarray:
        return perspective(self.current_fov, aspect, self.near, self.far)


class Scene:
    """A collection of named objects, a camera and a light.

    Args:
        background: a colour ``(r, g, b)`` or any texture value (image path, ...)
            drawn behind everything, scaled to cover the frame.
        ambient: light level in shadow (0-1); 1 disables shading.
        light_direction: direction the light travels in (from the light into the scene).
        light_color: ``(r, g, b)`` colour of the light.
        camera: a :class:`Camera`; a default one is created when omitted.
    """

    def __init__(self, *, background=(28, 30, 36), ambient: float = 0.45, light_direction=(-0.4, -1.0, -0.5),
                 light_color=(255, 255, 255), camera: Camera | None = None):
        self.background = background
        self.ambient = ambient
        self.light_direction = light_direction
        self.light_color = light_color
        self.camera = camera or Camera()
        self.objects: dict[str, SceneObject] = {}
        self.time = 0.0
        self._callbacks = []
        self._folders: list[TextureFolder] = []
        self._counters: dict[str, int] = {}
        self._started = False

    # -- objects ------------------------------------------------------------------------

    def add(self, obj: SceneObject, name: str | None = None) -> SceneObject:
        """Add an object (optionally naming it) and return it."""
        name = name or obj.name or self._auto_name(obj)
        if name in self.objects and self.objects[name] is not obj:
            raise ValueError(f"the scene already has an object named {name!r}")
        obj.name = name
        self.objects[name] = obj
        return obj

    def _auto_name(self, obj: SceneObject) -> str:
        base = type(obj).__name__.lower()
        while True:
            self._counters[base] = self._counters.get(base, 0) + 1
            name = f"{base}{self._counters[base]}"
            if name not in self.objects:
                return name

    def remove(self, obj_or_name) -> None:
        name = obj_or_name if isinstance(obj_or_name, str) else obj_or_name.name
        del self.objects[name]

    def __getitem__(self, name: str) -> SceneObject:
        try:
            return self.objects[name]
        except KeyError:
            raise KeyError(f"no object named {name!r}; the scene has: {', '.join(self.objects) or 'nothing'}") from None

    def get(self, name: str, default=None):
        return self.objects.get(name, default)

    def __contains__(self, name: str) -> bool:
        return name in self.objects

    def __iter__(self):
        return iter(list(self.objects.values()))

    def __len__(self) -> int:
        return len(self.objects)

    def set_texture(self, name: str, texture) -> SceneObject:
        """Paint the object called ``name`` with ``texture`` (path, array, colour, ...)."""
        obj = self[name]
        obj.texture = texture
        return obj

    # -- behaviour ----------------------------------------------------------------------

    def on_update(self, fn):
        """Register ``fn(scene, t, dt)`` to run every frame (usable as a decorator)."""
        self._callbacks.append(fn)
        return fn

    def watch_folder(self, folder="uploads", interval: float = 0.5) -> "TextureFolder":
        """Apply image/video files dropped into ``folder`` to the objects with the same name.

        ``uploads/poster.jpg`` becomes the texture of ``scene["poster"]``; replacing the
        file updates it, deleting the file restores the original texture.
        """
        folder = Path(folder).expanduser().resolve()
        for watcher in self._folders:
            if watcher.folder == folder:
                return watcher
        watcher = TextureFolder(self, folder, interval)
        self._folders.append(watcher)
        return watcher

    def reset(self) -> None:
        """Put every physics body back at its starting position and velocity."""
        self.time = 0.0
        for obj in self:
            if obj.body is not None:
                obj.body.reset(resolve(obj.position, 0.0), resolve(obj.rotation, 0.0))
        self._started = True

    def update(self, t: float, dt: float) -> None:
        """Advance the scene to simulation time ``t`` (``dt`` seconds after the previous update)."""
        if not self._started:
            self.reset()
        self.time = t
        for callback in self._callbacks:
            callback(self, t, dt)
        for folder in self._folders:
            folder.poll()
        steps = max(1, math.ceil(dt / MAX_PHYSICS_STEP)) if dt > 0 else 0
        for obj in self:
            if obj.on_update is not None:
                obj.on_update(obj, t, dt)
            if obj.body is not None and steps:
                scale = vec3(resolve(obj.scale, t), "scale")
                for _ in range(steps):
                    obj.body.step(dt / steps, lambda rot, o=obj, s=scale: o.extents(rot, s))
            obj.update(t)
        self.camera.update(t)


class TextureFolder:
    """Watches a folder and applies files to objects with the same (case-insensitive) name."""

    def __init__(self, scene: Scene, folder, interval: float = 0.5):
        self.scene = scene
        self.folder = Path(folder).expanduser()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.interval = interval
        self._applied: dict[str, tuple[Path, float]] = {}
        self._originals: dict[str, object] = {}
        self._warned: set[Path] = set()
        self._last_poll = -math.inf

    def poll(self, force: bool = False) -> None:
        now = time.monotonic()
        if not force and now - self._last_poll < self.interval:
            return
        self._last_poll = now
        files: dict[str, tuple[Path, float]] = {}
        try:
            entries = list(self.folder.iterdir())
        except OSError:
            return
        for path in entries:
            if path.name.startswith(".") or path.suffix.lower() not in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS:
                continue
            try:
                mtime = path.stat().st_mtime
            except OSError:
                continue
            key = path.stem.lower()
            if key not in files or mtime > files[key][1]:  # newest file wins
                files[key] = (path, mtime)

        objects = {name.lower(): obj for name, obj in self.scene.objects.items()}
        for key, (path, mtime) in files.items():
            obj = objects.get(key)
            if obj is None:
                if path not in self._warned:
                    print(f"[uploads] {path.name}: no object named {path.stem!r} "
                          f"(objects: {', '.join(self.scene.objects)})")
                    self._warned.add(path)
                continue
            if self._applied.get(obj.name) == (path, mtime):
                continue
            original = obj.texture
            try:
                obj.texture = path
            except Exception as exc:  # noqa: BLE001 - file may still be copying; retry next poll
                print(f"[uploads] cannot use {path.name} yet: {exc}")
                continue
            self._originals.setdefault(obj.name, original)
            self._applied[obj.name] = (path, mtime)
            print(f"[uploads] {path.name} -> {obj.name}")

        current = {objects[key].name for key in files if key in objects}
        for name in list(self._applied):
            if name not in current:
                del self._applied[name]
                obj = self.scene.get(name)
                if obj is not None:
                    obj.texture = self._originals.pop(name, None)
                    print(f"[uploads] restored original texture of {name}")
