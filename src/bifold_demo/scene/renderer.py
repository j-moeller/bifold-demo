"""Off-screen OpenGL renderer that turns a Scene into a BGR image (the 2D projection)."""

from __future__ import annotations

import weakref

import cv2
import numpy as np

from .scene import Scene, SceneObject
from .textures import Texture, as_texture, parse_color
from .transforms import gl_bytes

_OBJECT_VS = """
#version 330 core
uniform mat4 u_mvp;
uniform mat3 u_normal_matrix;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
out vec3 v_normal;
out vec2 v_uv;
void main() {
    v_normal = u_normal_matrix * in_normal;
    v_uv = in_uv;
    gl_Position = u_mvp * vec4(in_position, 1.0);
}
"""

_OBJECT_FS = """
#version 330 core
uniform sampler2D u_texture;
uniform vec3 u_color;
uniform bool u_lit;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_ambient;
in vec3 v_normal;
in vec2 v_uv;
out vec4 f_color;
void main() {
    vec4 texel = texture(u_texture, v_uv);
    if (texel.a < 0.5) discard;  // cut-outs from transparent PNGs
    vec3 rgb = texel.rgb * u_color;
    if (u_lit) {
        vec3 n = normalize(v_normal);
        if (!gl_FrontFacing) n = -n;  // light both sides of thin objects
        float diffuse = max(dot(n, -u_light_dir), 0.0);
        rgb *= u_ambient + (1.0 - u_ambient) * diffuse * u_light_color;
    }
    f_color = vec4(rgb.bgr, 1.0);  // BGR so OpenCV can use the pixels directly
}
"""

_BACKGROUND_VS = """
#version 330 core
uniform vec2 u_uv_scale;
out vec2 v_uv;
void main() {
    vec2 corner = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1)) - 1.0;
    v_uv = (corner * 0.5) * u_uv_scale + 0.5;
    gl_Position = vec4(corner, 0.0, 1.0);
}
"""

_BACKGROUND_FS = """
#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    f_color = vec4(texture(u_texture, v_uv).bgr, 1.0);
}
"""


class Renderer:
    """Renders scenes off-screen at a fixed resolution.

    The OpenGL context belongs to the thread that created the renderer; call
    :meth:`render` from that thread only.
    """

    def __init__(self, width: int = 1280, height: int = 720, samples: int = 4):
        import moderngl

        self._moderngl = moderngl
        self.width, self.height = int(width), int(height)
        self.ctx = moderngl.create_standalone_context(require=330)
        size = (self.width, self.height)
        samples = max(0, min(samples, self.ctx.max_samples))
        self._resolve_fbo = self.ctx.framebuffer(
            color_attachments=[self.ctx.renderbuffer(size)],
            depth_attachment=None if samples else self.ctx.depth_renderbuffer(size),
        )
        self._draw_fbo = (
            self.ctx.framebuffer(
                color_attachments=[self.ctx.renderbuffer(size, samples=samples)],
                depth_attachment=self.ctx.depth_renderbuffer(size, samples=samples),
            )
            if samples
            else self._resolve_fbo
        )
        self._program = self.ctx.program(vertex_shader=_OBJECT_VS, fragment_shader=_OBJECT_FS)
        self._bg_program = self.ctx.program(vertex_shader=_BACKGROUND_VS, fragment_shader=_BACKGROUND_FS)
        self._bg_vao = self.ctx.vertex_array(self._bg_program, [])
        self._program["u_texture"].value = 0
        self._bg_program["u_texture"].value = 0
        self._white = as_texture((255, 255, 255))
        self._meshes: dict[int, tuple] = {}  # id(mesh) -> (weakref, vao, vbo, ibo)
        self._textures: dict[int, list] = {}  # texture.id -> [weakref, gl texture, version]
        self.info = self.ctx.info.get("GL_RENDERER", "OpenGL")
        self._frames = 0

    def render(self, scene: Scene) -> np.ndarray:
        """Draw ``scene`` as evaluated by its last ``update`` and return an HxWx3 BGR image."""
        ctx, fbo = self.ctx, self._draw_fbo
        fbo.use()
        ctx.viewport = (0, 0, self.width, self.height)
        background = scene.background
        if callable(background):
            background = background(scene.time)
        bg_texture = None
        if isinstance(background, (tuple, list)) or (isinstance(background, str) and background.startswith("#")):
            r, g, b = (c / 255.0 for c in parse_color(background))
            fbo.clear(b, g, r, 1.0)
        else:
            bg_texture = as_texture(background)
            fbo.clear(0.0, 0.0, 0.0, 1.0)

        if bg_texture is not None:
            ctx.disable(self._moderngl.DEPTH_TEST)
            image_aspect, frame_aspect = bg_texture.aspect, self.width / self.height
            scale = (frame_aspect / image_aspect, 1.0) if image_aspect > frame_aspect else (1.0, image_aspect / frame_aspect)
            self._bg_program["u_uv_scale"].value = scale  # "cover": crop instead of stretching
            self._gl_texture(bg_texture.at_time(scene.time)).use(0)
            self._bg_vao.render(self._moderngl.TRIANGLES, vertices=3)

        ctx.enable(self._moderngl.DEPTH_TEST)
        ctx.disable(self._moderngl.CULL_FACE)
        camera = scene.camera
        view_proj = camera.projection_matrix(self.width / self.height) @ camera.view_matrix()
        light = np.asarray(scene.light_direction, dtype=np.float64)
        light = light / max(np.linalg.norm(light), 1e-9)
        prog = self._program
        prog["u_light_dir"].value = tuple(light)
        prog["u_light_color"].value = tuple(c / 255.0 for c in parse_color(scene.light_color))
        prog["u_ambient"].value = float(np.clip(scene.ambient, 0.0, 1.0))

        for obj in scene:
            if obj.current_visible:
                self._draw(obj, view_proj, camera.current_position)

        if fbo is not self._resolve_fbo:
            ctx.copy_framebuffer(self._resolve_fbo, fbo)
        data = self._resolve_fbo.read(components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._frames += 1
        if self._frames % 300 == 0:
            self._collect_garbage()
        return cv2.flip(image, 0)  # OpenGL rows start at the bottom

    def _draw(self, obj: SceneObject, view_proj: np.ndarray, camera_position) -> None:
        model = obj.model_matrix(camera_position)
        normal_matrix = np.linalg.inv(model[:3, :3]).T if abs(np.linalg.det(model[:3, :3])) > 1e-12 else np.eye(3)
        prog = self._program
        prog["u_mvp"].write(gl_bytes(view_proj @ model))
        prog["u_normal_matrix"].write(gl_bytes(normal_matrix))
        prog["u_color"].value = tuple(c / 255.0 for c in obj.current_color)
        prog["u_lit"].value = bool(obj.lit)
        self._gl_texture(obj.current_texture or self._white).use(0)
        self._vao(obj.mesh).render(self._moderngl.TRIANGLES)

    def _vao(self, mesh):
        entry = self._meshes.get(id(mesh))
        if entry is not None and entry[0]() is mesh:
            return entry[1]
        if entry is not None:  # id reused by a new mesh
            self._release_mesh(id(mesh))
        vbo = self.ctx.buffer(mesh.interleaved().tobytes())
        ibo = self.ctx.buffer(mesh.indices.astype("u4").tobytes())
        vao = self.ctx.vertex_array(
            self._program, [(vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")], index_buffer=ibo,
            skip_errors=True,
        )
        self._meshes[id(mesh)] = (weakref.ref(mesh), vao, vbo, ibo)
        return vao

    def _gl_texture(self, texture: Texture):
        entry = self._textures.get(texture.id)
        if entry is not None and entry[2] == texture.version:
            return entry[1]
        pixels = np.ascontiguousarray(np.flipud(texture.rgba)).tobytes()  # first row = bottom for OpenGL
        size = (texture.width, texture.height)
        if entry is not None and entry[1].size == size:
            gl_tex = entry[1]
            gl_tex.write(pixels)
        else:
            if entry is not None:
                entry[1].release()
            gl_tex = self.ctx.texture(size, 4, pixels)
            gl_tex.repeat_x = gl_tex.repeat_y = True
            try:
                gl_tex.anisotropy = 8.0
            except Exception:  # noqa: BLE001 - anisotropic filtering is optional
                pass
        if texture.width > 1 or texture.height > 1:
            gl_tex.build_mipmaps()
            gl_tex.filter = (self._moderngl.LINEAR_MIPMAP_LINEAR, self._moderngl.LINEAR)
        self._textures[texture.id] = [weakref.ref(texture), gl_tex, texture.version]
        return gl_tex

    def _collect_garbage(self) -> None:
        """Free GPU resources of meshes and textures that no longer exist."""
        for key, (ref, *_rest) in list(self._meshes.items()):
            if ref() is None:
                self._release_mesh(key)
        for key, (ref, gl_tex, _version) in list(self._textures.items()):
            if ref() is None:
                gl_tex.release()
                del self._textures[key]

    def _release_mesh(self, key: int) -> None:
        _ref, vao, vbo, ibo = self._meshes.pop(key)
        for resource in (vao, vbo, ibo):
            resource.release()

    def release(self) -> None:
        for key in list(self._meshes):
            self._release_mesh(key)
        for _ref, gl_tex, _version in self._textures.values():
            gl_tex.release()
        self._textures.clear()
        self.ctx.release()
