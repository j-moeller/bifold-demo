"""Off-screen OpenGL renderer that turns a Scene into a BGR image (the 2D projection)."""

from __future__ import annotations

import weakref

import cv2
import numpy as np

from .scene import Scene, SceneObject
from .textures import Texture, as_texture, parse_color
from .transforms import gl_bytes

# Shared by every program that draws lit, textured surfaces, so their output cannot drift apart.
_VS_EMIT = """
uniform mat4 u_mvp;
uniform mat3 u_normal_matrix;
out vec3 v_normal;
out vec2 v_uv;
out float v_depth;
void emit(vec4 position, vec3 normal, vec2 uv) {
    v_normal = u_normal_matrix * normal;
    v_uv = uv;
    gl_Position = u_mvp * position;
    v_depth = gl_Position.w;  // distance along the view direction
}
"""

_FOG = """
uniform vec3 u_fog_color;
uniform float u_fog_start;
uniform float u_fog_end;
uniform float u_fog_strength;
float fog_amount(float depth) {
    return clamp((depth - u_fog_start) / max(u_fog_end - u_fog_start, 1e-3), 0.0, 1.0) * u_fog_strength;
}
"""

_OBJECT_VS = "#version 330 core\n" + _VS_EMIT + """
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
void main() {
    emit(vec4(in_position, 1.0), in_normal, in_uv);
}
"""

# Linear-blend skinning. Joint j's skinning matrix (character space, column vectors) is stored in row j of an
# RGBA32F texture that is 3 texels wide: texel (r, j) holds matrix row r (the last row is always 0 0 0 1).
_SKINNED_VS = "#version 330 core\n" + _VS_EMIT + """
uniform sampler2D u_joints;
in vec3 in_position;
in vec3 in_normal;
in vec2 in_uv;
in vec4 in_joints;
in vec4 in_weights;
vec4 skin_row(int r, ivec4 j) {
    return in_weights.x * texelFetch(u_joints, ivec2(r, j.x), 0)
         + in_weights.y * texelFetch(u_joints, ivec2(r, j.y), 0)
         + in_weights.z * texelFetch(u_joints, ivec2(r, j.z), 0)
         + in_weights.w * texelFetch(u_joints, ivec2(r, j.w), 0);
}
void main() {
    ivec4 j = ivec4(in_joints + 0.5);
    vec4 r0 = skin_row(0, j), r1 = skin_row(1, j), r2 = skin_row(2, j);
    vec4 p = vec4(in_position, 1.0);
    vec3 n = in_normal;
    vec3 normal = vec3(dot(r0.xyz, n), dot(r1.xyz, n), dot(r2.xyz, n));
    emit(vec4(dot(r0, p), dot(r1, p), dot(r2, p), 1.0), normal, in_uv);
}
"""

_OBJECT_FS = "#version 330 core\n" + _FOG + """
uniform sampler2D u_texture;
uniform vec3 u_color;
uniform bool u_lit;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_ambient;
in vec3 v_normal;
in vec2 v_uv;
in float v_depth;
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
    rgb = mix(rgb, u_fog_color, fog_amount(v_depth));
    f_color = vec4(rgb.bgr, 1.0);  // BGR so OpenCV can use the pixels directly
}
"""

# Soft blob shadows: one rectangle per character on the floor (scene space), in_offset runs from -1 to 1 across it.
_SHADOW_VS = """
#version 330 core
uniform mat4 u_view_proj;
in vec3 in_position;
in vec2 in_offset;
out vec2 v_offset;
out float v_depth;
void main() {
    v_offset = in_offset;
    gl_Position = u_view_proj * vec4(in_position, 1.0);
    v_depth = gl_Position.w;
}
"""

_SHADOW_FS = "#version 330 core\n" + _FOG + """
uniform float u_strength;
in vec2 v_offset;
in float v_depth;
out vec4 f_color;
void main() {
    float falloff = 1.0 - min(dot(v_offset, v_offset), 1.0);
    f_color = vec4(0.0, 0.0, 0.0, u_strength * falloff * falloff * (1.0 - fog_amount(v_depth)));
}
"""

SHADOW_STRENGTH = 0.45  # darkening at the centre of a blob shadow (0-1)
SHADOW_LIFT = 0.015  # metres above the feet: above the floor and flat floor decals (carpets) up to 1 cm high
_SHADOW_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)], dtype=np.float64)
_UNSET = object()

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

    Objects with a ``skinned_model`` (:class:`bifold_demo.characters.character.Character`) are skinned on the
    GPU with their ``skin_palette`` and get a soft blob shadow on the floor.

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
        self._uniforms = {name: self._program[name] for name in ("u_mvp", "u_normal_matrix", "u_color", "u_lit")}
        self._bg_program["u_texture"].value = 0
        self._white = as_texture((255, 255, 255))
        self._meshes: dict[int, tuple] = {}  # id(mesh) -> (weakref, vao, vbo, ibo)
        self._textures: dict[int, list] = {}  # texture.id -> [weakref, gl texture, version]
        self._skinned_program = self._shadow_program = None  # compiled when the first character shows up
        self._shadow_vbo = self._shadow_vao = None
        self._primitives: dict[int, list] = {}  # id(primitive) -> [weakref, vao, vbo, ibo, texture spec, Texture]
        self._joints: dict[int, tuple] = {}  # id(character) -> (weakref, joint-palette texture)
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
        uniforms = _scene_uniforms(scene)
        _set_uniforms(self._program, uniforms)

        characters = []
        for obj in scene:
            if obj.current_visible:
                if getattr(obj, "skinned_model", None) is not None:
                    characters.append(obj)
                else:
                    self._draw(obj, view_proj, camera.current_position)
        if characters:
            self._draw_characters(characters, view_proj, camera.current_position, uniforms, scene.time)

        if fbo is not self._resolve_fbo:
            ctx.copy_framebuffer(self._resolve_fbo, fbo)
        data = self._resolve_fbo.read(components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(self.height, self.width, 3)
        self._frames += 1
        if self._frames % 300 == 0:
            self._collect_garbage()
        return cv2.flip(image, 0)  # OpenGL rows start at the bottom

    def _draw(self, obj: SceneObject, view_proj: np.ndarray, camera_position) -> None:
        model, normal_matrix = obj.matrices(camera_position)
        u = self._uniforms
        u["u_mvp"].write(gl_bytes(view_proj @ model))
        u["u_normal_matrix"].write(gl_bytes(normal_matrix))
        r, g, b = obj.current_color
        u["u_color"].value = (r / 255.0, g / 255.0, b / 255.0)
        u["u_lit"].value = bool(obj.lit)
        self._gl_texture(obj.current_texture or self._white).use(0)
        self._vao(obj.mesh).render(self._moderngl.TRIANGLES)

    def _draw_characters(self, characters, view_proj, camera_position, uniforms: dict, t: float) -> None:
        """GPU-skinned characters (opaque, alpha-tested), then their blob shadows blended over the scene."""
        prog = self._skinned()
        _set_uniforms(prog, uniforms)
        u, triangles = self._skinned_uniforms, self._moderngl.TRIANGLES
        for obj in characters:
            model, normal_matrix = obj.matrices(camera_position)
            u["u_mvp"].write(gl_bytes(view_proj @ model))
            u["u_normal_matrix"].write(gl_bytes(normal_matrix))
            r, g, b = obj.current_color
            u["u_color"].value = (r / 255.0, g / 255.0, b / 255.0)
            u["u_lit"].value = bool(obj.lit)
            self._joint_texture(obj).use(1)
            for primitive in obj.skinned_model.primitives:
                vao, texture = self._skinned_primitive(primitive)
                self._gl_texture(texture.at_time(t)).use(0)
                vao.render(triangles)
        shadows = [obj for obj in characters if getattr(obj, "shadow", False)]
        if shadows:
            self._draw_shadows(shadows, view_proj, camera_position, uniforms)

    def _draw_shadows(self, characters, view_proj, camera_position, uniforms: dict) -> None:
        """Soft dark ellipses on the floor in one draw call: depth-tested, blended, without writing depth."""
        mgl, ctx, fbo = self._moderngl, self.ctx, self._draw_fbo
        n = len(characters)
        models = np.array([obj.matrices(camera_position)[0][:3] for obj in characters])
        cx, cz, rx, rz = np.array([obj.shadow_ellipse for obj in characters], dtype=np.float64).T[:, :, None]
        local = np.ones((n, 6, 4))  # character-space corners of each ellipse's rectangle
        local[..., 0] = cx + _SHADOW_CORNERS[:, 0] * rx
        local[..., 1] = SHADOW_LIFT
        local[..., 2] = cz + _SHADOW_CORNERS[:, 1] * rz
        vertices = np.empty((n, 6, 5), dtype="f4")
        vertices[..., :3] = np.einsum("nij,nkj->nki", models, local)
        vertices[..., 3:] = _SHADOW_CORNERS
        data = vertices.tobytes()
        if self._shadow_program is None:
            self._shadow_program = ctx.program(vertex_shader=_SHADOW_VS, fragment_shader=_SHADOW_FS)
            self._shadow_program["u_strength"].value = SHADOW_STRENGTH
        if self._shadow_vbo is None or self._shadow_vbo.size < len(data):
            if self._shadow_vbo is not None:
                self._shadow_vao.release()
                self._shadow_vbo.release()
            self._shadow_vbo = ctx.buffer(reserve=max(2 * len(data), 4096), dynamic=True)
            self._shadow_vao = ctx.vertex_array(self._shadow_program,
                                                [(self._shadow_vbo, "3f 2f", "in_position", "in_offset")])
        self._shadow_vbo.write(data)
        prog = self._shadow_program
        _set_uniforms(prog, uniforms)
        prog["u_view_proj"].write(gl_bytes(view_proj))
        ctx.enable(mgl.BLEND)
        ctx.blend_func = mgl.SRC_ALPHA, mgl.ONE_MINUS_SRC_ALPHA
        fbo.depth_mask = False
        try:
            self._shadow_vao.render(mgl.TRIANGLES, vertices=6 * n)
        finally:
            fbo.depth_mask = True
            ctx.disable(mgl.BLEND)

    def _skinned(self):
        if self._skinned_program is None:
            prog = self.ctx.program(vertex_shader=_SKINNED_VS, fragment_shader=_OBJECT_FS)
            prog["u_texture"].value = 0
            prog["u_joints"].value = 1
            self._skinned_uniforms = {name: prog[name] for name in ("u_mvp", "u_normal_matrix", "u_color", "u_lit")}
            self._skinned_program = prog
        return self._skinned_program

    def _skinned_primitive(self, primitive):
        """``(vao, Texture)`` of a character's mesh part, uploaded once and shared by every character using it."""
        key = id(primitive)
        entry = self._primitives.get(key)
        if entry is None or entry[0]() is not primitive:
            if entry is not None:  # id reused by a new primitive
                self._release_primitive(key)
            vbo = self.ctx.buffer(primitive.interleaved().tobytes())
            ibo = self.ctx.buffer(np.asarray(primitive.indices).astype("u4").tobytes())
            vao = self.ctx.vertex_array(
                self._skinned(),
                [(vbo, "3f 3f 2f 4f 4f", "in_position", "in_normal", "in_uv", "in_joints", "in_weights")],
                index_buffer=ibo, skip_errors=True,
            )
            entry = [weakref.ref(primitive), vao, vbo, ibo, _UNSET, None]
            self._primitives[key] = entry
        if entry[4] is not primitive.texture:
            entry[4], entry[5] = primitive.texture, _character_texture(primitive.texture) or self._white
        return entry[1], entry[5]

    def _joint_texture(self, obj):
        """The character's joint-palette texture (one per character, rewritten every frame)."""
        palette = np.asarray(obj.skin_palette, dtype=np.float32)
        entry = self._joints.get(id(obj))
        if entry is None or entry[0]() is not obj or entry[1].height != len(palette):
            if entry is not None:
                entry[1].release()
            gl_tex = self.ctx.texture((3, len(palette)), 4, dtype="f4")
            gl_tex.filter = (self._moderngl.NEAREST, self._moderngl.NEAREST)
            entry = (weakref.ref(obj), gl_tex)
            self._joints[id(obj)] = entry
        entry[1].write(palette[:, :3, :].tobytes())
        return entry[1]

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
        for key, entry in list(self._primitives.items()):
            if entry[0]() is None:
                self._release_primitive(key)
        for key, (ref, gl_tex) in list(self._joints.items()):
            if ref() is None:
                gl_tex.release()
                del self._joints[key]

    def _release_mesh(self, key: int) -> None:
        _ref, vao, vbo, ibo = self._meshes.pop(key)
        for resource in (vao, vbo, ibo):
            resource.release()

    def _release_primitive(self, key: int) -> None:
        _ref, vao, vbo, ibo, _spec, _texture = self._primitives.pop(key)
        for resource in (vao, vbo, ibo):
            resource.release()

    def release(self) -> None:
        for key in list(self._meshes):
            self._release_mesh(key)
        for key in list(self._primitives):
            self._release_primitive(key)
        for _ref, gl_tex in self._joints.values():
            gl_tex.release()
        self._joints.clear()
        if self._shadow_vbo is not None:
            self._shadow_vao.release()
            self._shadow_vbo.release()
            self._shadow_vbo = self._shadow_vao = None
        for _ref, gl_tex, _version in self._textures.values():
            gl_tex.release()
        self._textures.clear()
        self.ctx.release()


def _scene_uniforms(scene: Scene) -> dict:
    """Light and fog uniform values shared by every surface program."""
    light = np.asarray(scene.light_direction, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-9)
    fog = scene.fog
    values = {
        "u_light_dir": tuple(light),
        "u_light_color": tuple(c / 255.0 for c in parse_color(scene.light_color)),
        "u_ambient": float(np.clip(scene.ambient, 0.0, 1.0)),
        "u_fog_strength": float(np.clip(fog.strength, 0.0, 1.0)) if fog else 0.0,
    }
    if fog:
        values["u_fog_color"] = tuple(c / 255.0 for c in parse_color(fog.color))
        values["u_fog_start"] = float(fog.start)
        values["u_fog_end"] = float(fog.end)
    return values


def _set_uniforms(program, values: dict) -> None:
    for name, value in values.items():
        uniform = program.get(name, None)
        if uniform is not None:
            uniform.value = value


def _character_texture(spec) -> Texture | None:
    """Character textures are RGB(A) arrays in RGB order (row 0 = top); other values go through ``as_texture``."""
    if isinstance(spec, np.ndarray) and spec.ndim == 3 and spec.shape[2] in (3, 4):
        return Texture.from_array(spec, order="rgb", name="character")
    return as_texture(spec)
