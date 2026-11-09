"""Off-screen OpenGL renderer that turns a Scene into a BGR image (the 2D projection).

Besides the whole view (:meth:`Renderer.render`) it draws *windows* (:meth:`Renderer.render_window`): a rectangle of
the camera's picture through an off-axis part of the same frustum, at a higher pixel scale than the view, with the
GPU resources the view already uploaded. That is the virtual high-resolution sensor of :mod:`bifold_demo.sensor`.

**Linear light.** Light adds up and pixels average in linear light, while images hold sRGB-encoded colours. So
textures are sRGB textures (the GPU decodes a colour when it samples it) with mipmaps averaged in linear light
(:func:`~.textures.mip_chain`), shading, fog and the blending of shadows happen in linear light, and the picture is
an sRGB framebuffer: the GPU encodes what the shaders write, and the MSAA samples of a pixel are averaged as light.
Colours given to the scene (an object's tint, the light, the fog, the background) are sRGB colours as everywhere
else and are converted; so are the scene's ``ambient`` level and the blob shadows' darkening, which are given as
they look, so that what lies in shadow looks as dark as it always did and only the light's fall-off between light
and shadow changes. Data such as normal maps are not decoded (``Texture.srgb``). ``Renderer.linear = False`` draws
as before instead, on the stored sRGB values (a picture that must not depend on the light model: the avatar renders
the head-texture pipeline analyses, :func:`bifold_demo.characters.headtex.landmarks.render_views`).
"""

from __future__ import annotations

import weakref

import cv2
import numpy as np

from .detail import DetailCache, base_copy, eye_pixels, is_large
from .frustum import Bounds, boxes_outside, frustum_planes, spheres_outside, world_boxes
from .scene import Scene, SceneObject, SubCamera
from .textures import (
    Texture,
    VideoTexture,
    as_data_texture,
    as_texture,
    color_to_linear,
    drop_pixels,
    parse_color,
    upload_levels,
)
from .transforms import crop_matrix, gl_bytes

SRGB8_ALPHA8 = 0x8C43  # the internal format of sRGB textures and of the picture's colour buffers
FRAMEBUFFER_SRGB = 0x8DB9  # with it enabled, writes to sRGB colour buffers are encoded (and blended in linear light)

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
_SKIN = """
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
vec3 skin_direction(vec4 r0, vec4 r1, vec4 r2, vec3 d) {
    return vec3(dot(r0.xyz, d), dot(r1.xyz, d), dot(r2.xyz, d));
}
"""

_SKINNED_VS = "#version 330 core\n" + _VS_EMIT + _SKIN + """
void main() {
    ivec4 j = ivec4(in_joints + 0.5);
    vec4 r0 = skin_row(0, j), r1 = skin_row(1, j), r2 = skin_row(2, j);
    vec4 p = vec4(in_position, 1.0);
    emit(vec4(dot(r0, p), dot(r1, p), dot(r2, p), 1.0), skin_direction(r0, r1, r2, in_normal), in_uv);
}
"""

# The light shared by the lit programs: the key light (Lambert) and ambient light, which with u_hemisphere > 0 comes
# more from above than from below (a hall's floor and walls bounce less light up than the ceiling's lamps send down;
# 0 at the low quality setting).
_LIGHT = """
uniform bool u_lit;
uniform vec3 u_light_dir;
uniform vec3 u_light_color;
uniform float u_ambient;
uniform float u_hemisphere;
vec3 hemisphere_light(vec3 n, float occlusion, float hemisphere) {
    float ambient = u_ambient * (1.0 + hemisphere * n.y) * occlusion;
    return vec3(ambient) + (1.0 - u_ambient) * max(dot(n, -u_light_dir), 0.0) * u_light_color;
}
vec3 diffuse_light(vec3 n, float occlusion) {
    return hemisphere_light(n, occlusion, u_hemisphere);
}
"""

_OBJECT_FS = "#version 330 core\n" + _FOG + _LIGHT + """
uniform sampler2D u_texture;
uniform vec3 u_color;
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
        rgb *= diffuse_light(n, 1.0);
    }
    rgb = mix(rgb, u_fog_color, fog_amount(v_depth));
    f_color = vec4(rgb.bgr, 1.0);  // BGR so OpenCV can use the pixels directly
}
"""

# Mesh parts with a packed map (heads, see characters/tangents.py) at the high quality setting: the skinned tangent
# frame turns the map's normal into the surface's, and a specular lobe (GGX with Schlick's Fresnel, F0 of skin) is
# scaled by the map's specular level, sharper where the level is high (lips, the wet eyes).
_MAPPED_VS = "#version 330 core\n" + _SKIN + """
uniform mat4 u_mvp;
uniform mat4 u_model;
uniform mat3 u_normal_matrix;
in vec4 in_tangent;
out vec3 v_normal;
out vec4 v_tangent;
out vec3 v_world;
out vec2 v_uv;
out float v_depth;
void main() {
    ivec4 j = ivec4(in_joints + 0.5);
    vec4 r0 = skin_row(0, j), r1 = skin_row(1, j), r2 = skin_row(2, j);
    vec4 q = vec4(in_position, 1.0);
    vec4 p = vec4(dot(r0, q), dot(r1, q), dot(r2, q), 1.0);
    v_normal = u_normal_matrix * skin_direction(r0, r1, r2, in_normal);
    // (the tangent lies in the surface: the model matrix would carry it, the normal matrix does the same for the
    // turns and the even scales characters have; a character scaled unevenly would need the model matrix's)
    v_tangent = vec4(u_normal_matrix * skin_direction(r0, r1, r2, in_tangent.xyz), in_tangent.w);
    v_world = (u_model * p).xyz;
    v_uv = in_uv;
    gl_Position = u_mvp * p;
    v_depth = gl_Position.w;
}
"""

_MAPPED_FS = "#version 330 core\n" + _FOG + _LIGHT + """
uniform sampler2D u_texture;
uniform sampler2D u_maps;
uniform sampler2D u_photo;  // red: how much of a texel is a face from photos (0 on the avatar's own texture)
uniform vec3 u_color;
uniform vec3 u_eye;
uniform float u_specular;
uniform float u_normal_strength;
uniform float u_photo_hemisphere;  // the share of the hemisphere light's variation a face from photos gets
uniform float u_photo_level;  // the specular level a face from photos gets instead of the map's
const float PI = 3.14159265;
const float SKIN_LEVEL = 0.07;  // the specular maps' level on most of the skin
const float SKIN_F0 = 0.028;  // Fresnel reflectance of skin head-on
in vec3 v_normal;
in vec4 v_tangent;
in vec3 v_world;
in vec2 v_uv;
in float v_depth;
out vec4 f_color;
void main() {
    vec4 texel = texture(u_texture, v_uv);
    if (texel.a < 0.5) discard;
    vec3 rgb = texel.rgb * u_color;
    if (u_lit) {
        vec4 maps = texture(u_maps, v_uv);
        // the maps hold the avatar's own relief and shine: a face from photos brings the person's own instead
        float photo = texture(u_photo, v_uv).r;
        float own = 1.0 - photo;
        vec3 ng = normalize(v_normal);
        if (!gl_FrontFacing) ng = -ng;
        vec3 t = v_tangent.xyz - ng * dot(ng, v_tangent.xyz);
        vec3 n = ng;
        if (dot(t, t) > 1e-12) {
            t = normalize(t);
            vec3 b = cross(ng, t) * (v_tangent.w < 0.0 ? -1.0 : 1.0);
            vec2 xy = (maps.rg * 2.0 - 1.0) * (u_normal_strength * own);
            n = normalize(t * xy.x + b * xy.y + ng * sqrt(max(1.0 - dot(xy, xy), 0.0)));
        }
        vec3 lit = rgb * hemisphere_light(n, maps.a, u_hemisphere * mix(1.0, u_photo_hemisphere, photo));
        vec3 l = -u_light_dir;
        float nl = max(dot(n, l), 0.0);
        float level = maps.b * own + u_photo_level * photo;
        if (nl > 0.0 && level > 0.0 && u_specular > 0.0) {
            vec3 v = normalize(u_eye - v_world);
            vec3 h = normalize(l + v);
            float nv = max(dot(n, v), 1e-3);
            float nh = max(dot(n, h), 0.0);
            float vh = max(dot(v, h), 0.0);
            float rough = mix(0.45, 0.2, smoothstep(0.15, 0.6, level));  // skin, then wetter: lips, eyes
            float a2 = rough * rough * rough * rough;
            float dd = nh * nh * (a2 - 1.0) + 1.0;
            float k = (rough + 1.0) * (rough + 1.0) / 8.0;
            float g = nl / (nl * (1.0 - k) + k) * nv / (nv * (1.0 - k) + k);
            float fresnel = SKIN_F0 + (1.0 - SKIN_F0) * pow(1.0 - vh, 5.0);
            float strength = min(level / SKIN_LEVEL, 3.0) * u_specular;
            // GGX lobe (times pi: the diffuse term above leaves out Lambert's 1/pi too)
            lit += (1.0 - u_ambient) * u_light_color * (a2 / (dd * dd) * g * fresnel * strength / (4.0 * nv));
        }
        rgb = lit;
    }
    rgb = mix(rgb, u_fog_color, fog_amount(v_depth));
    f_color = vec4(rgb.bgr, 1.0);
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
uniform float u_shadow_gamma;  // 2.2 in linear light, 1 on stored values (Renderer.linear)
in vec2 v_offset;
in float v_depth;
out vec4 f_color;
void main() {
    float falloff = 1.0 - min(dot(v_offset, v_offset), 1.0);
    float darken = u_strength * falloff * falloff * (1.0 - fog_amount(v_depth));  // as it looks
    f_color = vec4(0.0, 0.0, 0.0, 1.0 - pow(1.0 - darken, u_shadow_gamma));
}
"""

# A face from photos (a head's colour texture of a person's own, Character.photo_mask) carries the person's own
# relief and shine, and soft light from above: the photo's, or, from the head-texture stages, the avatar's painted
# light put back. So where it covers the head it is drawn without the avatar's normal and specular maps (they put
# the avatar's folds and shine on another face) and without the hemisphere light's variation (it would add that
# light a second time): measured with real people's photos against their other photos, the recognisers' genuine
# scores then are those of the low quality, and 0.04-0.09 lower with the maps and the hemisphere (see the README).
PHOTO_HEMISPHERE = 0.0  # the share of the hemisphere light's variation a face from photos gets
PHOTO_LEVEL = 0.0  # the specular level a face from photos gets instead of the avatar's map's
SHADOW_STRENGTH = 0.45  # darkening at the centre of a blob shadow (0-1)
SHADOW_LIFT = 0.015  # metres above the feet: above the floor and flat floor decals (carpets) up to 1 cm high
_SHADOW_CORNERS = np.array([(-1, -1), (1, -1), (1, 1), (-1, -1), (1, 1), (-1, 1)], dtype=np.float64)
_UNSET = object()

# A background image covers the whole picture; a window shows its part of it: u_ndc_scale and u_ndc_offset map the
# window's corners to where they are in the whole picture (1 and 0 for the whole view).
_BACKGROUND_VS = """
#version 330 core
uniform vec2 u_uv_scale;
uniform vec2 u_ndc_scale;
uniform vec2 u_ndc_offset;
out vec2 v_uv;
void main() {
    vec2 corner = vec2(float((gl_VertexID & 1) << 2), float((gl_VertexID & 2) << 1)) - 1.0;
    vec2 whole = corner * u_ndc_scale + u_ndc_offset;
    v_uv = (whole * 0.5) * u_uv_scale + 0.5;
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

# Counting what of someone would show (Renderer.visible_pixels): only the depth test matters, nothing is written.
_PROBE_FS = """
#version 330 core
void main() {
}
"""
_PROBE_SHADOW = 0.7  # the part of a blob shadow that counts (outside it, it darkens the floor by under 12 %)

# Where one person covers a window (Renderer.window_mask): white where they are the nearest surface; what textures cut
# out (hair cards) is left out as when drawing.
_MASK_FS = """
#version 330 core
uniform sampler2D u_texture;
in vec2 v_uv;
out vec4 f_color;
void main() {
    if (texture(u_texture, v_uv).a < 0.5) discard;
    f_color = vec4(1.0);
}
"""
WINDOW_BYTES = 192 << 20  # window framebuffers kept, in bytes (least recently used freed first; Renderer.window_bytes)


def _bucket(side: int) -> int:
    """A window side rounded up for its framebuffer: to a multiple of a quarter of the next power of two (at least
    16), so that windows of nearby sizes share framebuffers (a side at most half as long again as the window's, above
    32 pixels)."""
    step = max(16, (1 << max(0, int(side) - 1).bit_length()) // 4)
    return -(-int(side) // step) * step


class Renderer:
    """Renders scenes off-screen at a fixed resolution.

    Objects with a ``skinned_model`` (:class:`bifold_demo.characters.character.Character`) are skinned on the
    GPU with their ``skin_palette`` and get a soft blob shadow on the floor.

    :meth:`render_window` draws a rectangle of the same picture at another pixel scale (a window of a virtual
    high-resolution sensor, see :mod:`bifold_demo.sensor`), reusing what :meth:`render` uploaded.

    :attr:`linear` (True): shade, filter and blend in linear light (see the module docs); False: on the stored sRGB
    values, as the renderer did before (textures uploaded again when it changes).

    **Quality.** :attr:`quality` (the whole view) and :attr:`window_quality` (windows) are ``"high"`` or ``"low"``.
    High: mesh parts with ``tangents`` and ``maps`` (the heads) are drawn with their normal and specular maps, and the
    ambient light comes more from above than from below as much as the scene's ``hemisphere`` says. Low: the shading
    of before (Lambert and a constant ambient light, in linear light), a little cheaper per pixel on a software
    renderer. :attr:`specular` and :attr:`normal_strength` scale the specular lobe and the normal maps (1: as the maps
    say).

    **Detail.** A person whose eyes are at least :attr:`detail_from` pixels apart in the picture being drawn gets
    their head's 2048-pixel ``detail`` textures there (and a texture larger than 1024 that they wear instead of the
    model's is drawn at full size); everywhere else the 1024 ones. The detail textures of the last
    :attr:`detail_heads` heads stay on the GPU; :attr:`detail_wait` False reads new ones in the background (see
    :mod:`.detail`). :meth:`memory` says how much the textures take.

    The OpenGL context belongs to the thread that created the renderer; call
    :meth:`render` from that thread only. Every public method makes the renderer's own context current (with several
    renderers in one thread, the one made last is current otherwise).
    """

    linear = True  # the light model: linear light (False: the stored sRGB values, as before)
    quality = "high"  # the shading of the whole view: "high" or "low" (see the class docs)
    window_quality = "high"  # the shading of windows
    specular = 1.0  # scale of the heads' specular lobe (0: none)
    normal_strength = 1.0  # scale of the heads' normal maps' slopes (0: flat)
    detail_from = 64.0  # pixels between the eyes from which a head is drawn with its 2048 textures (inf: never)
    detail_heads = 6  # heads whose 2048 textures stay on the GPU (about 43 MiB each)
    detail_wait = True  # read a head's 2048 textures when a picture first needs them (False: in the background)

    def __init__(self, width: int = 1280, height: int = 720, samples: int = 4):
        import moderngl

        self._moderngl = moderngl
        self.width, self.height = int(width), int(height)
        self.ctx = moderngl.create_standalone_context(require=330)
        size = (self.width, self.height)
        samples = max(0, min(samples, self.ctx.max_samples))
        self._resolve_fbo = self.ctx.framebuffer(
            color_attachments=[self._colour_buffer(size)],
            depth_attachment=None if samples else self.ctx.depth_renderbuffer(size),
        )
        self._draw_fbo = (
            self.ctx.framebuffer(
                color_attachments=[self._colour_buffer(size, samples)],
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
        self._flat = Texture(np.array([[[128, 128, 0, 255]]], np.uint8), name="flat maps", srgb=False)  # no maps
        self._no_photo = Texture(np.array([[[0, 0, 0, 255]]], np.uint8), name="no photo", srgb=False)
        self._meshes: dict[int, tuple] = {}  # id(mesh) -> (weakref, vao, vbo, ibo)
        self._textures: dict[int, list] = {}  # texture.id -> [weakref, gl texture, version, mipmaps built]
        self._skinned_program = self._shadow_program = None  # compiled when the first character shows up
        self._mapped_program = None  # compiled when the first mesh part with maps is drawn at the high quality
        self._shadow_vbo = self._shadow_vao = None
        # id(primitive) -> [weakref, vao, vbo, ibo, texture spec, Texture, maps spec, maps Texture, mapped vao,
        #                   tangent vbo]
        self._primitives: dict[int, list] = {}
        self._joints: dict[int, tuple] = {}  # id(character) -> (weakref, joint-palette texture)
        self._details = DetailCache()  # the 2048 textures of the heads that need them most
        self._details_linear = None  # the light model the detail textures were uploaded for
        self._head_bones = weakref.WeakKeyDictionary()  # skeleton -> its head bone (-1: none)
        self._probes = None  # (object program, skinned program, shadow program): see visible_pixels
        self._drawn = None  # (view-projection, camera position) of the last frame
        self._samples = samples  # MSAA samples per pixel (0: none), for windows too
        self._window_targets: dict[tuple[int, int], tuple] = {}  # bucket (w, h) -> (draw, resolve, buffers, bytes)
        self.window_bytes = WINDOW_BYTES  # how much the window framebuffers kept may take
        self._bounds = Bounds()  # for culling (see frustum.py)
        self._frame_id = 0  # counts the frames drawn by render(), for the joint palettes windows reuse
        self.cull_view = False  # also cull the whole view (windows always do, unless asked not to)
        self.window_stats: dict = {}  # what the last culled window drew and left out (objects, people, shadows)
        self._last_window = None  # (target, size, view-projection, camera position, time) of the last window
        self._masks = None  # (object program, skinned program): see window_mask
        info = self.ctx.info
        self.max_window = int(info.get("GL_MAX_RENDERBUFFER_SIZE", 16384))  # the longest side a window may have
        self.info = info.get("GL_RENDERER", "OpenGL")
        self._frames = 0

    @property
    def aspect(self) -> float:
        """Width / height of the whole view."""
        return self.width / self.height

    def _colour_buffer(self, size, samples: int = 0):
        """An sRGB colour buffer for the picture (a multisample texture with ``samples``): with
        :data:`FRAMEBUFFER_SRGB` enabled, what is drawn into it is encoded, blended and resolved in linear light.
        (A renderbuffer cannot be made sRGB through moderngl.)"""
        return self.ctx.texture(size, 4, samples=samples, internal_format=SRGB8_ALPHA8)

    def render(self, scene: Scene) -> np.ndarray:
        """Draw ``scene`` as evaluated by its last ``update`` and return an HxWx3 BGR image. The renderer's own
        OpenGL context is made current for it (with several renderers in one thread, the one made last is current
        otherwise, and another one would draw into it)."""
        with self.ctx:
            return self._render(scene)

    def render_window(self, scene: Scene, ndc, size, aspect: float | None = None, cull: bool = True) -> np.ndarray:
        """Draw the rectangle ``ndc`` = ``(left, bottom, right, top)`` of the camera's picture (in normalised device
        coordinates of the whole picture, whose width / height is ``aspect``, the view's by default) at ``size`` =
        ``(width, height)`` pixels, and return that HxWx3 BGR image.

        The window is seen through an off-axis part of the camera's frustum (``crop_matrix(*ndc) @ P``,
        :class:`~bifold_demo.scene.scene.SubCamera`): when ``size`` is the rectangle's share of a picture of the whole
        view at some resolution, the window is that part of such a picture, with the same perspective, depth, fog,
        MSAA (the view's sample count) and texture filtering. Call it after :meth:`render` of the same frame (from the
        same thread), so that everything it needs is on the GPU and its windows show the frame's moment.

        With ``cull`` only what may show in the window is drawn (:mod:`.frustum`; no pixel changes), and the joint
        palettes the frame uploaded are used again. :attr:`window_stats` then says what was drawn and left out. The
        window is drawn in the corner of a framebuffer of its size bucket; those used last are kept within
        :attr:`window_bytes`."""
        with self.ctx:  # its own context (another renderer's may be current)
            return self._render_window(scene, ndc, size, aspect, cull)

    def window_mask(self, person) -> np.ndarray:
        """How much of each pixel of the window drawn last by :meth:`render_window` shows ``person`` (a character,
        with what they wear and carry) as the nearest surface: HxW float32, 0-1, soft at the edges as the MSAA
        samples fall. For the camera model's motion blur (only the moving person is smeared). Call it right after
        that window, from the same thread."""
        with self.ctx:  # its own context (another renderer's may be current)
            return self._window_mask(person)

    def _window_mask(self, person) -> np.ndarray:
        if self._last_window is None:
            raise RuntimeError("window_mask needs a window drawn first")
        (fbo, resolve), (width, height), view_proj, eye, t = self._last_window
        mgl, ctx = self._moderngl, self.ctx
        objects, skinned = self._mask_programs()
        fbo.use()
        ctx.viewport = viewport = (0, 0, width, height)  # (the window's corner of its framebuffer)
        fbo.depth_mask = False  # keep the window's depth: the clear below only clears the colour
        try:
            fbo.clear(0.0, 0.0, 0.0, 0.0, viewport=viewport)
            ctx.enable(mgl.DEPTH_TEST)
            ctx.depth_func = "<="  # the person meets their own depth where they are in front
            if person.current_visible:
                model, _ = person.matrices(eye)
                skinned["u_mvp"].write(gl_bytes(view_proj @ model))
                self._joint_texture(person, reuse=True).use(1)
                overrides = getattr(person, "texture_overrides", None)
                for primitive in (*person.skinned_model.primitives, *getattr(person, "worn", ())):
                    _vao, texture = self._skinned_primitive(primitive)
                    if overrides:
                        texture = overrides.get(primitive.name, texture)
                    texture = texture.at_time(t)
                    self._gl_texture(base_copy(texture) if is_large(texture) else texture).use(0)  # (its coverage)
                    self._probe_vao(primitive, skinned).render(mgl.TRIANGLES)
                for prop in getattr(person, "props", ()):
                    if prop.current_visible:
                        objects["u_mvp"].write(gl_bytes(view_proj @ prop.matrices(eye)[0]))
                        self._gl_texture(prop.current_texture or self._white).use(0)
                        self._probe_vao(prop.mesh, objects).render(mgl.TRIANGLES)
        finally:
            ctx.depth_func = "<"
            fbo.depth_mask = True
        if fbo is not resolve:
            ctx.copy_framebuffer(resolve, fbo)
        data = resolve.read(viewport=viewport, components=1, alignment=1)
        mask = np.frombuffer(data, dtype=np.uint8).reshape(height, width)
        return np.flipud(mask).astype(np.float32) / 255.0  # OpenGL rows start at the bottom

    def _mask_programs(self):
        if self._masks is None:
            self._probe_programs()  # (its table of vertex arrays is shared)
            objects = self.ctx.program(vertex_shader=_OBJECT_VS, fragment_shader=_MASK_FS)
            skinned = self.ctx.program(vertex_shader=_SKINNED_VS, fragment_shader=_MASK_FS)
            objects["u_texture"].value = skinned["u_texture"].value = 0
            skinned["u_joints"].value = 1
            self._masks = (objects, skinned)
        return self._masks

    def _render(self, scene: Scene) -> np.ndarray:
        self._frame_id += 1
        self._details.next_frame()  # (the windows drawn after it belong to its frame)
        camera = scene.camera
        view_proj = camera.projection_matrix(self.aspect) @ camera.view_matrix()
        part = (camera.aspect, camera.ndc) if isinstance(camera, SubCamera) else (None, None)  # a window's background
        image = self._draw_scene(scene, (self._draw_fbo, self._resolve_fbo), (self.width, self.height), view_proj,
                                 *part, cull=self.cull_view)
        self._drawn = (view_proj, camera.current_position)
        self._frames += 1
        if self._frames % 300 == 0:
            self._collect_garbage()
        return image

    def _render_window(self, scene: Scene, ndc, size, aspect: float | None, cull: bool) -> np.ndarray:
        width, height = (int(v) for v in size)
        crop = crop_matrix(*ndc)
        camera = scene.camera
        view_proj = crop @ camera.projection_matrix(self.aspect if aspect is None else float(aspect)) \
            @ camera.view_matrix()
        target = self._window_target(width, height)
        self._last_window = None
        image = self._draw_scene(scene, target, (width, height), view_proj, aspect=aspect, ndc=ndc, cull=cull,
                                 window=True)
        self._last_window = (target, (width, height), view_proj, camera.current_position, scene.time)
        return image

    def _draw_scene(self, scene: Scene, target, size, view_proj: np.ndarray, aspect: float | None = None,
                    ndc=None, cull: bool = False, window: bool = False) -> np.ndarray:
        """Draw ``scene`` with ``view_proj`` into ``target`` = (draw framebuffer, resolve framebuffer) of ``size``
        and read it back. ``ndc``: the window's rectangle of the whole picture, whose width / height is ``aspect``
        (``None``: the whole view); ``cull``: leave out what lies outside the frustum; ``window``: a window of the
        frame drawn last (its joint palettes are on the GPU already)."""
        ctx = self.ctx
        if not self.linear:  # the stored values as they are: nothing is encoded
            return self._draw_scene_srgb(scene, target, size, view_proj, aspect, ndc, cull, window)
        ctx.enable_direct(FRAMEBUFFER_SRGB)  # encode what is drawn: shading and blending happen in linear light
        try:
            return self._draw_scene_srgb(scene, target, size, view_proj, aspect, ndc, cull, window)
        finally:
            ctx.disable_direct(FRAMEBUFFER_SRGB)  # (coverage masks are drawn and resolved as they are)

    def _draw_scene_srgb(self, scene: Scene, target, size, view_proj: np.ndarray, aspect, ndc, cull: bool,
                         window: bool) -> np.ndarray:
        ctx = self.ctx
        fbo, resolve = target
        width, height = size
        fbo.use()
        ctx.viewport = viewport = (0, 0, width, height)  # (a window's corner of its framebuffer)
        background = scene.background
        if callable(background):
            background = background(scene.time)
        bg_texture = None
        if isinstance(background, (tuple, list)) or (isinstance(background, str) and background.startswith("#")):
            r, g, b = self._colour(background)  # a clear is encoded like a drawing
            fbo.clear(b, g, r, 1.0, viewport=viewport)
        else:
            bg_texture = as_texture(background)
            fbo.clear(0.0, 0.0, 0.0, 1.0, viewport=viewport)

        if bg_texture is not None:
            ctx.disable(self._moderngl.DEPTH_TEST)
            image_aspect = bg_texture.aspect
            frame_aspect = self.aspect if aspect is None else float(aspect)
            wide = image_aspect > frame_aspect
            scale = (frame_aspect / image_aspect, 1.0) if wide else (1.0, image_aspect / frame_aspect)
            self._bg_program["u_uv_scale"].value = scale  # "cover": crop instead of stretching
            left, bottom, right, top = (-1.0, -1.0, 1.0, 1.0) if ndc is None else ndc
            self._bg_program["u_ndc_scale"].value = ((right - left) / 2, (top - bottom) / 2)
            self._bg_program["u_ndc_offset"].value = ((right + left) / 2, (top + bottom) / 2)
            self._gl_texture(bg_texture.at_time(scene.time)).use(0)
            self._bg_vao.render(self._moderngl.TRIANGLES, vertices=3)

        ctx.enable(self._moderngl.DEPTH_TEST)
        ctx.disable(self._moderngl.CULL_FACE)
        eye = scene.camera.current_position
        high = _check_quality(self.window_quality if window else self.quality) == "high"
        uniforms = _scene_uniforms(scene, high, self.linear)
        _set_uniforms(self._program, uniforms)

        objects, characters = [], []
        for obj in scene:
            if obj.current_visible:
                if getattr(obj, "skinned_model", None) is not None:
                    characters.append(obj)
                    # what they wear and carry (see characters.accessories)
                    objects += [prop for prop in getattr(obj, "props", ()) if prop.current_visible]
                else:
                    objects.append(obj)
        shadows = [obj for obj in characters if getattr(obj, "shadow", False)]
        if cull:
            stats = {"objects": len(objects), "people": len(characters), "shadows": len(shadows)}
            objects, characters, shadows = self._in_frustum(frustum_planes(view_proj), eye, objects, characters,
                                                            shadows)
            stats.update(objects_drawn=len(objects), people_drawn=len(characters), shadows_drawn=len(shadows))
            if window:
                self.window_stats = stats
        for obj in objects:
            self._draw(obj, view_proj, eye)
        if characters or shadows:
            right = np.asarray(scene.camera.view_matrix(), np.float64)[0, :3]  # the camera's right in the scene
            self._draw_characters(characters, view_proj, eye, uniforms, scene.time, shadows, fbo, reuse=window,
                                  mapped=high, picture=(size, right / max(np.linalg.norm(right), 1e-12)))

        if fbo is not resolve:
            ctx.copy_framebuffer(resolve, fbo)
        data = resolve.read(viewport=viewport, components=3, alignment=1)
        image = np.frombuffer(data, dtype=np.uint8).reshape(height, width, 3)
        return cv2.flip(image, 0)  # OpenGL rows start at the bottom

    def _window_target(self, width: int, height: int) -> tuple:
        """(draw framebuffer, resolve framebuffer) for a window of ``width`` x ``height`` pixels, with the view's
        MSAA: of the size bucket (:func:`_bucket`) it falls in, drawn in its bottom-left corner. The framebuffers
        used last are kept while they take at most :attr:`window_bytes` together (the one in use always)."""
        limit = self.max_window
        if not (0 < width <= limit and 0 < height <= limit):
            raise ValueError(f"a window of {width}x{height} pixels cannot be drawn (at most {limit} a side)")
        key = (min(_bucket(width), limit), min(_bucket(height), limit))
        entry = self._window_targets.pop(key, None)
        if entry is None:
            ctx, samples = self.ctx, self._samples
            size = key[0] * key[1] * (4 + (8 * samples if samples else 4))  # colour, depth (and their samples)
            while self._window_targets and self.window_target_bytes + size > self.window_bytes:
                self._release_target(self._window_targets.pop(next(iter(self._window_targets))))
            colour = self._colour_buffer(key)
            depth = None if samples else ctx.depth_renderbuffer(key)
            resolve = ctx.framebuffer(color_attachments=[colour], depth_attachment=depth)
            buffers = [colour] + ([depth] if depth is not None else [])
            if samples:
                ms_colour = self._colour_buffer(key, samples)
                ms_depth = ctx.depth_renderbuffer(key, samples=samples)
                draw = ctx.framebuffer(color_attachments=[ms_colour], depth_attachment=ms_depth)
                buffers += [ms_colour, ms_depth]
            else:
                draw = resolve
            entry = (draw, resolve, buffers, size)
        self._window_targets[key] = entry  # the most recently used last
        return entry[0], entry[1]

    @property
    def window_target_bytes(self) -> int:
        """Bytes the window framebuffers kept take (colour and depth, with their MSAA samples)."""
        return sum(entry[3] for entry in self._window_targets.values())

    @staticmethod
    def _release_target(entry: tuple) -> None:
        draw, resolve, buffers, _size = entry
        for resource in {id(draw): draw, id(resolve): resolve}.values():
            resource.release()
        for buffer in buffers:
            buffer.release()

    def _draw(self, obj: SceneObject, view_proj: np.ndarray, camera_position) -> None:
        model, normal_matrix = obj.matrices(camera_position)
        u = self._uniforms
        u["u_mvp"].write(gl_bytes(view_proj @ model))
        u["u_normal_matrix"].write(gl_bytes(normal_matrix))
        u["u_color"].value = self._colour(obj.current_color)
        u["u_lit"].value = bool(obj.lit)
        self._gl_texture(obj.current_texture or self._white).use(0)
        self._vao(obj.mesh).render(self._moderngl.TRIANGLES)

    def _in_frustum(self, planes, eye, objects, characters, shadows) -> tuple[list, list, list]:
        """The objects, characters and blob shadows that may show in the frustum of ``planes`` (see :mod:`.frustum`:
        conservative, so leaving out the others changes no pixel)."""
        bounds = self._bounds
        if objects:
            outside = boxes_outside(planes, *bounds.boxes(objects, eye))
            objects = [obj for obj, out in zip(objects, outside) if not out]
        if characters:
            centres, radii, owners = bounds.spheres(characters, eye)
            seen = np.bincount(owners[~spheres_outside(planes, centres, radii)], minlength=len(characters)) > 0
            characters = [obj for obj, show in zip(characters, seen) if show]
        if shadows:  # the rectangles _draw_shadows draws
            cx, cz, rx, rz = np.array([obj.shadow_ellipse for obj in shadows], dtype=np.float64).T
            lift, flat = np.full_like(cx, SHADOW_LIFT), np.zeros_like(cx)
            boxes = world_boxes([obj.matrices(eye)[0] for obj in shadows], np.stack([cx, lift, cz], axis=1),
                                np.stack([rx, flat, rz], axis=1))
            shadows = [obj for obj, out in zip(shadows, boxes_outside(planes, *boxes)) if not out]
        return objects, characters, shadows

    def _draw_characters(self, characters, view_proj, camera_position, uniforms: dict, t: float, shadows=None,
                         fbo=None, reuse: bool = False, mapped: bool = False, picture=None) -> None:
        """GPU-skinned characters (opaque, alpha-tested), then blob shadows (those of ``characters`` that have one,
        or of ``shadows``) blended over the scene in ``fbo`` (the view's by default). ``reuse``: use the joint
        palettes this frame uploaded where the characters were not posed again since (windows of the frame).
        ``mapped``: draw mesh parts with tangents and maps with them (the high quality). ``picture``: ``(size, the
        camera's right)`` of the picture, for the heads' detail textures (``None``: never)."""
        if self._details_linear != self.linear:  # detail textures give their pixels up once on the GPU: read anew
            for texture in self._details.clear():
                self._forget_texture(texture)
            self._details_linear = self.linear
        prog = self._skinned()
        _set_uniforms(prog, uniforms)
        u, triangles = self._skinned_uniforms, self._moderngl.TRIANGLES
        mapped_program = None
        for obj in characters:
            model, normal_matrix = obj.matrices(camera_position)
            mvp, colour = gl_bytes(view_proj @ model), self._colour(obj.current_color)
            u["u_mvp"].write(mvp)
            u["u_normal_matrix"].write(gl_bytes(normal_matrix))
            u["u_color"].value = colour
            u["u_lit"].value = bool(obj.lit)
            self._joint_texture(obj, reuse).use(1)
            overrides = getattr(obj, "texture_overrides", None)
            own_maps = getattr(obj, "maps_overrides", None)  # this person's own packed maps, by part
            eye = None  # pixels between this person's eyes in the picture: worked out when it matters
            later = []  # the parts drawn with their maps
            for primitive in (*obj.skinned_model.primitives, *getattr(obj, "worn", ())):
                vao, texture = self._skinned_primitive(primitive)
                override = overrides.get(primitive.name) if overrides else None
                if override is not None:
                    texture = override.at_time(t)
                detail = None
                prepared = override is None and primitive.detail is not None
                if prepared or is_large(texture):  # a picture may draw it larger than 1024 or not
                    if eye is None:
                        eye = 0.0 if picture is None else self._eye_pixels(obj, view_proj, camera_position,
                                                                           picture)
                    near = eye >= self.detail_from
                    if near and prepared:
                        detail = self._details.get(primitive, eye, self.detail_wait, self.detail_heads)
                        if detail is not None:
                            texture = detail[0]
                    elif near and mapped and _has_maps(primitive) and getattr(primitive.detail, "maps", None) \
                            is not None:  # a face of its own at 2048: the head's 2048 maps beside it
                        detail = self._details.get(primitive, eye, self.detail_wait, self.detail_heads, maps_only=True)
                    elif not near and is_large(texture):
                        texture = base_copy(texture)
                if mapped and _has_maps(primitive):
                    maps = own_maps.get(primitive.name) if own_maps else None
                    if maps is None and detail is not None:
                        maps = detail[1]
                    photo = None  # a face from photos without maps of its own: where it is
                    if override is not None and hasattr(obj, "photo_mask"):
                        photo = obj.photo_mask(primitive)
                    later.append((primitive, texture, maps, photo))
                    continue
                self._gl_texture(texture.at_time(t)).use(0)
                if override is not None and uniforms["u_hemisphere"] and hasattr(obj, "photo_mask") and \
                        obj.photo_mask(primitive) is not None:  # a face from photos on a head without maps
                    prog["u_hemisphere"].value = uniforms["u_hemisphere"] * PHOTO_HEMISPHERE
                    vao.render(triangles)
                    prog["u_hemisphere"].value = uniforms["u_hemisphere"]
                    continue
                vao.render(triangles)
            if not later:
                continue
            if mapped_program is None:
                mapped_program = self._mapped()
                _set_uniforms(mapped_program, uniforms)
                _set_uniforms(mapped_program, {"u_eye": tuple(float(v) for v in camera_position),
                                               "u_specular": float(self.specular),
                                               "u_normal_strength": float(self.normal_strength),
                                               "u_photo_hemisphere": float(PHOTO_HEMISPHERE),
                                               "u_photo_level": float(PHOTO_LEVEL)})
            m = self._mapped_uniforms
            m["u_mvp"].write(mvp)
            m["u_model"].write(gl_bytes(model))
            m["u_normal_matrix"].write(gl_bytes(normal_matrix))
            m["u_color"].value = colour
            m["u_lit"].value = bool(obj.lit)
            for primitive, texture, these_maps, photo in later:
                vao, maps = self._mapped_primitive(primitive)
                self._gl_texture(texture.at_time(t)).use(0)
                self._gl_texture(these_maps or maps).use(2)  # (the detail maps, or the person's own)
                self._gl_texture(photo or self._no_photo).use(3)
                vao.render(triangles)
        for texture in self._details.trim(self.detail_heads):  # what is no longer kept leaves the GPU at once
            self._forget_texture(texture)
        if shadows is None:
            shadows = [obj for obj in characters if getattr(obj, "shadow", False)]
        if shadows:
            self._draw_shadows(shadows, view_proj, camera_position, uniforms, fbo)

    def _draw_shadows(self, characters, view_proj, camera_position, uniforms: dict, fbo=None) -> None:
        """Soft dark ellipses on the floor in one draw call: depth-tested, blended, without writing depth."""
        mgl, ctx, fbo = self._moderngl, self.ctx, self._draw_fbo if fbo is None else fbo
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

    def visible_pixels(self, people) -> list[int]:
        """How many pixels of each of ``people`` (characters posed for the time of the frame drawn last, in its scene
        or not) would show in that frame, in front of what it shows: 0 when hidden behind it or out of the picture,
        or when the person is invisible. Each is counted on its own, with what it wears and the middle of its blob
        shadow; what textures cut out (hair) counts as if it were there. For changes to a scene that nobody should
        see (``scenes.created_cast``): call it after :meth:`render`, from the same thread."""
        with self.ctx:  # its own context (another renderer's may be current)
            return self._visible_pixels(people)

    def _visible_pixels(self, people) -> list[int]:
        if self._drawn is None:
            raise RuntimeError("visible_pixels needs a frame drawn first")
        mgl, ctx, fbo = self._moderngl, self.ctx, self._draw_fbo
        view_proj, camera_position = self._drawn
        objects, skinned, shadow = self._probe_programs()
        fbo.use()
        ctx.viewport = (0, 0, self.width, self.height)
        ctx.enable(mgl.DEPTH_TEST)
        ctx.depth_func = "<="  # a person drawn in the frame meets their own depth
        fbo.color_mask = (False, False, False, False)
        fbo.depth_mask = False
        counts = []
        try:
            for person in people:
                if not person.current_visible:
                    counts.append(0)
                    continue
                query = self._probe_query
                with query:
                    model, _ = person.matrices(camera_position)
                    skinned["u_mvp"].write(gl_bytes(view_proj @ model))
                    self._joint_texture(person).use(1)
                    for primitive in (*person.skinned_model.primitives, *getattr(person, "worn", ())):
                        self._probe_vao(primitive, skinned).render(mgl.TRIANGLES)
                    for prop in getattr(person, "props", ()):
                        if prop.current_visible:
                            prop_model, _ = prop.matrices(camera_position)
                            objects["u_mvp"].write(gl_bytes(view_proj @ prop_model))
                            self._probe_vao(prop.mesh, objects).render(mgl.TRIANGLES)
                    if getattr(person, "shadow", False):
                        self._probe_shadow(person, view_proj, camera_position, shadow)
                counts.append(int(query.samples))
        finally:
            fbo.color_mask = (True, True, True, True)
            fbo.depth_mask = True
            ctx.depth_func = "<"
        samples = max(1, fbo.samples)
        return [-(-count // samples) for count in counts]  # whole pixels, rounded up

    def prepare(self, *things) -> bool:
        """Upload one of what drawing ``things`` needs and is not on the GPU yet: a character's mesh parts, textures
        and what it wears (in the scene or not yet), or a :class:`Texture`. True when all of it is there. Called once
        a frame before someone comes into the scene, it spreads the uploads over those frames, so that the frame they
        come in has none of them left to do."""
        with self.ctx:  # its own context (another renderer's may be current)
            return self._prepare(*things)

    def _prepare(self, *things) -> bool:
        for thing in things:
            if isinstance(thing, Texture):
                if not self._uploaded(thing):
                    self._upload_step(thing)
                    return False
            elif not self._prepare_character(thing):
                return False
        return True

    def _prepare_character(self, obj) -> bool:
        t = getattr(obj, "_time", 0.0)
        overrides = getattr(obj, "texture_overrides", None) or {}
        mapped = "high" in (self.quality, self.window_quality)
        for primitive in (*obj.skinned_model.primitives, *getattr(obj, "worn", ())):
            entry = self._primitives.get(id(primitive))
            if entry is None or entry[0]() is not primitive or entry[4] is not primitive.texture:
                self._skinned_primitive(primitive)
                return False
            texture = overrides.get(primitive.name, entry[5]).at_time(t)
            if is_large(texture):  # drawn at its 1024 level until a picture needs more
                texture = base_copy(texture)
            if not self._uploaded(texture):
                self._upload_step(texture)
                return False
            if mapped and _has_maps(primitive):
                if entry[8] is None or entry[6] is not primitive.maps:
                    self._mapped_primitive(primitive)
                    return False
                if not self._uploaded(entry[7]):
                    self._upload_step(entry[7])
                    return False
        for prop in getattr(obj, "props", ()):
            texture = as_texture(prop.texture) if not callable(prop.texture) else prop.current_texture
            texture = (texture or self._white).at_time(t)
            if not self._uploaded(texture):
                self._upload_step(texture)
                return False
            entry = self._meshes.get(id(prop.mesh))
            if entry is None or entry[0]() is not prop.mesh:
                self._vao(prop.mesh)
                return False
        return True

    def _uploaded(self, texture: Texture) -> bool:
        entry = self._textures.get(texture.id)
        return entry is not None and entry[2] == texture.version and entry[3] \
            and entry[4] == (bool(texture.srgb) and self.linear)

    def _colour(self, color) -> tuple[float, float, float]:
        """A colour (sRGB, as scenes give them) for the shaders: in linear light, or as stored (:attr:`linear`)."""
        if self.linear:
            return color_to_linear(color)
        r, g, b = parse_color(color)
        return r / 255.0, g / 255.0, b / 255.0

    def _probe_programs(self):
        if self._probes is None:
            ctx = self.ctx
            self._probes = (ctx.program(vertex_shader=_OBJECT_VS, fragment_shader=_PROBE_FS),
                            ctx.program(vertex_shader=_SKINNED_VS, fragment_shader=_PROBE_FS),
                            ctx.program(vertex_shader=_SHADOW_VS, fragment_shader=_PROBE_FS))
            self._probes[1]["u_joints"].value = 1
            self._probe_vaos: dict[tuple[int, int], tuple] = {}
            self._probe_query = ctx.query(samples=True)  # one, used again for each person
            self._probe_shadow_vbo = ctx.buffer(reserve=6 * 3 * 4, dynamic=True)
            self._probe_shadow_vao = ctx.vertex_array(self._probes[2], [(self._probe_shadow_vbo, "3f", "in_position")],
                                                      skip_errors=True)
        return self._probes

    def _probe_vao(self, mesh, program):
        """A vertex array of ``mesh`` (a skinned primitive or a mesh) for a probe program, sharing the buffers the
        drawing uses."""
        key = (id(mesh), program.glo)
        entry = self._probe_vaos.get(key)
        if entry is not None and entry[0]() is mesh:
            return entry[1]
        if hasattr(mesh, "joints"):
            _ref, _vao, vbo, ibo, *_ = self._primitive_entry(mesh)
            layout = (vbo, "3f 3f 2f 4f 4f", "in_position", "in_normal", "in_uv", "in_joints", "in_weights")
        else:
            self._vao(mesh)
            _ref, _vao, vbo, ibo = self._meshes[id(mesh)]
            layout = (vbo, "3f 3f 2f", "in_position", "in_normal", "in_uv")
        if entry is not None:
            entry[1].release()
        vao = self.ctx.vertex_array(program, [layout], index_buffer=ibo, skip_errors=True)
        self._probe_vaos[key] = (weakref.ref(mesh), vao)
        return vao

    def _probe_shadow(self, person, view_proj, camera_position, program) -> None:
        model = person.matrices(camera_position)[0][:3]
        cx, cz, rx, rz = (float(v) for v in person.shadow_ellipse)
        local = np.ones((6, 4))
        local[:, 0] = cx + _SHADOW_CORNERS[:, 0] * rx * _PROBE_SHADOW
        local[:, 1] = SHADOW_LIFT
        local[:, 2] = cz + _SHADOW_CORNERS[:, 1] * rz * _PROBE_SHADOW
        self._probe_shadow_vbo.write((local @ model.T).astype("f4").tobytes())
        program["u_view_proj"].write(gl_bytes(view_proj))
        self._probe_shadow_vao.render(self._moderngl.TRIANGLES, vertices=6)

    def _skinned(self):
        if self._skinned_program is None:
            prog = self.ctx.program(vertex_shader=_SKINNED_VS, fragment_shader=_OBJECT_FS)
            prog["u_texture"].value = 0
            prog["u_joints"].value = 1
            self._skinned_uniforms = {name: prog[name] for name in ("u_mvp", "u_normal_matrix", "u_color", "u_lit")}
            self._skinned_program = prog
        return self._skinned_program

    def _mapped(self):
        """The program of mesh parts drawn with their maps (see ``_MAPPED_FS``), compiled when first needed."""
        if self._mapped_program is None:
            prog = self.ctx.program(vertex_shader=_MAPPED_VS, fragment_shader=_MAPPED_FS)
            prog["u_texture"].value = 0
            prog["u_joints"].value = 1
            prog["u_maps"].value = 2
            prog["u_photo"].value = 3
            self._mapped_uniforms = {name: prog[name] for name in ("u_mvp", "u_model", "u_normal_matrix", "u_color",
                                                                   "u_lit")}
            self._mapped_program = prog
        return self._mapped_program

    def _primitive_entry(self, primitive) -> list:
        """``[weakref, vao, vbo, ibo, texture spec, Texture, maps spec, maps Texture, mapped vao, tangent vbo]`` of a
        character's mesh part: its buffers uploaded once and shared by every character using it (the texture is
        resolved by :meth:`_skinned_primitive`, the maps and tangents by :meth:`_mapped_primitive`)."""
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
            entry = [weakref.ref(primitive), vao, vbo, ibo, _UNSET, None, _UNSET, None, None, None]
            self._primitives[key] = entry
        return entry

    def _mapped_primitive(self, primitive):
        """``(vao, maps Texture)`` of a mesh part with tangents and maps, for the mapped program: the tangents in a
        buffer of their own beside the shared vertex buffer, uploaded once."""
        entry = self._primitive_entry(primitive)
        if entry[8] is None:
            tangents = self.ctx.buffer(np.ascontiguousarray(primitive.tangents, dtype="f4").tobytes())
            entry[8] = self.ctx.vertex_array(
                self._mapped(),
                [(entry[2], "3f 3f 2f 4f 4f", "in_position", "in_normal", "in_uv", "in_joints", "in_weights"),
                 (tangents, "4f", "in_tangent")],
                index_buffer=entry[3], skip_errors=True,
            )
            entry[9] = tangents
        if entry[6] is not primitive.maps:
            try:
                maps = as_data_texture(primitive.maps)
            except (OSError, ValueError) as exc:  # a damaged file: flat
                print(f"[renderer] cannot use a character's maps, drawing them flat: {exc}")
                maps = None
            entry[6], entry[7] = primitive.maps, maps or self._flat
        return entry[8], entry[7]

    def _skinned_primitive(self, primitive):
        """``(vao, Texture)`` of a character's mesh part, uploaded once and shared by every character using it."""
        entry = self._primitive_entry(primitive)
        if entry[4] is not primitive.texture:
            try:
                texture = _character_texture(primitive.texture)
            except (OSError, ValueError) as exc:  # e.g. a damaged file in the character cache: draw it plain
                print(f"[renderer] cannot use a character texture, drawing it untextured: {exc} "
                      "(bifold-characters fetch repairs damaged files)")
                texture = None
            entry[4], entry[5] = primitive.texture, texture or self._white
        return entry[1], entry[5]

    def _eye_pixels(self, obj, view_proj, camera_position, picture) -> float:
        """How many pixels apart ``obj``'s eyes are (a frontal face's) in the picture ``picture`` = (size, the camera's
        right): from its head bone (else the top of its model)."""
        skeleton = obj.skinned_model.skeleton
        bone = self._head_bones.get(skeleton)
        if bone is None:
            found = skeleton.find("Head")
            bone = self._head_bones[skeleton] = -1 if found is None else found
        local = obj.pose[bone, :3, 3] if bone >= 0 else np.array([0.0, obj.skinned_model.height, 0.0])
        model = obj.matrices(camera_position)[0]
        return eye_pixels(model[:3, :3] @ local + model[:3, 3], view_proj, *picture)

    def _forget_texture(self, texture: Texture) -> None:
        """Free ``texture``'s GPU copy now (it is uploaded again if it is drawn again)."""
        entry = self._textures.pop(texture.id, None)
        if entry is not None:
            entry[1].release()

    def memory(self) -> dict:
        """What the textures on the GPU take, in bytes (their mipmaps included): ``textures`` all of them,
        ``detail`` the heads' 2048 textures among them (``detail_heads``: how many heads have them), and how many
        detail texture pairs were read so far (``detail_loads``) and refused to a picture because the heads that hold
        them need them more (``detail_refused``); ``detail_cpu`` what the detail textures take in memory besides
        (their pixels until they are on the GPU)."""
        def size(entry) -> int:
            gl_tex = entry[1]
            pixels = gl_tex.width * gl_tex.height * 4
            return pixels * 4 // 3 if entry[3] else pixels

        detail_ids = {t.id for t in self._details.textures()}
        total = sum(size(e) for e in self._textures.values())
        detail = sum(size(e) for key, e in self._textures.items() if key in detail_ids)
        return {"textures": total, "detail": detail, "detail_heads": len(self._details),
                "detail_loads": self._details.loads, "detail_refused": self._details.refused,
                "detail_cpu": self._details.cpu_bytes()}

    def _joint_texture(self, obj, reuse: bool = False):
        """The character's joint-palette texture (one per character, rewritten every frame). With ``reuse`` it is
        not written again when it holds this very palette from this frame (a window of the frame)."""
        palette = obj.skin_palette
        entry = self._joints.get(id(obj))
        if entry is None or entry[0]() is not obj or entry[1].height != len(palette):
            if entry is not None:
                entry[1].release()
            gl_tex = self.ctx.texture((3, len(palette)), 4, dtype="f4")
            gl_tex.filter = (self._moderngl.NEAREST, self._moderngl.NEAREST)
            entry = [weakref.ref(obj), gl_tex, None, None]  # (..., palette written, frame it was written in)
            self._joints[id(obj)] = entry
        elif reuse and entry[2] is palette and entry[3] == self._frame_id:
            return entry[1]
        entry[1].write(np.asarray(palette, dtype=np.float32)[:, :3, :].tobytes())
        entry[2], entry[3] = palette, self._frame_id
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

    def _gl_texture(self, texture: Texture, mipmaps: bool = True):
        """``texture`` on the GPU (uploaded when new or changed), with its mipmaps unless ``mipmaps`` is False (then
        they are built the next time it is asked for with them)."""
        entry = self._textures.get(texture.id)
        if entry is not None and entry[2] == texture.version and entry[4] == (bool(texture.srgb) and self.linear):
            if mipmaps and not entry[3]:
                self._build_mipmaps(entry)
            return entry[1]
        if texture.__dict__.get("pixels_dropped"):
            raise RuntimeError(f"{texture.name}: its pixels were given up once on the GPU (drop_after_upload)")
        prepared = texture.__dict__.get("_upload")  # laid out already (textures.prepare_upload)
        if prepared is not None and prepared[0] == texture.version:
            pixels = prepared[1]
        else:
            pixels = np.ascontiguousarray(np.flipud(texture.rgba)).tobytes()  # first row = bottom for OpenGL
        size, srgb = (texture.width, texture.height), bool(texture.srgb) and self.linear
        if entry is not None and entry[1].size == size and entry[4] == srgb:
            gl_tex = entry[1]
            gl_tex.write(pixels)
        else:
            if entry is not None:
                entry[1].release()
            gl_tex = self.ctx.texture(size, 4, pixels, internal_format=SRGB8_ALPHA8 if srgb else None)
            gl_tex.repeat_x = gl_tex.repeat_y = True
            try:
                gl_tex.anisotropy = 8.0
            except Exception:  # noqa: BLE001 - anisotropic filtering is optional
                pass
        gl_tex.filter = (self._moderngl.LINEAR, self._moderngl.LINEAR)  # until the mipmaps are built for the pixels
        texture._on_gpu = texture.version  # (prepare_upload keeps nothing for pixels uploaded already)
        entry = self._textures[texture.id] = [weakref.ref(texture), gl_tex, texture.version, False, srgb]
        if mipmaps:
            self._build_mipmaps(entry)
        return gl_tex

    def _build_mipmaps(self, entry: list) -> None:
        """The mipmaps of an uploaded texture: averaged in linear light on the CPU (:func:`~.textures.mip_chain`,
        made beforehand by ``prepare_upload`` or now) for a texture that does not change, else (a video) the
        driver's own."""
        gl_tex = entry[1]
        if gl_tex.width > 1 or gl_tex.height > 1:
            gl_tex.build_mipmaps()  # (allocates the levels: moderngl writes only levels that exist)
            texture = entry[0]()
            prepared = texture.__dict__.get("_upload") if texture is not None else None
            current = prepared is not None and prepared[0] == entry[2] and prepared[2] is not None
            if texture is not None and texture.version == entry[2] and (current or not _changes(texture)) \
                    and self.linear:  # (on stored values the driver's, as before)
                for level, pixels in enumerate(upload_levels(texture), start=1):
                    gl_tex.write(pixels, level=level)
            gl_tex.filter = (self._moderngl.LINEAR_MIPMAP_LINEAR, self._moderngl.LINEAR)
        entry[3] = True
        texture = entry[0]()
        if texture is not None and texture.__dict__.get("drop_after_upload") and texture.version == entry[2]:
            drop_pixels(texture)  # (a head's detail texture: read again if it is needed again)

    def _upload_step(self, texture: Texture) -> None:
        """One step of putting ``texture`` on the GPU (for :meth:`prepare`): its pixels, then (the next time) its
        mipmaps; both at once for a small texture (for a large one on a software renderer each takes a few
        milliseconds or more: 5 to 25 ms the mipmaps of 1024 x 1024 here)."""
        entry = self._textures.get(texture.id)
        small = texture.width * texture.height <= 256 * 256
        self._gl_texture(texture, mipmaps=small or (entry is not None and entry[2] == texture.version))

    def _collect_garbage(self) -> None:
        """Free GPU resources of meshes and textures that no longer exist."""
        with self.ctx:  # its own context (another renderer's may be current)
            self._free_unused()

    def _free_unused(self) -> None:
        for key, (ref, *_rest) in list(self._meshes.items()):
            if ref() is None:
                self._release_mesh(key)
        for key, (ref, gl_tex, *_rest) in list(self._textures.items()):
            if ref() is None:
                gl_tex.release()
                del self._textures[key]
        for key, entry in list(self._primitives.items()):
            if entry[0]() is None:
                self._release_primitive(key)
        for key, (ref, gl_tex, *_written) in list(self._joints.items()):
            if ref() is None:
                gl_tex.release()
                del self._joints[key]
        for key, (ref, vao) in list(getattr(self, "_probe_vaos", {}).items()):
            if ref() is None:
                vao.release()
                del self._probe_vaos[key]
        self._bounds.forget_unused()

    def _release_mesh(self, key: int) -> None:
        _ref, vao, vbo, ibo = self._meshes.pop(key)
        for resource in (vao, vbo, ibo):
            resource.release()

    def _release_primitive(self, key: int) -> None:
        _ref, vao, vbo, ibo, _spec, _texture, _maps_spec, _maps, mapped_vao, tangents = self._primitives.pop(key)
        for resource in (vao, vbo, ibo, mapped_vao, tangents):
            if resource is not None:
                resource.release()

    def release(self) -> None:
        """Free all GPU resources (the renderer cannot draw afterwards)."""
        with self.ctx:  # its own context (another renderer's may be current)
            self._release_all()
        self.ctx.release()

    def _release_all(self) -> None:
        for key in list(self._meshes):
            self._release_mesh(key)
        for key in list(self._primitives):
            self._release_primitive(key)
        for _ref, gl_tex, *_written in self._joints.values():
            gl_tex.release()
        self._joints.clear()
        self._details = DetailCache()  # (their GPU copies are among the textures released below)
        if self._mapped_program is not None:
            self._mapped_program.release()
            self._mapped_program = None
        if self._probes is not None:
            for _ref, vao in self._probe_vaos.values():
                vao.release()
            self._probe_vaos.clear()
            self._probe_shadow_vao.release()
            self._probe_shadow_vbo.release()
            for program in self._probes:
                program.release()
            self._probes = None
        if self._masks is not None:
            for program in self._masks:
                program.release()
            self._masks = None
        if self._shadow_vbo is not None:
            self._shadow_vao.release()
            self._shadow_vbo.release()
            self._shadow_vbo = self._shadow_vao = None
        for _ref, gl_tex, *_rest in self._textures.values():
            gl_tex.release()
        self._textures.clear()
        for entry in self._window_targets.values():
            self._release_target(entry)
        self._window_targets.clear()
        self._last_window = None


QUALITIES = ("high", "low")


def _check_quality(quality: str) -> str:
    if quality not in QUALITIES:
        raise ValueError(f"unknown quality {quality!r}: choose one of {', '.join(QUALITIES)}")
    return quality


def _has_maps(primitive) -> bool:
    """Whether a mesh part is drawn with its maps at the high quality: it has tangents and a packed map."""
    return getattr(primitive, "tangents", None) is not None and getattr(primitive, "maps", None) is not None


def _scene_uniforms(scene: Scene, high: bool = True, linear: bool = True) -> dict:
    """Light and fog uniform values shared by every surface program, in linear light (see the module docs), or as
    stored (``linear`` False: the renderer's values before linear light); ``high``: the high quality's light (the
    ambient light from above, as the scene's ``hemisphere`` says)."""
    light = np.asarray(scene.light_direction, dtype=np.float64)
    light = light / max(np.linalg.norm(light), 1e-9)
    fog = scene.fog

    def colour(c):
        return color_to_linear(c) if linear else tuple(v / 255.0 for v in parse_color(c))

    values = {
        "u_light_dir": tuple(light),
        "u_light_color": colour(scene.light_color),
        "u_ambient": ambient_light(scene.ambient) if linear else float(np.clip(scene.ambient, 0.0, 1.0)),
        "u_hemisphere": float(np.clip(getattr(scene, "hemisphere", 0.0), 0.0, 1.0)) if high else 0.0,
        "u_fog_strength": float(np.clip(fog.strength, 0.0, 1.0)) if fog else 0.0,
        "u_shadow_gamma": 2.2 if linear else 1.0,
    }
    if fog:
        values["u_fog_color"] = colour(fog.color)
        values["u_fog_start"] = float(fog.start)
        values["u_fog_end"] = float(fog.end)
    return values


def ambient_light(level: float) -> float:
    """The light, 0-1, of a scene's ``ambient`` level, which is given as it looks: what lies in shadow is as dark,
    on screen, as ``level`` times what lies in full light (the level is decoded like an sRGB colour value)."""
    c = float(np.clip(level, 0.0, 1.0))
    return c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4


def _changes(texture: Texture) -> bool:
    """Whether ``texture``'s pixels change while it is drawn (a video, a picture updated): its mipmaps are then the
    driver's, which is quicker than averaging them on the CPU every time."""
    return isinstance(texture, VideoTexture) or texture.version > 0


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
