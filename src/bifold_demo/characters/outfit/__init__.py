"""Outfits for the 3D people: new clothes on an avatar's own body mesh and texture layout.

Component 2 of the character creator (see ``docs/research/outfit_textures.md``, section 3). Stages 0 and 1 are
deterministic; stage 2, a fabric made from a description, plugs into the same compositor as a
:class:`~.fills.Swatch`.

* Stage 0, :mod:`.regions`: every texel of an avatar's body texture is labelled skin, top, shirt, bottoms, shoes or
  other, from the bones that move it, its UV island and colour (k-means in Lab, the avatar's own skin colour from its
  face). The labels are cached per avatar and can be corrected by a small JSON file per avatar. The skin mask is for
  the face component (it matches the skin to a person's photos); outfits never write skin.
* :mod:`.surface`: cloth coordinates, where each clothes texel is on the body in metres (unrolled torso, sleeves,
  trouser legs and shoes), so patterns keep one size, run on across UV cuts and match front and back.
* Stage 1, :mod:`.fills`, :mod:`.patterns`, :mod:`.prompts`: what a region is filled with (a colour, a procedural
  pattern, an image swatch), and keywords that turn "navy pinstripe" or "white polka dots on black" into one.
* Stage 2, :mod:`.fabric`: a description ("blue paisley silk") made into seamless swatches by a text-to-image model
  (FLUX.2 [klein], made to tile by :mod:`.tiling`), checked and repaired by :mod:`.seamless`, with a fallback ladder
  down to the keyword fabric; :func:`fabric_fill` gives one as a :class:`Fabric` fill, whose spec keeps the image.
* :mod:`.tailor`: the compositor. It keeps the avatar's folds, seams and stitching (``new = fill x detail x
  shading``), and dresses a :class:`~..character.Character` in an :class:`Outfit` (``presets.json`` has ready ones)::

      from bifold_demo.characters.outfit import dress, fabric_fill, undress

      dress(person, "business navy")                                 # a preset
      dress(person, "red tartan jacket, white shirt, black trousers")  # a description
      dress(person, {"top": "olive camo", "shoes": "#3b2a1e"})        # region by region
      dress(person, {"top": fabric_fill("blue paisley silk")})        # a generated fabric
      undress(person)                                                # the avatar's own clothes again
"""

from .fabric import Fabric, FabricGenerator, FabricJob, FabricSwatch, PromptFilter, PromptRefused, fabric_fill
from .fills import Atlas, Fill, Keep, Pattern, Solid, Swatch, color, from_spec
from .patterns import PATTERNS
from .prompts import parse_fill, parse_outfit
from .regions import REGIONS, BodyRegions, body_regions
from .seamless import equalise_light, make_seamless, seam_score
from .tailor import Outfit, Tailor, dress, outfit_of, preset, presets, tailor_for, undress

__all__ = ["PATTERNS", "REGIONS", "Atlas", "BodyRegions", "Fabric", "FabricGenerator", "FabricJob", "FabricSwatch",
           "Fill", "Keep", "Outfit", "Pattern", "PromptFilter", "PromptRefused", "Solid", "Swatch", "Tailor",
           "body_regions", "color", "dress", "equalise_light", "fabric_fill", "from_spec", "make_seamless",
           "outfit_of", "parse_fill", "parse_outfit", "preset", "presets", "seam_score", "tailor_for", "undress"]
