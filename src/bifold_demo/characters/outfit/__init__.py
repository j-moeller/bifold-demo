"""Outfits for the 3D people: new clothes on an avatar's own body mesh and texture layout.

Component 2 of the character creator (see ``docs/research/outfit_textures.md``, section 3). This package has the
deterministic stages; the generative fabric (text to a seamless swatch) plugs into the same compositor later as a
:class:`~.fills.Swatch`.

* Stage 0, :mod:`.regions`: every texel of an avatar's body texture is labelled skin, top, shirt, bottoms, shoes or
  other, from the bones that move it, its UV island and colour (k-means in Lab, the avatar's own skin colour from its
  face). The labels are cached per avatar and can be corrected by a small JSON file per avatar. The skin mask is for
  the face component (it matches the skin to a person's photos); outfits never write skin.
* :mod:`.surface`: cloth coordinates, where each clothes texel is on the body in metres (unrolled torso, sleeves,
  trouser legs and shoes), so patterns keep one size, run on across UV cuts and match front and back.
* Stage 1, :mod:`.fills`, :mod:`.patterns`, :mod:`.prompts`: what a region is filled with (a colour, a procedural
  pattern, an image swatch), and keywords that turn "navy pinstripe" or "white polka dots on black" into one.
* :mod:`.tailor`: the compositor. It keeps the avatar's folds, seams and stitching (``new = fill x detail x
  shading``), and dresses a :class:`~..character.Character` in an :class:`Outfit` (``presets.json`` has ready ones)::

      from bifold_demo.characters.outfit import dress, undress

      dress(person, "business navy")                                 # a preset
      dress(person, "red tartan jacket, white shirt, black trousers")  # a description
      dress(person, {"top": "olive camo", "shoes": "#3b2a1e"})        # region by region
      undress(person)                                                # the avatar's own clothes again
"""

from .fills import Atlas, Fill, Keep, Pattern, Solid, Swatch, color, from_spec
from .patterns import PATTERNS
from .prompts import parse_fill, parse_outfit
from .regions import REGIONS, BodyRegions, body_regions
from .tailor import Outfit, Tailor, dress, outfit_of, preset, presets, tailor_for, undress

__all__ = ["PATTERNS", "REGIONS", "Atlas", "BodyRegions", "Fill", "Keep", "Outfit", "Pattern", "Solid", "Swatch",
           "Tailor", "body_regions", "color", "dress", "from_spec", "outfit_of", "parse_fill", "parse_outfit",
           "preset", "presets", "tailor_for", "undress"]
