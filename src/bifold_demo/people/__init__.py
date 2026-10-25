"""Images of people for the 3D scenes: generated with a text-to-image model, or cut out of photos.

Command line: ``bifold-people generate -n 20`` (see ``bifold-people --help``).
In a scene: ``people_for_scene()`` returns your library, or default people if it is empty.
"""

from __future__ import annotations

from .defaults import default_people
from .library import DEFAULT_FOLDER, PeopleLibrary, Person
from .prompts import PersonSpec, random_spec


def people_for_scene(folder=DEFAULT_FOLDER) -> list[Person]:
    """People from ``folder`` if it has any full-body ones, otherwise the default people."""
    people = PeopleLibrary(folder).people()
    if any(p.full_body for p in people):
        return people
    print(f"[people] {folder}/ has no people yet, using default people "
          "(make your own with: bifold-people generate -n 20)")
    return default_people() + people


__all__ = ["DEFAULT_FOLDER", "PeopleLibrary", "Person", "PersonSpec", "default_people", "people_for_scene",
           "random_spec"]
