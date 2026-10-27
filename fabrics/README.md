# Fabric cache

Fabric swatches made from descriptions (`bifold-characters fabric "blue
paisley silk"`, or the character creator's "Describe a fabric" box) are kept
here: a PNG and a JSON file each (the prompt, the model, the seed, the size,
the method and the seam score). The same description gives its swatches back
at once, without loading the model. Swatches whose seams showed are kept too,
so they are not made again.

Delete any of them, or the whole folder, to make them again. At most 400 are
kept; the oldest go first. `BIFOLD_FABRIC_CACHE=DIR` puts the cache elsewhere.

Everything in this folder except this README is ignored by git.
