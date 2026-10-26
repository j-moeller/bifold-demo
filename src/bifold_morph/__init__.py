"""bifold-morph: live webcam face editing with age and gender sliders.

e4e finds each face's StyleGAN2 code, the code moves along an age and a gender direction (found with MiVOLO as the
judge of how old and which gender a face looks), and StyleGAN2 draws the change back into the video.
"""

__version__ = "0.1.0"
