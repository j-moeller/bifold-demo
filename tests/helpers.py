"""Synthetic test images of people."""

import cv2
import numpy as np


def silhouette(height=200, width=80, color=(40, 60, 90)):
    """An RGBA 'person': an opaque rounded figure on a transparent background."""
    rgba = np.zeros((height, width, 4), np.uint8)
    cv2.ellipse(rgba, (width // 2, height // 2), (width // 3, height // 2 - 2), 0, 0, 360, (*color, 255), -1)
    rgba[height // 2 : height // 2 + 20, width - 12 : width - 2] = (200, 30, 30, 255)  # a bag on one side
    return rgba


def studio_photo(height=640, width=360, person_scale=0.8, top=None):
    """A figure on a plain light grey background (BGR), plus its true 0/1 mask."""
    image = np.full((height, width, 3), 205, np.uint8)
    mask = np.zeros((height, width), np.uint8)
    ph = int(height * person_scale)
    y = (height - ph) // 2 if top is None else top
    cv2.ellipse(mask, (width // 2, y + ph // 2), (width // 6, ph // 2), 0, 0, 360, 1, -1)
    image[mask > 0] = (70, 50, 40)
    return image, mask
