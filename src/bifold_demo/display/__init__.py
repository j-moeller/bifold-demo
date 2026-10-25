"""Combining video and detections on screen."""

from .overlay import class_color, draw_banner, draw_detections, draw_panel
from .window import Window

__all__ = ["Window", "class_color", "draw_banner", "draw_detections", "draw_panel"]
