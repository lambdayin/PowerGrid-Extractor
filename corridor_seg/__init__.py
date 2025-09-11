"""
Power Transmission Objects Extraction System
============================================

Automatic extraction of high-voltage power transmission objects from UAV LiDAR point clouds.
Based on Zhang et al., Remote Sensing, 2019.

Author: AI Implementation
"""

__version__ = "1.0.0"
__author__ = "AI Implementation"

from .main import CorridorSegmenter
from .config import Config

__all__ = ["CorridorSegmenter", "Config"]