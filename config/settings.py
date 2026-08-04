"""
AI Face Mask Detection System - Configuration Settings
------------------------------------------------------
Centralized configuration module containing system constants, UI themes,
model parameters, camera defaults, and file path definitions.

Author: Abid Ali
Diploma in AI & Machine Learning
"""

import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# System Paths
MODEL_PATH = BASE_DIR / "models" / "mask_detector_model.h5"
CASCADE_PATH = BASE_DIR / "src" / "haarcascade_frontalface_default.xml"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"
DEMO_PATH = BASE_DIR / "assets" / "demo.png"
DOCS_DIR = BASE_DIR / "docs"

# Camera Settings
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
TARGET_FPS = 30

# Face & Mask Detection Thresholds
FACE_CONFIDENCE_THRESHOLD = 0.5
MASK_CONFIDENCE_THRESHOLD = 0.5
MODEL_INPUT_SIZE = (224, 224)

# Modern Cyberpunk / Dark Mode UI Color Palette (BGR Format)
COLOR_MASK = (80, 220, 100)      # Neon Green for Mask
COLOR_NO_MASK = (70, 70, 245)    # Vibrant Red for No Mask
COLOR_ACCENT = (255, 180, 50)    # Electric Amber for Bounding Accents
COLOR_BG_DARK = (20, 22, 28)     # Dark Slate Panel Background
COLOR_HEADER_BG = (15, 16, 20)   # Top Bar Background
COLOR_TEXT_PRIMARY = (255, 255, 255) # Pure White Text
COLOR_TEXT_SECONDARY = (180, 190, 200) # Soft Gray Text
COLOR_STATUS_ACTIVE = (50, 205, 50)  # Active Indicator Dot

# UI Visual Parameters
UI_FONT_SCALE = 0.6
UI_FONT_THICKNESS = 1
BORDER_THICKNESS = 2
CORNER_LINE_LENGTH = 15

# System Metadata
APP_TITLE = "AI FACE MASK DETECTOR"
APP_SUBTITLE = "Real-Time Neural Computer Vision System"
AUTHOR_NAME = "Abid Ali"
VERSION = "1.0.0"
