"""
Utility Functions Module
------------------------
Provides image preprocessing, modern HUD drawing routines, bounding box
rendering with corner brackets, and asset generation routines.

Author: Abid Ali
"""

import os
import datetime
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont
from config import settings


def preprocess_face(face_image: np.ndarray, target_size: tuple = (224, 224)) -> np.ndarray:
    """
    Preprocess cropped face image for neural network input.

    :param face_image: Cropped face ROI in BGR format.
    :param target_size: Network target shape (width, height).
    :return: Preprocessed normalized image tensor with batch dimension (1, H, W, 3).
    """
    if face_image is None or face_image.size == 0:
        return np.zeros((1, target_size[0], target_size[1], 3), dtype=np.float32)

    # Convert BGR to RGB
    face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
    
    # Resize image to target dimensions
    resized = cv2.resize(face_rgb, target_size, interpolation=cv2.INTER_AREA)
    
    # Scale pixel values to range [0, 1] or MobileNetV2 preprocessing [-1, 1]
    normalized = resized.astype(np.float32) / 255.0
    
    # Add batch dimension
    expanded = np.expand_dims(normalized, axis=0)
    return expanded


def draw_corner_brackets(
    frame: np.ndarray,
    top_left: tuple,
    bottom_right: tuple,
    color: tuple,
    thickness: int = 2,
    length: int = 15
) -> None:
    """
    Draw futuristic corner brackets around bounding boxes for a polished HUD look.

    :param frame: Output video frame.
    :param top_left: (x1, y1)
    :param bottom_right: (x2, y2)
    :param color: BGR color tuple.
    :param thickness: Line thickness.
    :param length: Bracket corner arm length.
    """
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Top-Left Bracket
    cv2.line(frame, (x1, y1), (x1 + length, y1), color, thickness)
    cv2.line(frame, (x1, y1), (x1, y1 + length), color, thickness)

    # Top-Right Bracket
    cv2.line(frame, (x2, y1), (x2 - length, y1), color, thickness)
    cv2.line(frame, (x2, y1), (x2, y1 + length), color, thickness)

    # Bottom-Left Bracket
    cv2.line(frame, (x1, y2), (x1 + length, y2), color, thickness)
    cv2.line(frame, (x1, y2), (x1, y2 - length), color, thickness)

    # Bottom-Right Bracket
    cv2.line(frame, (x2, y2), (x2 - length, y2), color, thickness)
    cv2.line(frame, (x2, y2), (x2, y2 - length), color, thickness)


def draw_bounding_box(
    frame: np.ndarray,
    box: tuple,
    label: str,
    confidence: float,
    is_mask: bool
) -> np.ndarray:
    """
    Render colored bounding box, corner brackets, and label tag for a face detection.

    :param frame: Image frame array.
    :param box: (x, y, w, h) bounding box coordinates.
    :param label: Text label ("Mask" or "No Mask").
    :param confidence: Detection probability score (0.0 to 1.0).
    :param is_mask: True if wearing mask, False otherwise.
    :return: Frame with rendered graphics.
    """
    x, y, w, h = box
    color = settings.COLOR_MASK if is_mask else settings.COLOR_NO_MASK
    
    # Draw semi-transparent bounding box
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)

    # Draw main rectangle border
    cv2.rectangle(frame, (x, y), (x + w, y + h), color, settings.BORDER_THICKNESS)
    
    # Draw sleek corner brackets
    draw_corner_brackets(frame, (x, y), (x + w, y + h), color, thickness=3, length=settings.CORNER_LINE_LENGTH)

    # Format text tag
    text = f"{label}: {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    font_thickness = 2
    (text_w, text_h), baseline = cv2.getTextSize(text, font, font_scale, font_thickness)

    # Tag background panel
    tag_y1 = max(y - text_h - 12, 10)
    tag_y2 = tag_y1 + text_h + 10
    tag_x1 = x
    tag_x2 = x + text_w + 16

    # Draw tag background
    cv2.rectangle(frame, (tag_x1, tag_y1), (tag_x2, tag_y2), color, -1)
    
    # Draw label text
    cv2.putText(
        frame,
        text,
        (tag_x1 + 8, tag_y2 - 6),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )

    return frame


def draw_ui_overlay(
    frame: np.ndarray,
    fps: float,
    total_faces: int,
    mask_count: int,
    nomask_count: int,
    webcam_active: bool = True
) -> np.ndarray:
    """
    Render a professional Heads-Up Display (HUD) overlay over the video frame.

    :param frame: Video frame numpy array.
    :param fps: Current FPS.
    :param total_faces: Number of faces detected.
    :param mask_count: Number of faces with masks.
    :param nomask_count: Number of faces without masks.
    :param webcam_active: Webcam status flag.
    :return: Frame with HUD overlay.
    """
    h, w = frame.shape[:2]

    # Top Header Bar
    header_h = 60
    header_overlay = frame.copy()
    cv2.rectangle(header_overlay, (0, 0), (w, header_h), settings.COLOR_HEADER_BG, -1)
    cv2.addWeighted(header_overlay, 0.75, frame, 0.25, 0, frame)
    cv2.line(frame, (0, header_h), (w, header_h), (60, 65, 75), 1)

    # App Title & Subtitle
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, settings.APP_TITLE, (20, 28), font, 0.65, settings.COLOR_TEXT_PRIMARY, 2, cv2.LINE_AA)
    cv2.putText(frame, settings.APP_SUBTITLE, (20, 48), font, 0.4, settings.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA)

    # Right Header Info: Webcam Status & Live Clock
    status_color = settings.COLOR_STATUS_ACTIVE if webcam_active else (50, 50, 200)
    status_text = "WEBCAM: LIVE" if webcam_active else "WEBCAM: OFFLINE"
    
    # Webcam Status Pill
    pill_x = w - 180
    cv2.circle(frame, (pill_x, 22), 6, status_color, -1)
    cv2.putText(frame, status_text, (pill_x + 14, 26), font, 0.45, settings.COLOR_TEXT_PRIMARY, 1, cv2.LINE_AA)

    # Timestamp
    time_str = datetime.datetime.now().strftime("%Y-%m-%d  %H:%M:%S")
    cv2.putText(frame, time_str, (w - 210, 48), font, 0.4, settings.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA)

    # Bottom Dashboard Statistics Bar
    footer_h = 45
    footer_y = h - footer_h
    footer_overlay = frame.copy()
    cv2.rectangle(footer_overlay, (0, footer_y), (w, h), settings.COLOR_BG_DARK, -1)
    cv2.addWeighted(footer_overlay, 0.8, frame, 0.2, 0, frame)
    cv2.line(frame, (0, footer_y), (w, footer_y), (60, 65, 75), 1)

    # Statistics Pills
    stats = [
        (f"FACES: {total_faces}", (180, 180, 180)),
        (f"MASK: {mask_count}", settings.COLOR_MASK),
        (f"NO MASK: {nomask_count}", settings.COLOR_NO_MASK),
        (f"FPS: {fps:.1f}", (255, 200, 80))
    ]

    x_offset = 20
    for text, color in stats:
        cv2.rectangle(frame, (x_offset, footer_y + 8), (x_offset + 120, h - 8), (35, 40, 50), -1)
        cv2.rectangle(frame, (x_offset, footer_y + 8), (x_offset + 120, h - 8), color, 1)
        cv2.putText(frame, text, (x_offset + 10, footer_y + 28), font, 0.45, color, 1, cv2.LINE_AA)
        x_offset += 135

    # Control Shortcut Hint on bottom right
    hint_text = "Press 'Q' or 'ESC' to Exit | 'S' Save Snapshot"
    cv2.putText(frame, hint_text, (w - 360, footer_y + 28), font, 0.4, settings.COLOR_TEXT_SECONDARY, 1, cv2.LINE_AA)

    return frame


def generate_synthetic_assets() -> None:
    """
    Generate professional logo and demo assets programmatically if they do not exist.
    """
    os.makedirs(settings.LOGO_PATH.parent, exist_ok=True)

    # Generate logo.png if missing
    if not os.path.exists(settings.LOGO_PATH):
        img = Image.new("RGBA", (400, 400), (20, 24, 33, 255))
        draw = ImageDraw.Draw(img)
        
        # Outer Ring
        draw.ellipse([30, 30, 370, 370], outline=(80, 220, 100, 255), width=8)
        draw.ellipse([50, 50, 350, 350], outline=(50, 180, 250, 255), width=3)
        
        # Shield Graphic
        shield_pts = [(200, 90), (310, 140), (310, 250), (200, 320), (90, 250), (90, 140)]
        draw.polygon(shield_pts, fill=(30, 36, 48, 255), outline=(80, 220, 100, 255))
        
        # Mask Shape inside Shield
        draw.rounded_rectangle([130, 190, 270, 260], radius=15, fill=(80, 220, 100, 255))
        draw.line([(100, 205), (130, 215)], fill=(200, 220, 200, 255), width=4)
        draw.line([(300, 205), (270, 215)], fill=(200, 220, 200, 255), width=4)
        draw.line([(100, 245), (130, 235)], fill=(200, 220, 200, 255), width=4)
        draw.line([(300, 245), (270, 235)], fill=(200, 220, 200, 255), width=4)

        img.save(settings.LOGO_PATH)

    # Generate demo.png if missing
    if not os.path.exists(settings.DEMO_PATH):
        demo_frame = np.zeros((settings.FRAME_HEIGHT, settings.FRAME_WIDTH, 3), dtype=np.uint8)
        
        # Background subtle gradient/pattern
        for i in range(settings.FRAME_HEIGHT):
            val = int(25 + (i / settings.FRAME_HEIGHT) * 20)
            demo_frame[i, :] = (val, val + 5, val + 10)

        # Draw simulated human face silhouette with mask (Person 1 - Mask)
        f1_x, f1_y, f1_w, f1_h = 320, 180, 220, 280
        # Draw head oval
        cv2.ellipse(demo_frame, (f1_x + f1_w//2, f1_y + f1_h//2), (f1_w//2, f1_h//2), 0, 0, 360, (180, 195, 210), -1)
        # Draw eyes
        cv2.circle(demo_frame, (f1_x + 60, f1_y + 100), 12, (50, 50, 50), -1)
        cv2.circle(demo_frame, (f1_x + 160, f1_y + 100), 12, (50, 50, 50), -1)
        # Draw face mask
        cv2.rectangle(demo_frame, (f1_x + 30, f1_y + 140), (f1_x + 190, f1_y + 240), (220, 220, 220), -1)
        cv2.line(demo_frame, (f1_x + 30, f1_y + 150), (f1_x + 10, f1_y + 110), (200, 200, 200), 3)
        cv2.line(demo_frame, (f1_x + 190, f1_y + 150), (f1_x + 210, f1_y + 110), (200, 200, 200), 3)

        draw_bounding_box(demo_frame, (f1_x, f1_y, f1_w, f1_h), "Mask", 0.985, is_mask=True)

        # Draw simulated human face silhouette without mask (Person 2 - No Mask)
        f2_x, f2_y, f2_w, f2_h = 740, 190, 210, 270
        # Draw head oval
        cv2.ellipse(demo_frame, (f2_x + f2_w//2, f2_y + f2_h//2), (f2_w//2, f2_h//2), 0, 0, 360, (175, 190, 205), -1)
        # Draw eyes
        cv2.circle(demo_frame, (f2_x + 55, f2_y + 95), 11, (40, 40, 40), -1)
        cv2.circle(demo_frame, (f2_x + 155, f2_y + 95), 11, (40, 40, 40), -1)
        # Draw nose & mouth
        cv2.line(demo_frame, (f2_x + 105, f2_y + 110), (f2_x + 105, f2_y + 155), (120, 130, 140), 3)
        cv2.ellipse(demo_frame, (f2_x + 105, f2_y + 190), (35, 15), 0, 0, 180, (100, 80, 110), -1)

        draw_bounding_box(demo_frame, (f2_x, f2_y, f2_w, f2_h), "No Mask", 0.962, is_mask=False)

        # Draw HUD overlay
        draw_ui_overlay(demo_frame, fps=29.4, total_faces=2, mask_count=1, nomask_count=1, webcam_active=True)

        cv2.imwrite(str(settings.DEMO_PATH), demo_frame)


if __name__ == "__main__":
    generate_synthetic_assets()
    print("Assets generated successfully.")
