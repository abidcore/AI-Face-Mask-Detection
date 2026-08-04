"""
FPS Counter Module
------------------
Provides precise, smoothed Frame-Per-Second (FPS) tracking for real-time
video analytics and performance metrics.

Author: Abid Ali
"""

import time
from collections import deque


class FPSCounter:
    """
    High-precision FPS counter using moving average smoothing
    to eliminate frame-to-frame performance jitter.
    """

    def __init__(self, buffer_size: int = 30):
        """
        Initialize the FPS Counter.

        :param buffer_size: Number of historical frame durations to average.
        """
        self.buffer_size = buffer_size
        self.frame_times = deque(maxlen=buffer_size)
        self.last_timestamp = None
        self.current_fps = 0.0
        self.total_frames = 0
        self.start_time = None

    def start(self) -> None:
        """Start or restart the FPS timer."""
        self.last_timestamp = time.perf_counter()
        self.start_time = self.last_timestamp
        self.frame_times.clear()
        self.total_frames = 0

    def update(self) -> float:
        """
        Update the timer with a new frame timestamp.

        :return: Current calculated smoothed FPS value.
        """
        now = time.perf_counter()
        if self.start_time is None:
            self.start_time = now
            self.last_timestamp = now
            return 0.0

        if self.last_timestamp is not None:
            delta = now - self.last_timestamp
            if delta > 0:
                self.frame_times.append(delta)

        self.last_timestamp = now
        self.total_frames += 1

        if self.frame_times:
            avg_delta = sum(self.frame_times) / len(self.frame_times)
            self.current_fps = 1.0 / avg_delta if avg_delta > 0 else 0.0
        else:
            self.current_fps = 0.0

        return self.current_fps

    def get_fps(self) -> float:
        """
        Retrieve the latest smoothed FPS value.

        :return: Smoothed FPS float value.
        """
        return round(self.current_fps, 1)

    def get_average_fps(self) -> float:
        """
        Retrieve the cumulative average FPS since execution started.

        :return: Average FPS float value.
        """
        if self.start_time is None or self.total_frames == 0:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        return round(self.total_frames / elapsed, 1) if elapsed > 0 else 0.0

    def reset(self) -> None:
        """Reset the FPS tracking state."""
        self.start();
