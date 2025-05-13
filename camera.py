from dataclasses import dataclass, field
from typing import Optional, Tuple
import cv2
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
import logging

log = logging.getLogger(__name__)

@dataclass
class Camera:
    url: str
    name: str
    resolution: Tuple[int, int] = (1920, 1080)  # Target resolution
    motion_res: Tuple[int, int] = (256, 256)
    detection_res: Tuple[int, int] = (256, 256)
    motion_threshold: float = 0.1
    cooldown: int = 0
    previous_frame: Optional[np.ndarray] = None  # for motion detection
    face_count: int = 0          # number of faces currently detected
    motion_score: float = 0      # amount of motion (0-1)
    active: bool = False
    last_active_time: float = 0  # timestamp when camera last became active
    main_camera: bool = False    # is this the main camera in PiP mode?
    manual_main: bool = False    # manually selected as main camera
    pipeline: Optional[Gst.Pipeline] = None
    sink: Optional[Gst.Element] = None
    last_frame_time: float = 0   # timestamp of last received frame
    connection_lost: bool = False  # flag for connection status
    frame_rate: float = 30.0     # calculated frame rate
    last_frame: Optional[np.ndarray] = None
    input_resolution: Optional[Tuple[int, int]] = None  # Actual input resolution
    aspect_ratio: float = 16/9   # Default aspect ratio
    is_vertical: bool = False    # Flag for vertical orientation
    frozen_frame: Optional[np.ndarray] = None  # Last frame when connection was lost
    frozen_frame_time: float = 0  # Time when the frame was frozen
    
    # Frame rate calculation
    _frame_times: list[float] = field(default_factory=list)  # Store last N frame timestamps
    _max_frame_times: int = 30  # Keep last 30 frames for calculation (0.5s at 60fps)
    _last_fps_update: float = 0  # Last time we updated the FPS display
    _fps_update_interval: float = 0.5  # Update FPS display every 0.5 seconds
    
    # Bandwidth tracking
    _bandwidth_samples: list[float] = field(default_factory=list)  # Store bandwidth samples
    _max_bandwidth_samples: int = 20  # Keep last 20 samples (10 seconds at 0.5s intervals)
    _last_bandwidth_update: float = 0  # Last time we updated bandwidth
    _bandwidth_update_interval: float = 0.5  # Update bandwidth every 0.5 seconds
    _last_frame_size: int = 0  # Size of last frame in bytes
    _bandwidth: float = 0.0  # Current bandwidth in kbps
    
    # Memory management
    _last_frame: Optional[np.ndarray] = None
    
    def update_input_resolution(self, width: int, height: int) -> None:
        """Update the input resolution and calculate aspect ratio"""
        self.input_resolution = (width, height)
        self.aspect_ratio = width / height
        # Detect vertical orientation (aspect ratio < 1)
        self.is_vertical = self.aspect_ratio < 1
        
    def get_scaled_resolution(self, target_width: int) -> Tuple[int, int]:
        """Calculate scaled resolution maintaining aspect ratio"""
        if not self.input_resolution:
            return (target_width, int(target_width / self.aspect_ratio))
            
        # For vertical videos, scale based on height instead of width
        if self.is_vertical:
            target_height = target_width  # Use full height
            scaled_width = int(target_height * self.aspect_ratio)
            return (scaled_width, target_height)
        else:
            # For horizontal videos, scale based on width
            target_height = int(target_width / self.aspect_ratio)
            return (target_width, target_height)
    
    @property
    def last_frame(self) -> Optional[np.ndarray]:
        return self._last_frame
    
    @last_frame.setter
    def last_frame(self, frame: Optional[np.ndarray]):
        self._last_frame = frame
    
    def update_frame_rate(self, current_time: float) -> None:
        """Update frame rate calculation using a rolling window"""
        # Add current frame time
        self._frame_times.append(current_time)
        
        # Keep only the last N frame times
        if len(self._frame_times) > self._max_frame_times:
            self._frame_times.pop(0)
        
        # Update FPS display periodically
        if current_time - self._last_fps_update >= self._fps_update_interval:
            if len(self._frame_times) >= 2:
                # Calculate average time between frames
                time_diffs = [self._frame_times[i] - self._frame_times[i-1] 
                            for i in range(1, len(self._frame_times))]
                avg_time = sum(time_diffs) / len(time_diffs)
                
                # Calculate FPS (avoid division by zero)
                if avg_time > 0:
                    self.frame_rate = 1.0 / avg_time
                else:
                    self.frame_rate = 0.0
            
            self._last_fps_update = current_time
    
    def update_bandwidth(self, frame_size: int, current_time: float) -> None:
        """Update bandwidth calculation"""
        self._last_frame_size = frame_size
        
        # Update bandwidth periodically
        if current_time - self._last_bandwidth_update >= self._bandwidth_update_interval:
            # Calculate bandwidth in kbps
            if len(self._frame_times) >= 2:
                time_diff = self._frame_times[-1] - self._frame_times[-2]
                if time_diff > 0:
                    # Convert bytes to kbps
                    kbps = (frame_size * 8) / (time_diff * 1000)
                    self._bandwidth_samples.append(kbps)
                    
                    # Keep only the last N samples
                    if len(self._bandwidth_samples) > self._max_bandwidth_samples:
                        self._bandwidth_samples.pop(0)
                    
                    # Calculate average bandwidth
                    self._bandwidth = sum(self._bandwidth_samples) / len(self._bandwidth_samples)
            elif self.connection_lost or current_time - self.last_frame_time > 0.5:
                # Continue sampling zeros during disconnection or frame drops
                self._bandwidth_samples.append(0)
                if len(self._bandwidth_samples) > self._max_bandwidth_samples:
                    self._bandwidth_samples.pop(0)
                self._bandwidth = 0
            
            self._last_bandwidth_update = current_time
    
    def draw_bandwidth_graph(self, frame: np.ndarray, x: int, y: int, width: int, height: int) -> None:
        """Draw a bandwidth area chart on the frame"""
        if not self._bandwidth_samples:
            return
            
        # Create graph background
        cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 0, 0), -1)
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Find max bandwidth for scaling
        max_bw = max(self._bandwidth_samples) if self._bandwidth_samples else 1000
        max_bw = max(max_bw, 1000)  # Minimum scale of 1000 kbps
        
        # Create points for the graph
        points = []
        for i, bw in enumerate(self._bandwidth_samples):
            x_pos = x + int((i / len(self._bandwidth_samples)) * width)
            y_pos = y + height - int((bw / max_bw) * height)
            points.append((x_pos, y_pos))
        
        if len(points) > 1:
            # Create a polygon for the area chart
            polygon_points = points.copy()
            # Add bottom corners to close the polygon
            polygon_points.append((x + width, y + height))
            polygon_points.append((x, y + height))
            
            # Use red for disconnected state, green for connected
            color = (0, 0, 255) if self.connection_lost else (0, 100, 0)
            line_color = (0, 255, 255) if self.connection_lost else (0, 255, 0)
            
            # Fill the area
            cv2.fillPoly(frame, [np.array(polygon_points, dtype=np.int32)], color)
            
            # Draw the top line with anti-aliasing
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], line_color, 1, cv2.LINE_AA)
    
    def cleanup(self):
        """Clean up resources when camera is no longer needed"""
        self.previous_frame = None
        self._last_frame = None
        self.frozen_frame = None  # Clear frozen frame
        self._frame_times.clear()  # Clear frame time history
        self._bandwidth_samples.clear()  # Clear bandwidth history
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.sink = None
            self.pipeline = None 