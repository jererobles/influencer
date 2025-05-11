import asyncio
import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
from enum import Enum, auto
import logging
import mediapipe as mp
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import time
import datetime
import yaml  # Add import for yaml to load secrets
import gc  # Explicit garbage collection
import weakref  # For weak references

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

# Initialize GStreamer once
Gst.init(None)

class RetryManager:
    """Manages connection retry state for cameras"""
    def __init__(self, max_retries: int = 10, base_retry_interval: float = 1.0, max_retry_interval: float = 30.0):
        self.max_retries = max_retries
        self.base_retry_interval = base_retry_interval
        self.max_retry_interval = max_retry_interval
        self.retry_states: Dict[str, Dict] = {}  # camera_name -> retry state
    
    def get_retry_state(self, camera_name: str) -> Dict:
        """Get or create retry state for a camera"""
        if camera_name not in self.retry_states:
            self.retry_states[camera_name] = {
                'retry_count': 0,
                'next_retry_time': 0,
                'connection_lost': False,
                'last_retry_time': 0  # Track when we last attempted a retry
            }
        return self.retry_states[camera_name]
    
    def handle_connection_loss(self, camera_name: str, current_time: float) -> None:
        """Handle connection loss and setup retry timing"""
        state = self.get_retry_state(camera_name)
        if not state['connection_lost']:
            state['connection_lost'] = True
            state['retry_count'] += 1
            state['next_retry_time'] = current_time + self.get_retry_interval(state['retry_count'])
            state['last_retry_time'] = current_time
            log.debug(f"Connection loss for {camera_name}, retry count: {state['retry_count']}")
    
    def should_retry(self, camera_name: str, current_time: float) -> bool:
        """Check if it's time to retry connection"""
        state = self.get_retry_state(camera_name)
        return (state['connection_lost'] and 
                state['retry_count'] < self.max_retries and 
                current_time >= state['next_retry_time'])
    
    def prepare_next_retry(self, camera_name: str, current_time: float) -> None:
        """Prepare for the next retry attempt"""
        state = self.get_retry_state(camera_name)
        # Only increment retry count if this is a new retry attempt
        if current_time - state['last_retry_time'] > self.base_retry_interval:
            state['retry_count'] += 1
            state['last_retry_time'] = current_time
        state['next_retry_time'] = current_time + self.get_retry_interval(state['retry_count'])
        retry_interval = self.get_retry_interval(state['retry_count'])
        log.info(f"Attempting reconnection to {camera_name} (attempt {state['retry_count']}/{self.max_retries}, next retry in {retry_interval:.1f}s)")
    
    def get_retry_interval(self, retry_count: int) -> float:
        """Calculate retry interval using exponential backoff"""
        interval = self.base_retry_interval * (2 ** retry_count)
        return min(interval, self.max_retry_interval)
    
    def reset_retry_state(self, camera_name: str) -> None:
        """Reset retry state for a camera after successful connection"""
        state = self.get_retry_state(camera_name)
        state['retry_count'] = 0
        state['connection_lost'] = False
        state['next_retry_time'] = 0
        state['last_retry_time'] = 0
        log.debug(f"Reset retry state for {camera_name}")
    
    def cleanup(self, camera_name: str) -> None:
        """Clean up retry state for a camera"""
        if camera_name in self.retry_states:
            del self.retry_states[camera_name]
            log.debug(f"Cleaned up retry state for {camera_name}")

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
        
    def get_scaled_resolution(self, target_width: int) -> Tuple[int, int]:
        """Calculate scaled resolution maintaining aspect ratio"""
        if not self.input_resolution:
            return (target_width, int(target_width / self.aspect_ratio))
            
        # Calculate height based on aspect ratio
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

    def get_retry_interval(self) -> float:
        """Calculate retry interval using exponential backoff"""
        # Start with base interval and double with each retry
        interval = self.base_retry_interval * (2 ** self.retry_count)
        # Cap at max_retry_interval
        return min(interval, self.max_retry_interval)

    def handle_connection_loss(self, current_time: float) -> None:
        """Handle connection loss and setup retry timing"""
        if not self.connection_lost:
            # Immediately mark as inactive and lost connection
            self.active = False
            self.connection_lost = True
            self.face_count = 0
            self.motion_score = 0
            self.frame_rate = 0
            self._bandwidth = 0
            self._frame_times.clear()
            
            # Don't reset retry count here, only increment
            self.retry_count += 1
            self.next_retry_time = current_time + self.get_retry_interval()
            
            # If this was the main camera, remove that status
            if self.main_camera:
                self.main_camera = False
                self.manual_main = False  # Also remove manual lock
            
            # Store frozen frame if we have a last frame
            if self.last_frame is not None:
                try:
                    gray = cv2.cvtColor(self.last_frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (15, 15), 20)
                    self.frozen_frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
                    self.frozen_frame_time = current_time
                except Exception as e:
                    log.error(f"Error creating frozen frame for {self.name}: {e}")
                    self.frozen_frame = self.last_frame.copy()
                    self.frozen_frame_time = current_time

    def should_retry(self, current_time: float) -> bool:
        """Check if it's time to retry connection"""
        return (self.connection_lost and 
                self.retry_count < self.max_retries and 
                current_time >= self.next_retry_time)

    def prepare_next_retry(self, current_time: float) -> None:
        """Prepare for the next retry attempt"""
        self.next_retry_time = current_time + self.get_retry_interval()
        retry_interval = self.get_retry_interval()
        log.info(f"Attempting reconnection to {self.name} (attempt {self.retry_count}/{self.max_retries}, next retry in {retry_interval:.1f}s)")

class ViewMode(Enum):
    INPUT = auto()      # show input view in grid
    OUTPUT = auto()     # output mode

class FramePool:
    """Memory pool for frame buffers to avoid constant allocations"""
    
    def __init__(self, max_frames=5):
        self.available = []
        self.max_frames = max_frames
        self.size_map = {}  # Track frame sizes
    
    def get_frame(self, shape, dtype=np.uint8):
        """Get a frame from the pool or create a new one if needed"""
        key = (shape, dtype)
        
        if key in self.size_map:
            frames = self.size_map[key]
            if frames:
                return frames.pop()
        
        # If we reach here, we need to create a new frame
        return np.zeros(shape, dtype=dtype)
    
    def return_frame(self, frame):
        """Return a frame to the pool"""
        if frame is None:
            return
            
        key = (frame.shape, frame.dtype)
        
        if key not in self.size_map:
            self.size_map[key] = []
            
        frames = self.size_map[key]
        
        # Only keep a limited number of frames of each size
        if len(frames) < self.max_frames:
            frames.append(frame)

class GstBuffer:
    """Wrapper for GStreamer buffer management"""
    
    def __init__(self, size=0):
        self.buffer = None
        self.size = 0
        if size > 0:
            self.ensure_size(size)
    
    def ensure_size(self, size):
        """Ensure the buffer is at least the requested size"""
        if self.buffer is None or self.size < size:
            self.buffer = bytearray(size)
            self.size = size
            return True
        return False
    
    def get_view(self, size=None):
        """Get a memory view of the buffer"""
        if size is None or size == self.size:
            return memoryview(self.buffer)
        else:
            return memoryview(self.buffer)[:size]
    
    def cleanup(self):
        """Release the buffer"""
        self.buffer = None
        self.size = 0

class WorkshopStream:
    def __init__(self, debug: bool = False):
        self.cameras: Dict[str, Camera] = {}
        self.output_frame: Optional[np.ndarray] = None
        self.clean_frame_for_recording: Optional[np.ndarray] = None
        self.running = False
        self.debug = debug
        self.view_mode = ViewMode.OUTPUT
        
        # Recording related attributes
        self.recording = False
        self.recording_pipeline = None
        self.recording_src = None
        self.recording_paused = False
        self.auto_recording = False
        self.frame_count = 0  # Initialize frame counter
        
        # Streaming related attributes
        self.streaming = False
        self.streaming_pipeline = None
        self.streaming_src = None
        
        # Memory management
        self.frame_pool = FramePool()
        
        # Load secrets
        self.twitch_stream_key = self._load_twitch_stream_key()
        
        # Initialize retry manager
        self.retry_manager = RetryManager()
        
        # Initialize mediapipe detector
        BaseOptions = mp.tasks.BaseOptions
        Detector = mp.tasks.vision.ObjectDetector
        DetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Load lock icon for overlays
        self.pushpin_icon = cv2.imread('icons8-lock-30.png', cv2.IMREAD_UNCHANGED)
        if self.pushpin_icon is None:
            log.warning('Could not load lock.png for overlays.')

        # Create detector for person detection
        options = DetectorOptions(
            base_options=BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
            running_mode=VisionRunningMode.IMAGE,
            score_threshold=0.29,
            category_allowlist=['person'])
        self.detector = Detector.create_from_options(options)
        
        # Add timestamp for last tab press
        self.last_tab_time = 0
        self.tab_cooldown = 5.0  # 5 seconds cooldown
        
    def add_camera(self, url: str, name: str) -> None:
        """Add a new camera to the stream"""
        camera = Camera(url=url, name=name)
        self.cameras[name] = camera
        # Initialize retry state for the new camera
        self.retry_manager.get_retry_state(name)
        log.info(f"Added camera: {name} @ {url}")
        
    def _load_twitch_stream_key(self) -> str:
        """Load Twitch stream key from secrets.yaml file"""
        try:
            with open('secrets.yaml', 'r') as f:
                secrets = yaml.safe_load(f)
                stream_key = secrets.get('twitch_stream_key', '')
                if not stream_key:
                    log.warning("Twitch stream key not found in secrets.yaml. Streaming will not work.")
                return stream_key
        except Exception as e:
            log.error(f"Failed to load Twitch stream key: {e}")
            return ''
        
    def _create_camera_pipeline(self, camera: Camera) -> None:
        """Create and configure GStreamer pipeline for a camera"""
        # Create GStreamer pipeline based on camera URL type
        if camera.url.isdigit():  # USB webcam
            pipeline_str = (
                f'avfvideosrc device-index={camera.url} ! '
                'video/x-raw,format=YUY2,width=1920,height=1080,framerate=60/1 ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=4096 drop=False'
            )
        elif camera.url.startswith('http://'):  # HTTP stream
            pipeline_str = (
                f'souphttpsrc location={camera.url} ! '
                'decodebin ! videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=4096 drop=False'
            )
        else:  # RTSP stream
            pipeline_str = (
                f'rtspsrc location={camera.url} latency=100 buffer-mode=auto do-retransmission=true '
                'drop-on-latency=false ntp-sync=false protocols=tcp ! '
                'rtpjitterbuffer latency=500 drop-on-latency=false ! '
                'queue max-size-buffers=4096 max-size-bytes=0 max-size-time=0 ! '
                'rtph264depay ! h264parse ! '
                'avdec_h264 max-threads=4 ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=4096 drop=False'
            )

        log.debug(f"Creating pipeline for {camera.name}: {pipeline_str}")
        
        # Create and store the pipeline in the camera object
        camera.pipeline = Gst.parse_launch(pipeline_str)
        camera.sink = camera.pipeline.get_by_name('sink')
        
        # Add bus watch to monitor pipeline state changes and errors
        bus = camera.pipeline.get_bus()
        bus.add_signal_watch()
        
        def on_bus_message(bus, message):
            t = message.type
            if t == Gst.MessageType.ERROR:
                err, debug = message.parse_error()
                log.error(f"Pipeline error for {camera.name}: {err.message}")
                log.debug(f"Debug info: {debug}")
                camera.connection_lost = True
            elif t == Gst.MessageType.STATE_CHANGED:
                old_state, new_state, pending_state = message.parse_state_changed()
                if message.src == camera.pipeline:
                    log.debug(f"Pipeline state changed for {camera.name}: {old_state.value_nick} -> {new_state.value_nick}")
            elif t == Gst.MessageType.EOS:
                log.warning(f"End of stream for {camera.name}")
                camera.connection_lost = True
            return True
            
        bus.connect('message', on_bus_message)
        
    async def _capture_frames(self, camera: Camera) -> None:
        """Capture frames from a camera and detect people"""
        detection_counter = 0
        
        while self.running:
            try:
                # Create GStreamer pipeline for this camera
                self._create_camera_pipeline(camera)

                # Create a reference to self that won't prevent garbage collection
                stream_ref = weakref.ref(self)
                
                # Flag to track if we've received the first frame
                first_frame_received = False
                connection_start_time = time.time()
                connection_timeout = 5.0  # 5 seconds timeout for initial connection

                # Setup frame callback
                def on_new_sample(appsink):
                    try:
                        # Get the stream reference
                        stream = stream_ref()
                        if not stream:
                            return Gst.FlowReturn.ERROR

                        current_time = time.time()
                        
                        # Get the sample from appsink
                        sample = appsink.emit("pull-sample")
                        if not sample:
                            return Gst.FlowReturn.ERROR

                        # Get the buffer from the sample
                        buffer = sample.get_buffer()
                        if not buffer:
                            return Gst.FlowReturn.ERROR

                        # Map the buffer for reading
                        success, map_info = buffer.map(Gst.MapFlags.READ)
                        if not success:
                            return Gst.FlowReturn.ERROR

                        try:
                            # Update camera state
                            camera.last_frame_time = current_time
                            if camera.connection_lost:
                                # Reset retry state when we get a frame after being disconnected
                                stream.retry_manager.reset_retry_state(camera.name)
                            camera.connection_lost = False  # We got a frame, so connection is not lost
                            
                            # Mark that we've received our first frame
                            nonlocal first_frame_received
                            first_frame_received = True
                            
                            # Update frame rate calculation
                            camera.update_frame_rate(current_time)

                            # Get caps to determine actual resolution
                            caps = sample.get_caps()
                            if caps:
                                structure = caps.get_structure(0)
                                if structure:
                                    width = structure.get_value('width')
                                    height = structure.get_value('height')
                                    if width and height:
                                        camera.update_input_resolution(width, height)

                            # Update bandwidth calculation
                            frame_size = buffer.get_size()
                            camera.update_bandwidth(frame_size, current_time)

                            # Get a reusable frame from the pool
                            old_frame = camera.last_frame
                            
                            # Create frame with actual input resolution
                            if camera.input_resolution:
                                camera.last_frame = np.ndarray(
                                    shape=(camera.input_resolution[1], camera.input_resolution[0], 3),
                                    dtype=np.uint8,
                                    buffer=map_info.data
                                )
                            else:
                                # Fallback to target resolution if input resolution not known
                                camera.last_frame = np.ndarray(
                                    shape=(camera.resolution[1], camera.resolution[0], 3),
                                    dtype=np.uint8,
                                    buffer=map_info.data
                                )
                            
                            # Return old frame to the pool
                            if old_frame is not None:
                                stream.frame_pool.return_frame(old_frame)
                            
                            # Handle detection on a subset of frames
                            nonlocal detection_counter
                            detection_counter += 1
                            
                            if detection_counter % 10 == 0:  # Every 10th frame
                                try:
                                    # Get a frame for detection resizing
                                    small_frame = cv2.resize(camera.last_frame, camera.detection_res)
                                    
                                    # Convert to RGB for MediaPipe
                                    small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                                    
                                    # Create MediaPipe image
                                    mp_image = mp.Image(
                                        image_format=mp.ImageFormat.SRGB,
                                        data=small_frame_rgb
                                    )
                                    
                                    # Run detection
                                    results = stream.detector.detect(mp_image)
                                    
                                    # Update camera state
                                    camera.face_count = len(results.detections)
                                    if camera.face_count > 0:
                                        if not camera.active:
                                            log.info(f"Camera {camera.name} became active")
                                        camera.active = True
                                        camera.cooldown = 30
                                        camera.last_active_time = time.time()
                                    elif camera.cooldown > 0:
                                        camera.cooldown -= 1
                                    else:
                                        if camera.active:
                                            log.info(f"Camera {camera.name} became inactive")
                                        camera.active = False
                                        camera.face_count = 0
                                    
                                    # Calculate motion score
                                    # Convert to grayscale for motion detection
                                    gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                                    
                                    if camera.previous_frame is not None:
                                        # Calculate difference for motion detection
                                        diff = cv2.absdiff(camera.previous_frame, gray)
                                        camera.motion_score = np.mean(diff) / 255.0
                                    
                                    # Save current frame for next motion detection
                                    camera.previous_frame = gray
                                    
                                except Exception as e:
                                    log.error(f"Error in detection for {camera.name}: {e}")
                                    # Don't let detection errors affect the main frame processing
                                    pass
                                finally:
                                    # Clean up temporary objects
                                    if 'small_frame' in locals():
                                        del small_frame
                                    if 'small_frame_rgb' in locals():
                                        del small_frame_rgb
                                    if 'mp_image' in locals():
                                        del mp_image

                        finally:
                            # Always unmap the buffer
                            buffer.unmap(map_info)

                    except Exception as e:
                        log.error(f"Error in frame callback for {camera.name}: {e}")
                        camera.connection_lost = True
                        self.retry_manager.handle_connection_loss(camera.name, time.time())
                        return Gst.FlowReturn.ERROR

                    return Gst.FlowReturn.OK

                # Connect callback and start pipeline
                camera.sink.connect('new-sample', on_new_sample)
                camera.pipeline.set_state(Gst.State.PLAYING)

                log.info(f"Pipeline started for {camera.name}")

                # Wait for first frame or timeout
                while self.running and not first_frame_received:
                    if time.time() - connection_start_time > connection_timeout:
                        log.error(f"Connection timeout for {camera.name} after {connection_timeout} seconds")
                        camera.connection_lost = True
                        self.retry_manager.handle_connection_loss(camera.name, time.time())
                        break
                    await asyncio.sleep(0.1)

                # If we didn't get a first frame, clean up and retry
                if not first_frame_received:
                    self._cleanup_camera(camera)
                    retry_state = self.retry_manager.get_retry_state(camera.name)
                    if retry_state['retry_count'] < self.retry_manager.max_retries:
                        self.retry_manager.prepare_next_retry(camera.name, time.time())
                        await asyncio.sleep(self.retry_manager.get_retry_interval(retry_state['retry_count']))
                        continue
                    else:
                        log.error(f"Max retries ({self.retry_manager.max_retries}) reached for camera {camera.name}")
                        # Remove the camera from the system
                        camera_name = camera.name
                        del self.cameras[camera_name]
                        self.retry_manager.cleanup(camera_name)
                        log.info(f"Removed camera {camera_name} after max retries")
                        break

                # Main loop to keep pipeline running and check for connection loss
                last_metrics_update = time.time()
                metrics_update_interval = 0.1  # Update metrics every 100ms for smoother graphs
                last_frame_time = camera.last_frame_time  # Track last actual frame time

                while self.running:
                    current_time = time.time()
                    
                    # Update metrics more frequently than connection check
                    if current_time - last_metrics_update >= metrics_update_interval:
                        # If we haven't received a frame recently, update metrics to show degradation
                        if current_time - camera.last_frame_time > 0.5:  # 500ms threshold
                            # Clear frame times if we haven't received a frame in a while
                            if current_time - last_frame_time > 1.0:  # 1 second threshold
                                camera._frame_times.clear()
                                camera.frame_rate = 0.0
                            
                            # Update bandwidth to show degradation
                            camera.update_bandwidth(0, current_time)
                        else:
                            # We received a frame recently, update last_frame_time
                            last_frame_time = camera.last_frame_time
                        
                        last_metrics_update = current_time
                    
                    # Check for connection loss (no frames for 3 seconds)
                    if current_time - camera.last_frame_time > 3:
                        if not camera.connection_lost:
                            try:
                                # Immediately mark as inactive and lost connection
                                camera.active = False
                                camera.connection_lost = True
                                camera.face_count = 0
                                camera.motion_score = 0
                                camera.frame_rate = 0
                                
                                self.retry_manager.handle_connection_loss(camera.name, current_time)
                                log.warning(f"Connection lost to camera {camera.name}")
                                
                                # If this was the main camera, select another one
                                if camera.main_camera:
                                    camera.main_camera = False
                                    camera.manual_main = False  # Also remove manual lock
                                    # Select a new main camera
                                    new_main = self._select_main_camera()
                                    if new_main:
                                        log.info(f"Switched main camera to {new_main} after {camera.name} lost connection")
                                    else:
                                        log.warning("No active cameras available to switch to")
                                
                                # Clean up the existing pipeline
                                self._cleanup_camera(camera)
                            except Exception as e:
                                log.error(f"Error handling connection loss for {camera.name}: {e}")
                                # Force cleanup even if there was an error
                                self._cleanup_camera(camera)
                        
                        # Check retry limits
                        retry_state = self.retry_manager.get_retry_state(camera.name)
                        if retry_state['retry_count'] >= self.retry_manager.max_retries:
                            log.error(f"Max retries ({self.retry_manager.max_retries}) reached for camera {camera.name}")
                            # Remove the camera from the system
                            camera_name = camera.name
                            self._cleanup_camera(camera)
                            del self.cameras[camera_name]
                            self.retry_manager.cleanup(camera_name)
                            log.info(f"Removed camera {camera_name} after max retries")
                            break
                        
                        # Check if it's time to retry
                        if self.retry_manager.should_retry(camera.name, current_time):
                            try:
                                self.retry_manager.prepare_next_retry(camera.name, current_time)
                                # Break the inner loop to recreate the pipeline
                                break
                            except Exception as e:
                                log.error(f"Error during reconnection attempt for {camera.name}: {e}")
                                # If we hit an error during reconnection, wait a bit longer
                                await asyncio.sleep(self.retry_manager.get_retry_interval(
                                    self.retry_manager.get_retry_state(camera.name)['retry_count']) * 2)
                                break
                    
                    await asyncio.sleep(0.05)  # Check more frequently for smoother updates

            except Exception as e:
                log.error(f"Error in camera pipeline {camera.name}: {e}")
                self._cleanup_camera(camera)
                
                # Handle retries
                retry_state = self.retry_manager.get_retry_state(camera.name)
                if retry_state['retry_count'] < self.retry_manager.max_retries:
                    self.retry_manager.prepare_next_retry(camera.name, time.time())
                    await asyncio.sleep(self.retry_manager.get_retry_interval(retry_state['retry_count']))
                else:
                    log.error(f"Max retries ({self.retry_manager.max_retries}) reached for camera {camera.name}")
                    # Remove the camera from the system
                    camera_name = camera.name
                    del self.cameras[camera_name]
                    self.retry_manager.cleanup(camera_name)
                    log.info(f"Removed camera {camera_name} after max retries")
                    break

        # Final cleanup
        self._cleanup_camera(camera)
        self.retry_manager.cleanup(camera.name)
    
    def _cleanup_camera(self, camera: Camera) -> None:
        """Clean up camera resources"""
        try:
            # Store frozen frame before cleanup if we have a last frame
            if camera.connection_lost and camera.last_frame is not None:
                try:
                    # Convert to grayscale and blur for the frozen effect
                    gray = cv2.cvtColor(camera.last_frame, cv2.COLOR_BGR2GRAY)
                    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
                    # Convert back to BGR for display
                    camera.frozen_frame = cv2.cvtColor(blurred, cv2.COLOR_GRAY2BGR)
                    camera.frozen_frame_time = time.time()
                    log.debug(f"Updated frozen frame for {camera.name}")
                except Exception as e:
                    log.error(f"Error creating frozen frame for {camera.name}: {e}")
                    # If we can't create a frozen frame, just use the last frame
                    camera.frozen_frame = camera.last_frame.copy()
                    camera.frozen_frame_time = time.time()
            
            if camera.pipeline:
                # Set pipeline to NULL state with timeout
                ret = camera.pipeline.set_state(Gst.State.NULL)
                if ret == Gst.StateChangeReturn.ASYNC:
                    # Wait for state change to complete
                    ret = camera.pipeline.get_state(timeout=Gst.SECOND)
                    if ret[0] != Gst.StateChangeReturn.SUCCESS:
                        log.warning(f"Failed to set pipeline to NULL state for {camera.name}")
                
                # Clear references
                camera.pipeline = None
                camera.sink = None
            
            # Clear frame references
            if camera.previous_frame is not None:
                self.frame_pool.return_frame(camera.previous_frame)
                camera.previous_frame = None
                
            if camera.last_frame is not None:
                self.frame_pool.return_frame(camera.last_frame)
                camera.last_frame = None
            
            # Only clear frame times and bandwidth samples if we're removing the camera
            # (i.e., not just handling a connection loss)
            retry_state = self.retry_manager.get_retry_state(camera.name)
            if retry_state['retry_count'] >= self.retry_manager.max_retries:
                camera._frame_times.clear()
                camera._bandwidth_samples.clear()
            
        except Exception as e:
            log.error(f"Error during camera cleanup for {camera.name}: {e}")
            # Force clear references even if cleanup fails
            camera.pipeline = None
            camera.sink = None
            camera.previous_frame = None
            camera.last_frame = None

    def _select_main_camera(self) -> str:
        """Select the best camera to show as main view"""
        # First check for manually selected camera
        manual_main = next((name for name, camera in self.cameras.items() 
                           if camera.manual_main and not camera.connection_lost), None)
        if manual_main:
            # Reset all main_camera flags
            for camera in self.cameras.values():
                camera.main_camera = False
            # Set main_camera flag for manually selected camera
            self.cameras[manual_main].main_camera = True
            return manual_main

        # If no manual selection, check cooldown for auto-switching
        current_time = time.time()
        if current_time - self.last_tab_time < self.tab_cooldown:
            # Return current main camera if cooldown hasn't elapsed
            current_main = next((name for name, cam in self.cameras.items() 
                               if cam.main_camera and not cam.connection_lost), None)
            if current_main:
                return current_main

        # Update last tab time for next auto-switch
        self.last_tab_time = current_time

        best_camera = None
        best_score = -1

        for name, camera in self.cameras.items():
            # Skip inactive or disconnected cameras
            if not camera.active or camera.connection_lost:
                continue

            # Calculate a score based on multiple factors
            score = 0
            score += camera.face_count * 2  # Faces are important
            score += camera.motion_score    # Motion adds interest
            
            # Prefer recently activated cameras
            time_since_active = time.time() - camera.last_active_time
            score += max(0, 5 - time_since_active)  # Bonus for recent activity
            
            # If this was previously the main camera, give it a slight boost
            if camera.main_camera:
                score += 1

            if score > best_score:
                best_score = score
                best_camera = name

        # Reset all main_camera flags
        for camera in self.cameras.values():
            camera.main_camera = False
            
        # Set main_camera flag for the best camera
        if best_camera:
            self.cameras[best_camera].main_camera = True
            log.info(f"Selected {best_camera} as main camera with score {best_score}")

        return best_camera

    def _any_camera_active(self) -> bool:
        """Check if any camera is currently active"""
        return any(camera.active for camera in self.cameras.values())

    def start_recording(self) -> None:
        """Start recording the debug view to an MKV file"""
        if self.recording:
            if self.recording_paused:
                self.resume_recording()
            else:
                log.info("Already recording")
            return
            
        # Create a timestamp-based filename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"recordings/{timestamp}.mkv"
        
        # Ensure recordings directory exists
        os.makedirs("recordings", exist_ok=True)
        
        # Get the resolution and frame rate of the first active camera for the pipeline
        # Default to 1080p@30fps if no active cameras
        width, height = 1920, 1080
        framerate = 30
        for camera in self.cameras.values():
            if camera.active and camera.input_resolution:
                width, height = camera.input_resolution
                framerate = int(camera.frame_rate)  # Use actual camera frame rate
                break
        
        # Create GStreamer pipeline for recording
        pipeline_str = (
            'appsrc name=src is-live=true format=time do-timestamp=true ! '
            f'video/x-raw,format=BGR,width={width},height={height},framerate={framerate}/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'x264enc speed-preset=superfast tune=zerolatency bitrate=10000 key-int-max=60 ! '
            'h264parse ! matroskamux ! '
            f'filesink location={output_file}'
        )
        
        log.info(f"Creating recording pipeline: {pipeline_str}")
        
        try:
            self.recording_pipeline = Gst.parse_launch(pipeline_str)
            self.recording_src = self.recording_pipeline.get_by_name('src')
            
            # Configure appsrc for controlled pushing
            self.recording_src.set_property('emit-signals', True)
            self.recording_src.set_property('block', False)  # Non-blocking mode
            self.recording_src.set_property('max-bytes', 0)  # No limit on queue size
            self.recording_src.set_property('format', Gst.Format.TIME)
            
            # Start the pipeline
            self.recording_pipeline.set_state(Gst.State.PLAYING)
            self.recording = True
            self.recording_paused = False
            self.frame_count = 0
            self.start_time = time.time()
            self.pause_start_time = 0  # Initialize pause start time
            log.info(f"Started recording to {output_file}")
        except Exception as e:
            log.error(f"Failed to start recording: {e}")
            self.recording = False
            self._cleanup_recording_pipeline()

    def pause_recording(self) -> None:
        """Pause the current recording without stopping the pipeline"""
        if not self.recording or self.recording_paused:
            return
            
        log.info("Pausing recording")
        self.recording_paused = True
        self.pause_start_time = time.time()
        
        # Actually pause the pipeline
        if self.recording_pipeline:
            self.recording_pipeline.set_state(Gst.State.PAUSED)
            log.info("Pipeline paused")

    def resume_recording(self) -> None:
        """Resume a paused recording"""
        if not self.recording or not self.recording_paused:
            return
            
        log.info("Resuming recording")
        self.recording_paused = False
        
        # Resume the pipeline
        if self.recording_pipeline:
            self.recording_pipeline.set_state(Gst.State.PLAYING)
            # Update start time to account for the pause duration
            pause_duration = time.time() - self.pause_start_time
            self.start_time += pause_duration
            log.info(f"Pipeline resumed after {pause_duration:.2f}s pause")

    def stop_recording(self) -> None:
        """Stop recording"""
        if not self.recording:
            return
            
        log.info("Stopping recording")
        
        # Calculate recording duration
        duration = time.time() - self.start_time
        if self.recording_paused:
            # Subtract the time spent in pause
            pause_duration = time.time() - self.pause_start_time
            duration -= pause_duration
        
        log.info(f"Recording stopped. Duration: {duration:.2f} seconds, Frames: {self.frame_count}")
        
        # Properly clean up the pipeline
        self._cleanup_recording_pipeline()
        
        self.recording = False
        self.recording_paused = False
        
        # Run garbage collection after stopping recording
        gc.collect()
    
    def _cleanup_recording_pipeline(self) -> None:
        """Clean up recording pipeline resources"""
        if self.recording_pipeline:
            # Send EOS and wait for it to propagate
            self.recording_pipeline.send_event(Gst.Event.new_eos())
            
            # Set pipeline to NULL state
            self.recording_pipeline.set_state(Gst.State.NULL)
            
            # Clear references
            self.recording_pipeline = None
            self.recording_src = None

    def start_streaming(self) -> None:
        """Start streaming to Twitch"""
        if self.streaming:
            log.info("Streaming is already active")
            return
            
        if not self.twitch_stream_key:
            log.error("Cannot start streaming: No Twitch stream key found in secrets.yaml")
            return
            
        # Create GStreamer pipeline for streaming to Twitch
        pipeline_str = (
            'appsrc name=src is-live=true format=time do-timestamp=true ! '
            'video/x-raw,format=BGR,width=1920,height=1080,framerate=60/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'x264enc speed-preset=veryfast tune=zerolatency bitrate=3500 key-int-max=60 ! '
            'h264parse ! flvmux streamable=true ! '
            f'rtmpsink location=rtmp://live.twitch.tv/app/{self.twitch_stream_key} sync=false'
        )
        
        log.info("Creating streaming pipeline")
        
        try:
            self.streaming_pipeline = Gst.parse_launch(pipeline_str)
            self.streaming_src = self.streaming_pipeline.get_by_name('src')
            
            # Configure appsrc for controlled pushing
            self.streaming_src.set_property('emit-signals', True)
            self.streaming_src.set_property('block', False)  # Non-blocking mode
            self.streaming_src.set_property('max-bytes', 0)  # No limit on queue size
            self.streaming_src.set_property('format', Gst.Format.TIME)
            
            # Start the pipeline
            self.streaming_pipeline.set_state(Gst.State.PLAYING)
            self.streaming = True
            log.info("Started streaming to Twitch")
        except Exception as e:
            log.error(f"Failed to start streaming: {e}")
            self.streaming = False
            self._cleanup_streaming_pipeline()
    
    def _cleanup_streaming_pipeline(self) -> None:
        """Clean up streaming pipeline resources"""
        if self.streaming_pipeline:
            # Send EOS and wait for it to propagate
            self.streaming_pipeline.send_event(Gst.Event.new_eos())
            
            # Set pipeline to NULL state
            self.streaming_pipeline.set_state(Gst.State.NULL)
            
            # Clear references
            self.streaming_pipeline = None
            self.streaming_src = None
    
    def stop_streaming(self) -> None:
        """Stop streaming to Twitch"""
        if not self.streaming:
            return
            
        log.info("Stopping Twitch stream")
        
        # Properly clean up the pipeline
        self._cleanup_streaming_pipeline()
        
        self.streaming = False
        
        # Run garbage collection after stopping streaming
        gc.collect()
    
    def toggle_streaming(self) -> None:
        """Toggle streaming on/off"""
        if self.streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
            
    def push_frame_to_recording(self, frame: np.ndarray) -> None:
        """Push a frame to the recording pipeline using zero-copy approach"""
        if not self.recording or self.recording_paused or self.recording_src is None:
            return
            
        # Increment frame counter
        self.frame_count += 1
        
        # Get frame size and dimensions
        height, width = frame.shape[:2]
        
        # Find the active camera to get its frame rate
        framerate = 30  # Default to 30fps
        for camera in self.cameras.values():
            if camera.active and camera.last_frame is frame:  # Check if this is the source frame
                framerate = int(camera.frame_rate)
                break
        
        # Create GStreamer buffer directly from frame data
        gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        
        # Set buffer timestamp
        duration = 1 / framerate * Gst.SECOND  # Use actual frame rate
        pts = (time.time() - self.start_time) * Gst.SECOND
        gst_buffer.pts = pts
        gst_buffer.duration = duration
        
        
        # Push buffer to pipeline
        ret = self.recording_src.emit('push-buffer', gst_buffer)
        
        # Explicitly release reference to help garbage collection
        gst_buffer = None
        
        if ret != Gst.FlowReturn.OK:
            log.warning(f"Error pushing buffer to recording: {ret}")
    
    def push_frame_to_streaming(self, frame: np.ndarray) -> None:
        """Push a frame to the streaming pipeline using zero-copy approach"""
        if not self.streaming or self.streaming_src is None:
            return
        
        # Create GStreamer buffer directly from frame data
        gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        
        # Set buffer timestamp
        duration = 1 / 60 * Gst.SECOND  # Using 60 fps
        pts = time.time() * Gst.SECOND
        gst_buffer.pts = pts
        gst_buffer.duration = duration
        
        # Push buffer to pipeline
        ret = self.streaming_src.emit('push-buffer', gst_buffer)
        
        # Explicitly release reference to help garbage collection
        gst_buffer = None
        
        if ret != Gst.FlowReturn.OK:
            log.warning(f"Error pushing buffer to streaming: {ret}")
    
    async def _create_debug_view(self) -> np.ndarray:
        """Create debug view based on current view mode with memory optimizations"""
        # For OUTPUT mode
        if self.view_mode == ViewMode.OUTPUT:
            return await self._create_output_view()
        
        # For INPUT mode
        return await self._create_input_view()
        
    async def _create_output_view(self) -> np.ndarray:
        """Create output view with memory optimizations"""
        # Prepare output frames - get from pool
        output = self.frame_pool.get_frame((1080, 1920, 3))
        clean_output = None  # Will be set to original resolution frame
        
        # Initialize to black
        output.fill(0)
        
        # Select main camera
        main_camera_name = self._select_main_camera()
        
        # If no main camera, return blank screen with UI
        if main_camera_name is None:
            # Create clean output at 1080p for recording
            clean_output = self.frame_pool.get_frame((1080, 1920, 3))
            clean_output.fill(0)
            self.clean_frame_for_recording = clean_output
            output = self._add_ui_elements(output)
            return output
        
        # Update main_camera flags
        for name, camera in self.cameras.items():
            camera.main_camera = (name == main_camera_name)
        
        # Get main camera frame
        main_camera = self.cameras[main_camera_name]
        main_frame = main_camera.last_frame if not main_camera.connection_lost else main_camera.frozen_frame
        
        if main_frame is None:
            # Create clean output at 1080p for recording
            clean_output = self.frame_pool.get_frame((1080, 1920, 3))
            clean_output.fill(0)
            self.clean_frame_for_recording = clean_output
            output = self._add_ui_elements(output)
            return output
        
        # Store original frame for recording
        clean_output = main_frame
        
        # Scale main frame to 1080p for display
        scaled_frame = self.frame_pool.get_frame((1080, 1920, 3))
        cv2.resize(main_frame, (1920, 1080), dst=scaled_frame)
        
        # Copy scaled frame to output
        np.copyto(output, scaled_frame)
        
        # Return scaled frame to pool
        self.frame_pool.return_frame(scaled_frame)
        
        # Add smaller overlays for active cameras (excluding main)
        pip_width = 480  # 1/4 of screen width
        pip_height = 270  # Keep 16:9 aspect ratio
        padding = 20  # Space between PiP windows
        
        # Get active cameras excluding main
        active_cameras = [(name, camera) for name, camera in self.cameras.items() 
                         if name != main_camera_name and camera.active and not camera.connection_lost and 
                         (camera.last_frame is not None or camera.frozen_frame is not None)]
        
        # Prepare PiP frame from pool once
        pip_frame = self.frame_pool.get_frame((pip_height, pip_width, 3))
        
        for i, (name, camera) in enumerate(active_cameras):
            # Calculate position for PiP
            x = 1920 - pip_width - padding
            y = padding + i * (pip_height + padding)
            
            # Skip if we run out of vertical space
            if y + pip_height > 1080:
                break
            
            # Get frame to display (use frozen frame if connection lost)
            display_frame = camera.frozen_frame if camera.connection_lost else camera.last_frame
            
            # Scale camera frame to PiP size
            cv2.resize(display_frame, (pip_width, pip_height), dst=pip_frame)
            
            # Insert PiP into output frames with overlay effect
            region = output[y:y+pip_height, x:x+pip_width]
            cv2.addWeighted(pip_frame, 0.8, region, 0.2, 0, dst=region)
        
        # Return PiP frame to pool
        self.frame_pool.return_frame(pip_frame)
        
        # Store clean output for recording
        self.clean_frame_for_recording = clean_output
        
        # Add UI elements to display view
        output = self._add_ui_elements(output)
        
        return output
        
    async def _create_input_view(self) -> np.ndarray:
        """Create grid view with memory optimizations"""
        # Collect frames based on view mode
        frames = []
        clean_frames = []
        
        for name, camera in self.cameras.items():
            # Skip if no frame available
            if camera.last_frame is None and camera.frozen_frame is None:
                continue
                
            # Get frame to display (use frozen frame if connection lost)
            display_frame = camera.frozen_frame if camera.connection_lost else camera.last_frame
                
            # Scale frame to 1080p for display
            display_frame_scaled = self.frame_pool.get_frame((1080, 1920, 3))
            cv2.resize(display_frame, (1920, 1080), dst=display_frame_scaled)
            
            # Add camera name overlay
            self._add_camera_overlay(display_frame_scaled, camera)
            
            # Add to frames list
            frames.append((name, display_frame_scaled))
            clean_frames.append((name, display_frame))  # Use original frame for recording
        
        # If no frames, return blank screen
        if not frames:
            blank = self.frame_pool.get_frame((1080, 1920, 3))
            blank.fill(0)
            self.clean_frame_for_recording = blank
            return self._add_ui_elements(blank)
        
        # Calculate grid layout
        n = len(frames)
        grid_size = int(np.ceil(np.sqrt(n)))
        cell_w = 1920 // grid_size
        cell_h = 1080 // grid_size
        
        # Create output frames
        output = self.frame_pool.get_frame((1080, 1920, 3))
        clean_output = self.frame_pool.get_frame((1080, 1920, 3))
        
        # Initialize to black
        output.fill(0)
        clean_output.fill(0)
        
        # Create a single reusable cell frame
        cell_frame = self.frame_pool.get_frame((cell_h, cell_w, 3))
        
        # Place frames in grid
        for i, ((name, frame), (_, clean_frame)) in enumerate(zip(frames, clean_frames)):
            y = (i // grid_size) * cell_h
            x = (i % grid_size) * cell_w
            
            # Resize frame to cell size for display
            cv2.resize(frame, (cell_w, cell_h), dst=cell_frame)
            output[y:y+cell_h, x:x+cell_w] = cell_frame
            
            # For clean output, resize original frame to cell size
            cv2.resize(clean_frame, (cell_w, cell_h), dst=cell_frame)
            clean_output[y:y+cell_h, x:x+cell_w] = cell_frame
            
            # Draw thick border if this is the main camera, accounting for the top bar
            if self.cameras[name].main_camera:
                line_width = 4
                cv2.rectangle(
                    output,
                    (x + line_width-1, y + line_width-1),
                    (x + cell_w - line_width+1, y + cell_h - line_width+1),
                    (0, 255, 255), line_width
                )
        
        # Return cell frame to pool
        self.frame_pool.return_frame(cell_frame)
        
        # Return display frames to pool
        for _, frame in frames:
            self.frame_pool.return_frame(frame)
        
        # Store clean output for recording
        self.clean_frame_for_recording = clean_output
        
        # Add UI elements
        output = self._add_ui_elements(output)
        
        return output
    
    def _add_camera_overlay(self, frame: np.ndarray, camera: Camera) -> None:
        """Add camera name overlay to frame, with improved legibility and new metrics layout"""
        # Set font parameters for better visibility
        font_scale = 1.1  # Larger font
        font_thickness = 2  # Thicker text
        font = cv2.FONT_HERSHEY_SIMPLEX
        line_type = cv2.LINE_AA  # Antialiased

        # Camera name (simulate bold by drawing twice)
        name_text = camera.name
        
        # Add connection status if disconnected
        status_text = ""
        if camera.connection_lost:
            retry_state = self.retry_manager.get_retry_state(camera.name)
            if retry_state['retry_count'] < self.retry_manager.max_retries:
                time_until_retry = int(max(0, retry_state['next_retry_time'] - time.time()))
                status_text = f"WAITING {time_until_retry}s..."
            else:
                status_text = "DISCONNECTED"
        
        # Metrics
        fps = int(camera.frame_rate)
        motion = f"Motion: {camera.motion_score:.2f}"
        faces = f"Faces: {camera.face_count}"
        status = "active" if camera.active else "standby"
        metrics = f"{status.upper()} | FPS: {fps} | {motion} | {faces}"

        # Calculate text sizes
        name_size = cv2.getTextSize(name_text, font, font_scale, font_thickness)[0]
        status_size = cv2.getTextSize(status_text, font, font_scale * 0.8, font_thickness)[0] if status_text else (0, 0)
        metrics_size = cv2.getTextSize(metrics, font, font_scale * 0.8, font_thickness)[0]

        # Icon size
        icon_width = 0
        icon_height = 0
        if self.pushpin_icon is not None:
            icon_width = self.pushpin_icon.shape[1]
            icon_height = self.pushpin_icon.shape[0]

        # Calculate total width needed for the background box
        box_padding_w = 10
        total_width = max(
            name_size[0] + icon_width + box_padding_w*2,  # Name + icon + padding
            status_size[0] + box_padding_w*2,  # Status + padding
            metrics_size[0] + box_padding_w*2   # Metrics + padding
        )
        box_padding_h = 15
        total_height = name_size[1] + metrics_size[1] + (box_padding_h * 2)  # Text heights + padding

        # Calculate position for the background box
        box_margin = 10
        box_x = box_margin
        box_y = box_margin

        # Draw semi-transparent background box
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x, box_y), (box_x + total_width, box_y + total_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        # Draw camera name with icon
        text_x = box_x + box_padding_w
        text_y = box_y + name_size[1] + 5

        # Draw name text (simulate bold)
        cv2.putText(frame, name_text, (text_x+1, text_y+1), font, font_scale, (0,0,0), font_thickness+2, line_type)
        cv2.putText(frame, name_text, (text_x, text_y), font, font_scale, (255,255,255), font_thickness, line_type)

        # Add lock icon if camera is manually selected
        if camera.manual_main and self.pushpin_icon is not None:
            icon_x = text_x + name_size[0] + 5
            icon_y = text_y - icon_height
            self._overlay_image_with_alpha(frame, self.pushpin_icon, icon_x, icon_y)

        # Draw status text if disconnected
        if status_text:
            status_x = text_x + name_size[0] + box_padding_h
            status_y = text_y
            # Use red for disconnected, yellow for reconnecting
            status_color = (0, 0, 255) if "DISCONNECTED" in status_text else (0, 255, 255)
            cv2.putText(frame, status_text, (status_x+1, status_y+1), font, font_scale*0.8, (0,0,0), font_thickness+2, line_type)
            cv2.putText(frame, status_text, (status_x, status_y), font, font_scale*0.8, status_color, font_thickness, line_type)

        # Draw metrics line with 5px gap
        metrics_x = text_x
        metrics_y = text_y + metrics_size[1] + 15
        cv2.putText(frame, metrics, (metrics_x+1, metrics_y+1), font, font_scale*0.8, (0,0,0), font_thickness+2, line_type)
        cv2.putText(frame, metrics, (metrics_x, metrics_y), font, font_scale*0.8, (200,200,200), font_thickness, line_type)

        # Bandwidth graph: make it the same size as the metrics container
        graph_x = box_x + total_width + box_margin
        graph_y = box_y
        graph_w = 150
        graph_h = total_height
        camera.draw_bandwidth_graph(frame, graph_x, graph_y, graph_w, graph_h)
        
        
        
        
        
        
        
    
    def _overlay_image_with_alpha(self, background, overlay, x, y):
        """Overlay an RGBA image on a BGR background at (x, y)"""
        h, w = overlay.shape[:2]
        if overlay.shape[2] == 4:  # Has alpha channel
            alpha = overlay[:, :, 3] / 255.0
            for c in range(3):
                background[y:y+h, x:x+w, c] = (
                    alpha * overlay[:, :, c] +
                    (1 - alpha) * background[y:y+h, x:x+w, c]
                )
        else:
            background[y:y+h, x:x+w] = overlay
    
    def _add_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Add UI elements with resolution-independent scaling"""
        # Get original frame dimensions
        h, w = frame.shape[:2]
        base_height = 1080  # Reference height for UI scaling
        
        # Scale UI elements based on frame height
        scale_factor = h / base_height
        
        # Constants for UI elements (scaled)
        top_bar_height = int(60 * scale_factor)
        bottom_bar_height = int(50 * scale_factor)
        tab_width = w // 2  # 2 view modes
        
        # Create a canvas with extra space for UI
        canvas = self.frame_pool.get_frame((h + top_bar_height + bottom_bar_height, w, 3))
        canvas.fill(0)
        
        # Copy frame to middle of canvas
        canvas[top_bar_height:top_bar_height+h] = frame
        
        # Create top and bottom toolbars (dark gray)
        canvas[:top_bar_height].fill(40)
        canvas[top_bar_height+h:].fill(40)
        
        # Add tabs for each view mode
        view_modes = [
            (1, "RAW INPUT", ViewMode.INPUT),
            (2, "OUTPUT", ViewMode.OUTPUT)
        ]
        
        for i, (num, name, mode) in enumerate(view_modes):
            x_start = i * tab_width
            x_end = (i + 1) * tab_width
            
            # Highlight active tab
            if mode == self.view_mode:
                cv2.rectangle(canvas, (x_start, 0), (x_end, top_bar_height), (0, 120, 255), -1)
            
            # Add tab border
            cv2.rectangle(canvas, (x_start, 0), (x_end, top_bar_height), (100, 100, 100), 1)
            
            # Add tab text with number
            tab_text = f"{num}: {name}"
            font_scale = 0.8 * scale_factor
            thickness = max(1, int(2 * scale_factor))
            text_size = cv2.getTextSize(tab_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x_start + (tab_width - text_size[0]) // 2
            text_y = (top_bar_height + text_size[1]) // 2
            
            # Use white text for active tab, gray for inactive
            text_color = (255, 255, 255) if mode == self.view_mode else (180, 180, 180)
            cv2.putText(canvas, tab_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness)
        
        # Add keyboard shortcuts to bottom toolbar
        shortcuts = [
            "1-2: Switch Views",
            "A: Auto-Recording",
            "S: Start/Stop Recording",
            "T: Toggle Streaming",
            "M: Remux Last Recording",
            "TAB: Cycle Cameras/Auto (OUTPUT)",
            "Q: Quit"
        ]
        
        shortcut_width = w // len(shortcuts)
        for i, shortcut in enumerate(shortcuts):
            x_start = i * shortcut_width
            font_scale = 0.7 * scale_factor
            thickness = max(1, int(scale_factor))
            text_size = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x_start + (shortcut_width - text_size[0]) // 2
            text_y = top_bar_height + h + (bottom_bar_height + text_size[1]) // 2
            cv2.putText(canvas, shortcut, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
        
        # Create status indicators if needed
        indicators = []
        
        # Add main camera name if in OUTPUT mode
        if self.view_mode == ViewMode.OUTPUT:
            main_camera_name = self._select_main_camera()
            if main_camera_name:
                # Check if we're in auto mode or manual selection
                any_manual = any(cam.manual_main for cam in self.cameras.values())
                if any_manual:
                    # Show which camera is manually selected
                    indicators.append(("CAMERA", f"{main_camera_name} [manual]", (255, 255, 255), False))
                else:
                    # Show that we're in auto mode
                    indicators.append(("CAMERA", f"{main_camera_name} [auto]", (200, 200, 200), False))
        
        # Add recording status
        if self.recording:
            status = "PAUSED" if self.recording_paused else "REC"
            color = (255, 165, 0) if self.recording_paused else (0, 0, 255)  # Orange for paused, red for recording
            indicators.append((status, "", color, True))
        
        # Add streaming status
        if self.streaming:
            indicators.append(("LIVE", "", (0, 0, 255), True))  # Red for live
            
        # Add auto-recording indicator if enabled
        if self.auto_recording:
            # Determine the auto-recording status
            any_active = self._any_camera_active()
            
            if self.recording and not self.recording_paused and any_active:
                # Auto-recording is actively recording
                auto_rec_text = "AUTO-REC: ACTIVE"
                auto_rec_color = (50, 200, 50)  # Green
            elif self.recording and self.recording_paused and not any_active:
                # Auto-recording is paused due to no activity
                auto_rec_text = "AUTO-REC: PAUSED"
                auto_rec_color = (200, 200, 50)  # Yellow
            else:
                # Auto-recording is enabled but not currently recording
                auto_rec_text = "AUTO-REC: READY"
                auto_rec_color = (100, 150, 100)  # Light green
                
            indicators.append((auto_rec_text, "", auto_rec_color, True))
        
        # If we have indicators to show, create a status bar
        if indicators:
            # Calculate total width needed for the status bar
            total_width = 0
            padding = int(15 * scale_factor)  # Padding between indicators
            
            for label, value, _, has_bg in indicators:
                display_text = f"{label}" if not value else f"{label}: {value}"
                font_scale = 0.8 * scale_factor
                thickness = max(1, int(2 * scale_factor))
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                # Width for text + spacing + background if needed
                indicator_width = text_size[0] + padding
                if has_bg:
                    indicator_width += int(20 * scale_factor)  # Extra space for background
                total_width += indicator_width
            
            # Create a semi-transparent background for the status bar
            status_bar_height = int(40 * scale_factor)
            status_bar_y = top_bar_height + int(10 * scale_factor)
            status_bar_x = w - total_width - padding
            
            # Draw the rectangle for status bar background
            cv2.rectangle(canvas, 
                        (status_bar_x, status_bar_y), 
                        (w - padding, status_bar_y + status_bar_height), 
                        (40, 40, 40), -1)
            
            # Draw a subtle border around the status bar
            cv2.rectangle(canvas, 
                        (status_bar_x, status_bar_y), 
                        (w - padding, status_bar_y + status_bar_height), 
                        (100, 100, 100), 1)
            
            # Start position for the leftmost indicator
            x_pos = status_bar_x + padding
            y_pos = status_bar_y + status_bar_height//2 + int(5 * scale_factor)  # Adjust for text baseline
            
            # Draw indicators from left to right
            for label, value, color, has_bg in indicators:
                display_text = f"{label}" if not value else f"{label}: {value}"
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # For status indicators (REC, LIVE, PAUSED), add a colored background
                if has_bg:
                    # Calculate background rectangle dimensions
                    bg_padding = int(10 * scale_factor)
                    bg_x = x_pos - bg_padding
                    bg_y = status_bar_y + int(5 * scale_factor)
                    bg_w = text_size[0] + bg_padding * 2
                    bg_h = status_bar_height - int(10 * scale_factor)
                    
                    # Draw colored background
                    cv2.rectangle(canvas, 
                                (bg_x, bg_y), 
                                (bg_x + bg_w, bg_y + bg_h), 
                                color, -1)
                    
                    # Draw text in white on colored background
                    cv2.putText(canvas, display_text, (x_pos, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    
                    # Move right for next indicator with padding
                    x_pos += bg_w + padding
                else:
                    # Draw regular text for camera name
                    cv2.putText(canvas, display_text, (x_pos, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
                    
                    # Move right for next indicator with padding
                    x_pos += text_size[0] + padding
        
        return canvas

    async def start(self) -> None:
        """Start the stream processing with explicit memory management"""
        log.info("Starting workshop stream")
        self.running = True
        
        # Create tasks for input view
        tasks = [
            asyncio.create_task(self._capture_frames(camera))
            for camera in self.cameras.values()
        ]
        
        # Add debug viewer task if debug mode is enabled
        if self.debug:
            tasks.append(asyncio.create_task(self._run_debug_viewer()))
        
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutting down...")
            self.running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
            self.stop()
    
    async def _run_debug_viewer(self):
        """Run debug viewer with memory management"""
        try:
            while self.running:
                # Update bandwidth for input view, including disconnected ones
                current_time = time.time()
                for camera in self.cameras.values():
                    if camera.connection_lost:
                        # Force bandwidth update for disconnected cameras
                        camera.update_bandwidth(0, current_time)
                
                # Create debug view
                view = await self._create_debug_view()
                
                # Handle auto-recording based on camera activity
                self._handle_auto_recording()
                
                # Handle recording and streaming
                self._handle_recording_and_streaming(view)
                
                # Show the view
                cv2.imshow('Debug View', view)
                
                # Process keyboard input
                key = cv2.waitKey(1) & 0xFF
                self._handle_keyboard_input(key)
                
                # Return view frame to pool
                self.frame_pool.return_frame(view)
                
                # Run garbage collection periodically
                if self.frame_count % 600 == 0:  # Every 10 seconds at 60fps
                    gc.collect()
                
                # Sleep to maintain frame rate
                await asyncio.sleep(1/60)
        finally:
            cv2.destroyAllWindows()
            if self.recording:
                self.stop_recording()
            if self.streaming:
                self.stop_streaming()
    
    def _handle_auto_recording(self):
        """Handle auto-recording logic"""
        if not self.auto_recording:
            return
            
        any_active = self._any_camera_active()
        
        # Start recording if any camera is active and we're not recording
        if any_active and not self.recording:
            self.start_recording()
        
        # Resume recording if any camera is active and recording is paused
        elif any_active and self.recording and self.recording_paused:
            self.resume_recording()
        
        # Pause recording if no camera is active and recording is not paused
        elif not any_active and self.recording and not self.recording_paused:
            # Store the time when we pause
            self.pause_start_time = time.time()
            self.pause_recording()
    
    def _handle_recording_and_streaming(self, view):
        """Handle recording and streaming of frames"""
        # Get the main camera's frame for recording/streaming
        main_camera_name = self._select_main_camera()
        recording_frame = None
        
        if main_camera_name and self.cameras[main_camera_name].last_frame is not None:
            recording_frame = self.cameras[main_camera_name].last_frame
        else:
            # If no main camera frame, create a blank frame
            recording_frame = self.frame_pool.get_frame((1080, 1920, 3))
            recording_frame.fill(0)
        
        # Push frame to recording if active
        if self.recording:
            self.push_frame_to_recording(recording_frame)
        
        # Push frame to streaming if active
        if self.streaming:
            self.push_frame_to_streaming(recording_frame)
            
        # Return blank frame to pool if we created one
        if recording_frame is not None and main_camera_name is None:
            self.frame_pool.return_frame(recording_frame)
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input for the debug viewer"""
        if key == ord('q'):
            raise KeyboardInterrupt
        
        # Number keys for view modes (1-2)
        elif key == ord('1'):
            self.view_mode = ViewMode.INPUT
        elif key == ord('2'):
            self.view_mode = ViewMode.OUTPUT
        
        # Letter shortcuts for other functions
        elif key == ord('a'):  # 'a' to toggle auto-recording
            self.auto_recording = not self.auto_recording
            log.info(f"Auto-recording {'enabled' if self.auto_recording else 'disabled'}")
        elif key == ord('t'):  # 't' to toggle streaming
            self.toggle_streaming()
        elif key == ord('s'):  # 's' to start/stop recording
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('p'):  # 'p' to pause/resume recording
            if self.recording:
                if self.recording_paused:
                    self.resume_recording()
                else:
                    self.pause_recording()
        elif key == ord('l'):  # 'l' to lock/unlock current camera
            self._toggle_lock_current_camera()
        elif key == ord('\t'):  # TAB to cycle main camera
            self._cycle_main_camera()
        elif key == ord('m'):  # 'm' to remux last recording
            self._remux_last_recording()
    
    def _toggle_lock_current_camera(self):
        """Lock or unlock the current main camera"""
        # Get current main camera
        main_camera_name = self._select_main_camera()
        if not main_camera_name:
            return
            
        # Get the camera
        camera = self.cameras[main_camera_name]
        
        # Toggle lock state
        camera.manual_main = not camera.manual_main
        
        # If we're locking this camera, unlock all others
        if camera.manual_main:
            for other_camera in self.cameras.values():
                if other_camera != camera:
                    other_camera.manual_main = False
            log.info(f"Locked on camera: {main_camera_name}")
        else:
            log.info(f"Unlocked camera: {main_camera_name}")
    
    def _cycle_main_camera(self):
        """Cycle through cameras for OUTPUT mode"""
        log.info("TAB pressed. Cycling camera selection.")
        
        # Get list of active cameras (excluding offline ones)
        active_cameras = [name for name, cam in self.cameras.items() 
                         if cam.active and not cam.connection_lost]
        
        if not active_cameras:
            return
        
        # Find current main camera
        current_main = next((name for name, cam in self.cameras.items() 
                           if cam.main_camera), None)
        
        if current_main in active_cameras:
            # Find next camera in cycle
            current_idx = active_cameras.index(current_main)
            next_idx = (current_idx + 1) % len(active_cameras)
            next_camera = active_cameras[next_idx]
            
            # Update main camera flags
            self.cameras[current_main].main_camera = False
            self.cameras[next_camera].main_camera = True
            log.info(f"Switched main camera to: {next_camera}")
        else:
            # If no main camera or current main is not active, select first active camera
            first_camera = active_cameras[0]
            self.cameras[first_camera].main_camera = True
            log.info(f"Selected first active camera: {first_camera}")
    
    def _remux_last_recording(self) -> None:
        """Remux the last recording from MKV to MP4 format"""
        try:
            # Get list of MKV files in recordings directory with their modification times
            mkv_files = []
            for f in os.listdir('recordings'):
                if f.endswith('.mkv'):
                    full_path = os.path.join('recordings', f)
                    mkv_files.append((full_path, os.path.getmtime(full_path)))
            
            if not mkv_files:
                log.warning("No MKV recordings found to remux")
                return
                
            # Sort by modification time (newest first) and get the most recent file
            mkv_files.sort(key=lambda x: x[1], reverse=True)
            last_mkv = mkv_files[0][0]
            output_mp4 = last_mkv.replace('.mkv', '.mp4')
            
            # Skip if MP4 already exists
            if os.path.exists(output_mp4):
                log.info(f"MP4 already exists: {output_mp4}")
                return
                
            log.info(f"Remuxing {last_mkv} to {output_mp4}")
            
            # Create GStreamer pipeline for remuxing
            pipeline_str = (
                f'filesrc location={last_mkv} ! '
                'matroskademux ! '
                'h264parse ! '
                'mp4mux ! '
                f'filesink location={output_mp4}'
            )
            
            # Create and run pipeline
            pipeline = Gst.parse_launch(pipeline_str)
            pipeline.set_state(Gst.State.PLAYING)
            
            # Wait for pipeline to finish
            bus = pipeline.get_bus()
            msg = bus.timed_pop_filtered(Gst.CLOCK_TIME_NONE, 
                                       Gst.MessageType.ERROR | Gst.MessageType.EOS)
            
            # Clean up
            pipeline.set_state(Gst.State.NULL)
            
            if msg:
                if msg.type == Gst.MessageType.ERROR:
                    err, debug = msg.parse_error()
                    log.error(f"Error during remuxing: {err.message}")
                    if os.path.exists(output_mp4):
                        os.remove(output_mp4)  # Clean up failed output
                else:
                    log.info(f"Successfully remuxed to {output_mp4}")
            
        except Exception as e:
            log.error(f"Error during remuxing: {e}")
            if os.path.exists(output_mp4):
                os.remove(output_mp4)  # Clean up failed output

    def stop(self) -> None:
        """Stop the stream processing and clean up resources"""
        self.running = False
        
        # Stop recording and streaming
        if self.recording:
            self.stop_recording()
        if self.streaming:
            self.stop_streaming()
        
        # Clean up cameras
        for camera in self.cameras.values():
            camera.cleanup()
        
        # No GStreamer buffers to clean up anymore
        
        # Clean up other resources
        if self.debug:
            cv2.destroyAllWindows()
        
        # Clear references to large objects
        self.clean_frame_for_recording = None
        self.output_frame = None
        
        # Run garbage collection
        gc.collect()

if __name__ == "__main__":
    # Quick test with debug viewer
    stream = WorkshopStream(debug=True)
    stream.add_camera("rtsp://192.168.1.155:8554/live", "iphone 13")
    stream.add_camera("rtsp://192.168.1.156:8554/live", "iphone xs")
    stream.add_camera("rtsp://192.168.1.114:8080/h264_pcm.sdp", "galaxy s21")
    
    print("Controls:")
    print("  1 - input view (all cameras)")
    print("  2 - output view (for streaming/recording)")
    print("  TAB - cycle main camera")
    print("  r - release manual control")
    print("  s - start/stop recording")
    print("  p - pause/resume recording")
    print("  a - toggle auto-recording")
    print("  t - toggle streaming to Twitch")
    print("  m - remux last recording")
    print("  q - quit")
    
    # Check if Twitch stream key is available
    if not stream.twitch_stream_key:
        print("\nWARNING: No Twitch stream key found in secrets.yaml")
        print("To enable streaming, add 'twitch_stream_key: \"your_key_here\"' to secrets.yaml")
    
    try:
        asyncio.run(stream.start())
    except KeyboardInterrupt:
        stream.stop()
        print("Exiting...")