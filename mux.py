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
import math  # Add import for math functions
import subprocess  # Add import for subprocess to run caffeinate
import atexit  # Add import for atexit to ensure cleanup
import threading
from pathlib import Path
import queue  # Add this import for the thread-safe queue

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s | %(levelname)s | %(message)s')
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
                    # If we can't create a frozen frame, just use the last frame
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
        
        # Add task manager for camera tasks
        self.camera_tasks: Dict[str, asyncio.Task] = {}
        self.main_task: Optional[asyncio.Task] = None
        
        # Add thread-safe camera operation queue for RTMP server
        self.camera_ops_queue = queue.Queue()
        
        # Add caffeinate process reference
        self._caffeinate_process = None
        
        # Register cleanup function to ensure sleep prevention is disabled on exit
        atexit.register(self._restore_sleep)
        
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
        
        # RTMP server integration
        self.rtmp_server = None
        self.rtmp_server_process = None
        self.rtmp_notify_fd = None
        self.rtmp_notify_thread = None

        # Add connection tracking for RTMP streams
        self.rtmp_connections = {}  # Map IP:port -> camera_name

    def _start_rtmp_server(self):
        """Start the RTMP server in the background"""
        try:
            # Import here to avoid circular imports
            from rtmp_srt_server import get_mediamtx, create_config, run_server
            
            # Get MediaMTX binary
            executable = get_mediamtx()
            
            # Create config file
            config_path = Path.cwd() / "mediamtx.yml"
            create_config(config_path)
            
            # Start server in background with output capture
            self.rtmp_server_process = subprocess.Popen(
                [str(executable), str(config_path)],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                bufsize=1  # Line buffered
            )
            
            # Start a thread to read and log the server output
            def log_server_output():
                while self.running and self.rtmp_server_process:
                    try:
                        line = self.rtmp_server_process.stdout.readline()
                        if line:
                            log.info(f"[RTMP Server] {line.strip()}")
                            
                            # Parse RTMP server messages for camera lifecycle events
                            if "is publishing to path" in line:
                                # Extract path from message
                                # Example: "is publishing to path 'live/mystream', 2 tracks (H264, MPEG-4 Audio)"
                                try:
                                    path = line.split("path '")[1].split("'")[0]
                                    rtsp_url = f"rtsp://127.0.0.1:8554/{path}"
                                    camera_name = f"RTMP-{path.split('/')[-1]}"
                                    
                                    # Extract connection info - looks like [conn 192.168.1.155:50058]
                                    if "[conn " in line:
                                        conn_info = line.split("[conn ")[1].split("]")[0]
                                        # Store the association between connection and camera
                                        self.rtmp_connections[conn_info] = camera_name
                                        log.info(f"Tracking RTMP connection {conn_info} for camera {camera_name}")
                                    
                                    # Add the camera operation to the thread-safe queue
                                    self.camera_ops_queue.put(("add", camera_name, rtsp_url))
                                    log.info(f"Queued add operation for camera: {camera_name}")
                                except Exception as e:
                                    log.error(f"Error parsing RTMP publish message: {e}")
                                    
                            elif "closed: EOF" in line and "RTMP" in line:
                                # RTMP connection closed, find and remove the corresponding camera
                                try:
                                    # Extract connection info from message
                                    # Example: "[conn 192.168.1.155:50058] closed: EOF"
                                    if "[conn " in line:
                                        conn_info = line.split("[conn ")[1].split("]")[0]
                                        
                                        # Find the camera associated with this connection
                                        camera_name = self.rtmp_connections.get(conn_info)
                                        
                                        if camera_name:
                                            # Add the remove operation to the queue
                                            self.camera_ops_queue.put(("remove", camera_name))
                                            log.info(f"RTMP connection {conn_info} closed, queued remove operation for camera: {camera_name}")
                                            
                                            # Remove from connection tracking
                                            del self.rtmp_connections[conn_info]
                                        else:
                                            # Fallback to the old method if we don't have tracking info
                                            rtmp_cameras = [name for name in self.cameras.keys() if name.startswith("RTMP-")]
                                            if len(rtmp_cameras) == 1:
                                                camera_name = rtmp_cameras[0]
                                                # Add the remove operation to the queue
                                                self.camera_ops_queue.put(("remove", camera_name))
                                                log.info(f"Queued remove operation for camera: {camera_name} (using fallback method)")
                                except Exception as e:
                                    log.error(f"Error handling RTMP close message: {e}")
                                    
                        elif self.rtmp_server_process.poll() is not None:
                            break
                    except Exception as e:
                        log.error(f"Error reading RTMP server output: {e}")
                        break
            
            threading.Thread(target=log_server_output, daemon=True).start()
            
            log.info("RTMP server started")
        except Exception as e:
            log.error(f"Failed to start RTMP server: {e}")
            raise

    async def start(self) -> None:
        """Start the stream processing with explicit memory management"""
        log.info("Starting workshop stream")
        self.running = True
        
        # Prevent sleep when starting
        self._prevent_sleep()
        
        # Start RTMP server
        self._start_rtmp_server()
        
        # Create tasks for existing cameras
        for camera in self.cameras.values():
            self._start_camera_task(camera)
        
        # Add debug viewer task if debug mode is enabled
        if self.debug:
            self.main_task = asyncio.create_task(self._run_debug_viewer())
        
        try:
            # Wait for the main task (debug viewer) to complete
            if self.main_task:
                await self.main_task
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutting down...")
            self.running = False
            # Cancel all camera tasks
            for task in self.camera_tasks.values():
                if not task.done():
                    task.cancel()
            # Cancel main task if it exists
            if self.main_task and not self.main_task.done():
                self.main_task.cancel()
            self.stop()

    async def _run_debug_viewer(self):
        """Run debug viewer with memory management"""
        try:
            while self.running:
                # Process any pending camera operations from RTMP server
                self._process_camera_ops()
                
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
    
    def _process_camera_ops(self):
        """Process camera operations from the thread-safe queue"""
        # Process up to 10 operations at a time to prevent blocking
        for _ in range(10):
            try:
                # Get operation from queue (non-blocking)
                op_type, *args = self.camera_ops_queue.get_nowait()
                
                if op_type == "add":
                    camera_name, rtsp_url = args
                    # Add the camera if it doesn't exist
                    if camera_name not in self.cameras:
                        log.info(f"Processing add operation: {camera_name} from {rtsp_url}")
                        self.add_camera(rtsp_url, camera_name)
                elif op_type == "remove":
                    camera_name = args[0]
                    # Remove the camera if it exists
                    if camera_name in self.cameras:
                        log.info(f"Processing remove operation: {camera_name}")
                        self.remove_camera(camera_name)
                
                # Mark task as done
                self.camera_ops_queue.task_done()
            except queue.Empty:
                # No more operations to process
                break
            except Exception as e:
                log.error(f"Error processing camera operation: {e}")

    def _start_camera_task(self, camera: Camera) -> None:
        """Start a task for a camera if it doesn't already have one"""
        if camera.name not in self.camera_tasks or self.camera_tasks[camera.name].done():
            self.camera_tasks[camera.name] = asyncio.create_task(self._capture_frames(camera))
            log.info(f"Started task for camera: {camera.name}")

    def _stop_camera_task(self, camera_name: str) -> None:
        """Stop a camera's task if it exists"""
        if camera_name in self.camera_tasks:
            task = self.camera_tasks[camera_name]
            if not task.done():
                task.cancel()
            del self.camera_tasks[camera_name]
            log.info(f"Stopped task for camera: {camera_name}")

    def add_camera(self, url: str, name: str) -> None:
        """Add a new camera to the stream"""
        camera = Camera(url=url, name=name)
        self.cameras[name] = camera
        # Initialize retry state for the new camera
        self.retry_manager.get_retry_state(name)
        log.info(f"Added camera: {name} @ {url}")
        
        # Start a task for the new camera if the stream is running
        if self.running:
            self._start_camera_task(camera)

    def remove_camera(self, name: str) -> None:
        """Remove a camera from the stream"""
        if name in self.cameras:
            # Stop the camera's task
            self._stop_camera_task(name)
            
            # Clean up camera resources
            camera = self.cameras[name]
            camera.cleanup()
            
            # Remove from cameras dict
            del self.cameras[name]
            
            # Clean up retry state
            self.retry_manager.cleanup(name)
            
            log.info(f"Removed camera: {name}")
            
            # If this was the main camera, select another one
            if camera.main_camera:
                new_main = self._select_main_camera()
                if new_main:
                    log.info(f"Switched main camera to {new_main} after removing {name}")
                else:
                    log.warning("No active cameras available to switch to")

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
        
        # Clear connection tracking
        self.rtmp_connections.clear()
        
        # Stop RTMP server
        if self.rtmp_server_process:
            try:
                self.rtmp_server_process.terminate()
                self.rtmp_server_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.rtmp_server_process.kill()
            except Exception as e:
                log.error(f"Error stopping RTMP server: {e}")
        
        # Restore sleep behavior
        self._restore_sleep()
        
        # Clean up other resources
        if self.debug:
            cv2.destroyAllWindows()
        
        # Clear references to large objects
        self.clean_frame_for_recording = None
        self.output_frame = None
        
        # Run garbage collection
        gc.collect()

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
        """Create and configure GStreamer pipeline for a camera using uridecodebin"""
        # Create GStreamer pipeline based on camera URL type
        if camera.url.isdigit():  # USB webcam
            pipeline_str = (
                f'avfvideosrc device-index={camera.url} ! '
                'video/x-raw,format=YUY2,width=1920,height=1080,framerate=60/1 ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=4096 drop=False'
            )
        else:  # RTSP/HTTP stream
            # pipeline_str = (
            #     f'uridecodebin uri={camera.url} name=src ! '
            #     'queue max-size-buffers=4096 max-size-bytes=0 max-size-time=0 ! '
            #     'videoconvert ! video/x-raw,format=BGR ! '
            #     'appsink name=sink emit-signals=True max-buffers=4096 drop=False'
            # )
            # os.environ["GST_RTSP_TRANSPORT"] = "udp"
            pipeline_str = (
                f'uridecodebin uri={camera.url} name=src '
                'src. ! queue max-size-time=10000000000 leaky=downstream ! '
                'videoconvert ! videorate max-rate=30 ! '
                'videoscale method=lanczos ! video/x-raw,width=1280,height=720,format=BGR ! '
                'videobalance contrast=1.1 brightness=0.05 ! '
                'appsink name=sink emit-signals=True drop=False sync=false '
                'src. ! queue max-size-time=10000000000 leaky=downstream ! '
                'audioconvert ! audioresample quality=10 ! volume volume=1.5 ! '
                'autoaudiosink sync=false'
            )
# GST_RTSP_TRANSPORT=tcp gst-launch-1.0 uridecodebin uri=rtsp://192.168.1.155:8554/live name=src \
#   src. ! queue max-size-time=10000000000 leaky=downstream ! videoconvert ! videorate max-rate=30 ! \
#   videoscale method=lanczos ! video/x-raw,width=1280,height=720 ! videobalance contrast=1.1 brightness=0.05 ! \
#   autovideosink sync=false \
#   src. ! queue max-size-time=10000000000 leaky=downstream ! \
#   audioconvert ! audioresample quality=10 ! volume volume=1.5 ! \
#   autoaudiosink sync=false
        log.debug(f"Creating pipeline for {camera.name}: {pipeline_str}")
        
        # Create and store the pipeline in the camera object
        camera.pipeline = Gst.parse_launch(pipeline_str)
        camera.sink = camera.pipeline.get_by_name('sink')
        
        # For network streams, we need to handle dynamic pad creation
        if not camera.url.isdigit():
            src = camera.pipeline.get_by_name('src')
            
            def on_pad_added(element, pad):
                # Get pad capabilities
                caps = pad.query_caps(None)
                if caps:
                    # Check if this is a video pad
                    if caps.is_subset(Gst.Caps.from_string('video/x-raw')):
                        # Get the queue element
                        queue = camera.pipeline.get_by_name('queue0')
                        if queue:
                            # Link the pad to the queue
                            pad.link(queue.get_static_pad('sink'))
                            log.debug(f"Linked video pad for {camera.name}")
            
            # Connect to pad-added signal
            src.connect('pad-added', on_pad_added)
        
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
                framerate = max(1, int(camera.frame_rate))  # Ensure framerate is at least 1
                break
        
        # Create GStreamer pipeline for recording to MKV file with audio
        pipeline_str = (
            # Video branch
            'appsrc name=videosrc is-live=true format=time do-timestamp=true ! '
            f'video/x-raw,format=BGR,width={width},height={height},framerate={framerate}/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'x264enc speed-preset=superfast tune=zerolatency bitrate=10000 key-int-max=60 ! '
            'h264parse ! '
            'queue ! mux. '
            # Audio branch - mix audio from all active cameras
            'audiomixer name=mixer ! '
            'audioconvert ! audioresample ! '
            'audiorate ! audio/x-raw,rate=48000 ! '
            'avenc_aac bitrate=192000 ! aacparse ! '
            'queue ! mux. '
            # Muxer
            'matroskamux name=mux ! '
            f'filesink location={output_file} sync=false'
        )
        
        # Add audio sources for each active camera
        for i, camera in enumerate(self.cameras.values()):
            if camera.active and not camera.connection_lost:
                # Add audio source for this camera
                pipeline_str = (
                    f'uridecodebin uri={camera.url} name=src{i} '
                    f'src{i}. ! queue ! audioconvert ! audio/x-raw,rate=48000 ! '
                    'audiorate ! mixer. ' + pipeline_str
                )
        
        log.info(f"Creating recording pipeline: {pipeline_str}")
        
        try:
            self.recording_pipeline = Gst.parse_launch(pipeline_str)
            self.recording_src = self.recording_pipeline.get_by_name('videosrc')
            
            # Configure appsrc for controlled pushing
            self.recording_src.set_property('emit-signals', True)
            self.recording_src.set_property('block', False)  # Non-blocking mode
            self.recording_src.set_property('max-bytes', 0)  # No limit on queue size
            self.recording_src.set_property('format', Gst.Format.TIME)
            
            # Set up bus watch for error handling
            bus = self.recording_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_recording_bus_message)
            
            # Start the pipeline
            self.recording_pipeline.set_state(Gst.State.PLAYING)
            self.recording = True
            self.recording_paused = False
            self.frame_count = 0
            self.start_time = time.time()
            self.pause_start_time = 0  # Initialize pause start time
            log.info(f"Started recording to {output_file} with audio")
        except Exception as e:
            log.error(f"Failed to start recording: {e}")
            self.recording = False
            self._cleanup_recording_pipeline()

    def _handle_camera_activity(self, camera: Camera) -> None:
        """Handle camera activity changes and update recording pipeline if needed"""
        if not self.recording or self.recording_paused:
            return
            
        # Get the mixer element
        mixer = self.recording_pipeline.get_by_name('mixer')
        if not mixer:
            return
            
        # Get the camera's audio source
        src_name = f"src{list(self.cameras.values()).index(camera)}"
        src = self.recording_pipeline.get_by_name(src_name)
        if not src:
            return
            
        if camera.active and not camera.connection_lost:
            # Camera is active, ensure its audio is connected
            if not src.get_state(0)[1] == Gst.State.PLAYING:
                src.set_state(Gst.State.PLAYING)
                log.debug(f"Enabled audio for camera {camera.name}")
        else:
            # Camera is inactive, stop its audio
            if src.get_state(0)[1] == Gst.State.PLAYING:
                src.set_state(Gst.State.NULL)
                log.debug(f"Disabled audio for camera {camera.name}")

    def _on_recording_bus_message(self, bus, message) -> bool:
        """Handle GStreamer bus messages for recording pipeline"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error(f"Pipeline error: {err.message}")
            log.debug(f"Debug info: {debug}")
            self.recording = False
            self._cleanup_recording_pipeline()
        elif t == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == self.recording_pipeline:
                log.debug(f"Pipeline state changed: {old_state.value_nick} -> {new_state.value_nick}")
        elif t == Gst.MessageType.EOS:
            log.warning("End of stream")
            self.recording = False
            self._cleanup_recording_pipeline()
        return True

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
            try:
                # Send EOS and wait for it to propagate
                self.recording_pipeline.send_event(Gst.Event.new_eos())
                
                # Set pipeline to NULL state
                self.recording_pipeline.set_state(Gst.State.NULL)
                
                # Clear references
                self.recording_pipeline = None
                self.recording_src = None
                
                # Reset state
                self.recording = False
                self.streaming = False
                self.recording_paused = False
                
                # Run garbage collection
                gc.collect()
            except Exception as e:
                log.error(f"Error cleaning up recording pipeline: {e}")

    def start_streaming(self) -> None:
        """Start streaming to Twitch"""
        if self.streaming:
            log.info("Streaming is already active")
            return
            
        if not self.twitch_stream_key:
            log.error("Cannot start streaming: No Twitch stream key found in secrets.yaml")
            return
            
        # If we're already recording, streaming is handled by the integrated pipeline
        if self.recording:
            log.info("Streaming is already active through the integrated pipeline")
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
            
            # Set up bus watch for error handling
            bus = self.streaming_pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_streaming_bus_message)
            
            # Start the pipeline
            self.streaming_pipeline.set_state(Gst.State.PLAYING)
            self.streaming = True
            log.info("Started streaming to Twitch")
        except Exception as e:
            log.error(f"Failed to start streaming: {e}")
            self.streaming = False
            self._cleanup_streaming_pipeline()

    def _on_streaming_bus_message(self, bus, message) -> bool:
        """Handle GStreamer bus messages for streaming pipeline"""
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            log.error(f"Streaming pipeline error: {err.message}")
            log.debug(f"Debug info: {debug}")
            self.streaming = False
            self._cleanup_streaming_pipeline()
        elif t == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == self.streaming_pipeline:
                log.debug(f"Streaming pipeline state changed: {old_state.value_nick} -> {new_state.value_nick}")
        elif t == Gst.MessageType.EOS:
            log.warning("End of streaming")
            self.streaming = False
            self._cleanup_streaming_pipeline()
        return True

    def _cleanup_streaming_pipeline(self) -> None:
        """Clean up streaming pipeline resources"""
        if self.streaming_pipeline:
            try:
                # Send EOS and wait for it to propagate
                self.streaming_pipeline.send_event(Gst.Event.new_eos())
                
                # Set pipeline to NULL state
                self.streaming_pipeline.set_state(Gst.State.NULL)
                
                # Clear references
                self.streaming_pipeline = None
                self.streaming_src = None
                
                # Reset state
                self.streaming = False
                
                # Run garbage collection
                gc.collect()
            except Exception as e:
                log.error(f"Error cleaning up streaming pipeline: {e}")

    def stop_streaming(self) -> None:
        """Stop streaming to Twitch"""
        if not self.streaming:
            return
            
        log.info("Stopping Twitch stream")
        
        # If we're recording, streaming is handled by the integrated pipeline
        if self.recording:
            log.info("Streaming is part of the integrated pipeline - use stop_recording() instead")
            return
        
        # Properly clean up the pipeline
        self._cleanup_streaming_pipeline()

    def toggle_streaming(self) -> None:
        """Toggle streaming on/off"""
        if self.streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
            
    def push_frame_to_recording(self, frame: np.ndarray) -> None:
        """Push a frame to the integrated pipeline using zero-copy approach"""
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
                framerate = max(1, int(camera.frame_rate))  # Ensure framerate is at least 1
                break
        
        try:
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
                log.warning(f"Error pushing buffer to pipeline: {ret}")
                if ret == Gst.FlowReturn.FLUSHING:
                    # Pipeline is being shut down
                    self.recording = False
                    self.streaming = False
                    self._cleanup_recording_pipeline()
        except Exception as e:
            log.error(f"Error in push_frame_to_recording: {e}")
            self.recording = False
            self.streaming = False
            self._cleanup_recording_pipeline()
    
    def push_frame_to_streaming(self, frame: np.ndarray) -> None:
        """Push a frame to the streaming pipeline using zero-copy approach"""
        if not self.streaming or self.streaming_src is None:
            return
        
        try:
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
                if ret == Gst.FlowReturn.FLUSHING:
                    # Pipeline is being shut down
                    self.streaming = False
                    self._cleanup_streaming_pipeline()
        except Exception as e:
            log.error(f"Error in push_frame_to_streaming: {e}")
            self.streaming = False
            self._cleanup_streaming_pipeline()
    
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
        
        if main_camera.is_vertical:
            # For vertical videos, scale to full height and center horizontally
            scale_factor = 1080 / main_frame.shape[0]
            scaled_width = int(main_frame.shape[1] * scale_factor)
            scaled_height = 1080
            
            # Create a temporary frame for scaling
            temp_frame = self.frame_pool.get_frame((scaled_height, scaled_width, 3))
            cv2.resize(main_frame, (scaled_width, scaled_height), dst=temp_frame)
            
            # Calculate centering offsets
            x_offset = (1920 - scaled_width) // 2
            
            # Copy scaled frame to center of output
            output[:, x_offset:x_offset+scaled_width] = temp_frame
            
            # Return temporary frame to pool
            self.frame_pool.return_frame(temp_frame)
        else:
            # For horizontal videos, scale to full width
            cv2.resize(main_frame, (1920, 1080), dst=scaled_frame)
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
            
            if camera.is_vertical:
                # For vertical videos in PiP, scale to full height and center horizontally
                scale_factor = pip_height / display_frame.shape[0]
                scaled_width = int(display_frame.shape[1] * scale_factor)
                
                # Create a temporary frame for scaling
                temp_frame = self.frame_pool.get_frame((pip_height, scaled_width, 3))
                cv2.resize(display_frame, (scaled_width, pip_height), dst=temp_frame)
                
                # Calculate centering offsets
                x_offset = x + (pip_width - scaled_width) // 2
                
                # Copy scaled frame to center of PiP region
                region = output[y:y+pip_height, x_offset:x_offset+scaled_width]
                cv2.addWeighted(temp_frame, 0.8, region, 0.2, 0, dst=region)
                
                # Return temporary frame to pool
                self.frame_pool.return_frame(temp_frame)
            else:
                # For horizontal videos in PiP, scale to PiP size
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
            try:
                # Skip if no frame available
                if camera.last_frame is None and camera.frozen_frame is None:
                    continue
                    
                # Get frame to display (use frozen frame if connection lost)
                display_frame = camera.frozen_frame if camera.connection_lost else camera.last_frame
                
                # Skip if frame is empty or invalid
                if display_frame is None or display_frame.size == 0:
                    log.warning(f"Empty frame received from camera {name}, skipping")
                    continue
                    
                # Scale frame to 1080p for display
                display_frame_scaled = self.frame_pool.get_frame((1080, 1920, 3))
                try:
                    if camera.is_vertical:
                        # For vertical videos, scale to full height and center horizontally
                        scale_factor = 1080 / display_frame.shape[0]
                        scaled_width = int(display_frame.shape[1] * scale_factor)
                        scaled_height = 1080
                        
                        # Create a temporary frame for scaling
                        temp_frame = self.frame_pool.get_frame((scaled_height, scaled_width, 3))
                        cv2.resize(display_frame, (scaled_width, scaled_height), dst=temp_frame)
                        
                        # Calculate centering offsets
                        x_offset = (1920 - scaled_width) // 2
                        
                        # Copy scaled frame to center of output
                        display_frame_scaled.fill(0)  # Clear the frame
                        display_frame_scaled[:, x_offset:x_offset+scaled_width] = temp_frame
                        
                        # Return temporary frame to pool
                        self.frame_pool.return_frame(temp_frame)
                    else:
                        # For horizontal videos, scale to full width
                        cv2.resize(display_frame, (1920, 1080), dst=display_frame_scaled)
                except cv2.error as e:
                    log.error(f"Error resizing frame from camera {name}: {e}")
                    self.frame_pool.return_frame(display_frame_scaled)
                    continue
                
                # Add camera name overlay
                self._add_camera_overlay(display_frame_scaled, camera)
                
                # Add to frames list
                frames.append((name, display_frame_scaled))
                clean_frames.append((name, display_frame))  # Use original frame for recording
            except Exception as e:
                log.error(f"Error processing frame from camera {name}: {e}")
                # Don't mark as disconnected - let the connection timeout handle it
                continue
        
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
            try:
                y = (i // grid_size) * cell_h
                x = (i % grid_size) * cell_w
                
                # Resize frame to cell size for display
                try:
                    camera = self.cameras[name]
                    if camera.is_vertical:
                        # For vertical videos in grid, scale to full height and center horizontally
                        scale_factor = cell_h / frame.shape[0]
                        scaled_width = int(frame.shape[1] * scale_factor)
                        
                        # Create a temporary frame for scaling
                        temp_frame = self.frame_pool.get_frame((cell_h, scaled_width, 3))
                        cv2.resize(frame, (scaled_width, cell_h), dst=temp_frame)
                        
                        # Calculate centering offsets
                        x_offset = x + (cell_w - scaled_width) // 2
                        
                        # Copy scaled frame to center of cell
                        cell_frame.fill(0)  # Clear the cell
                        cell_frame[:, x_offset-x:x_offset-x+scaled_width] = temp_frame
                        
                        # Return temporary frame to pool
                        self.frame_pool.return_frame(temp_frame)
                    else:
                        # For horizontal videos, scale to cell size
                        cv2.resize(frame, (cell_w, cell_h), dst=cell_frame)
                    
                    output[y:y+cell_h, x:x+cell_w] = cell_frame
                    
                    # For clean output, resize original frame to cell size
                    if camera.is_vertical:
                        # For vertical videos in grid, scale to full height and center horizontally
                        scale_factor = cell_h / clean_frame.shape[0]
                        scaled_width = int(clean_frame.shape[1] * scale_factor)
                        
                        # Create a temporary frame for scaling
                        temp_frame = self.frame_pool.get_frame((cell_h, scaled_width, 3))
                        cv2.resize(clean_frame, (scaled_width, cell_h), dst=temp_frame)
                        
                        # Calculate centering offsets
                        x_offset = x + (cell_w - scaled_width) // 2
                        
                        # Copy scaled frame to center of cell
                        cell_frame.fill(0)  # Clear the cell
                        cell_frame[:, x_offset-x:x_offset-x+scaled_width] = temp_frame
                        
                        # Return temporary frame to pool
                        self.frame_pool.return_frame(temp_frame)
                    else:
                        # For horizontal videos, scale to cell size
                        cv2.resize(clean_frame, (cell_w, cell_h), dst=cell_frame)
                    
                    clean_output[y:y+cell_h, x:x+cell_w] = cell_frame
                except cv2.error as e:
                    log.error(f"Error resizing cell frame for camera {name}: {e}")
                    continue
                
                # Draw thick border if this is the main camera, accounting for the top bar
                if self.cameras[name].main_camera:
                    line_width = 4
                    cv2.rectangle(
                        output,
                        (x + line_width-1, y + line_width-1),
                        (x + cell_w - line_width+1, y + cell_h - line_width+1),
                        (0, 255, 255), line_width
                    )
            except Exception as e:
                log.error(f"Error placing frame in grid for camera {name}: {e}")
                continue
        
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
                status_text = f"ATTEMPT {retry_state['retry_count']+1}/{self.retry_manager.max_retries}   WAITING {time_until_retry}s..."
        
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
            status_x = (frame.shape[1] - status_size[0]) // 2  # Center horizontally
            status_y = (frame.shape[0] - status_size[1]) // 2  # Center vertically
            # Use red for disconnected, yellow for reconnecting
            cv2.putText(frame, "DISCONNECTED", (status_x+1, status_y-(status_size[1]*2)+1), font, font_scale*0.8, (0,0,0), font_thickness+2, line_type)
            cv2.putText(frame, "DISCONNECTED", (status_x, status_y-(status_size[1]*2)), font, font_scale*0.8, (0, 0, 255), font_thickness, line_type)
            cv2.putText(frame, status_text, (status_x+1, status_y+1), font, font_scale*0.8, (0,0,0), font_thickness+2, line_type)
            cv2.putText(frame, status_text, (status_x, status_y), font, font_scale*0.8, (0, 255, 255), font_thickness, line_type)

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
        
        # Calculate animation factor for pulsing glow (0.8 to 1.2 range)
        current_time = time.time()
        pulse_factor = 1.0 + 0.2 * math.sin(current_time * 2)  # 2 Hz pulse
        
        # Create top and bottom toolbars with smooth gradient background
        for y in range(top_bar_height):
            # Smooth gradient using cosine interpolation
            t = y / top_bar_height
            alpha = 0.7 - 0.2 * (1 - math.cos(t * math.pi)) / 2
            color = (int(40 * alpha), int(40 * alpha), int(40 * alpha))
            canvas[y] = color
            
        for y in range(bottom_bar_height):
            # Smooth gradient using cosine interpolation
            t = y / bottom_bar_height
            alpha = 0.7 - 0.2 * (1 - math.cos(t * math.pi)) / 2
            color = (int(40 * alpha), int(40 * alpha), int(40 * alpha))
            canvas[top_bar_height+h+y] = color
        
        # Add tabs for each view mode
        view_modes = [
            (1, "RAW INPUT", ViewMode.INPUT),
            (2, "OUTPUT", ViewMode.OUTPUT)
        ]
        
        for i, (num, name, mode) in enumerate(view_modes):
            x_start = i * tab_width
            x_end = (i + 1) * tab_width
            
            # Create semi-transparent background for tab
            tab_bg = canvas[0:top_bar_height, x_start:x_end].copy()
            if mode == self.view_mode:
                # Active tab: orange gradient with smooth transition
                for x in range(x_end - x_start):
                    t = x / (x_end - x_start)
                    alpha = 0.8 - 0.2 * (1 - math.cos(t * math.pi)) / 2
                    color = (int(0 * alpha), int(120 * alpha), int(255 * alpha))
                    tab_bg[:, x] = color
            else:
                # Inactive tab: dark gray gradient with smooth transition
                for x in range(x_end - x_start):
                    t = x / (x_end - x_start)
                    alpha = 0.6 - 0.2 * (1 - math.cos(t * math.pi)) / 2
                    color = (int(60 * alpha), int(60 * alpha), int(60 * alpha))
                    tab_bg[:, x] = color
            
            # Add subtle border with anti-aliasing
            cv2.rectangle(tab_bg, (0, 0), (x_end-x_start-1, top_bar_height-1), (100, 100, 100), 1, cv2.LINE_AA)
            
            # Blend tab background with smooth transition
            cv2.addWeighted(tab_bg, 0.8, canvas[0:top_bar_height, x_start:x_end], 0.2, 0, 
                          canvas[0:top_bar_height, x_start:x_end])
            
            # Add tab text with number
            tab_text = f"{num}: {name}"
            font_scale = 0.8 * scale_factor
            thickness = max(1, int(2 * scale_factor))
            text_size = cv2.getTextSize(tab_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x_start + (tab_width - text_size[0]) // 2
            text_y = (top_bar_height + text_size[1]) // 2
            
            # Add text shadow for depth with anti-aliasing
            shadow_offset = max(1, int(2 * scale_factor))
            cv2.putText(canvas, tab_text, (text_x+shadow_offset, text_y+shadow_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
            
            # Use white text for active tab, gray for inactive
            text_color = (255, 255, 255) if mode == self.view_mode else (180, 180, 180)
            cv2.putText(canvas, tab_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, thickness, cv2.LINE_AA)
        
        # Add keyboard shortcuts to bottom toolbar with enhanced styling
        shortcuts = [
            "A: Auto-Record",
            "S: Recording",
            "P: Pause",
            "T: Streaming",
            "R: Remux Last File",
            "TAB: Next Camera",
            "L: Lock In",
            "Q: Quit"
        ]
        
        shortcut_width = w // len(shortcuts)
        for i, shortcut in enumerate(shortcuts):
            x_start = i * shortcut_width
            font_scale = 0.7 * scale_factor
            thickness = max(2, int(2 * scale_factor))  # Increased base thickness
            text_size = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
            text_x = x_start + (shortcut_width - text_size[0]) // 2
            text_y = top_bar_height + h + (bottom_bar_height + text_size[1]) // 2
            
            # Add glowing background for recording/streaming shortcuts
            if "Recording" in shortcut and self.recording:
                # Create animated glow effect for recording background
                for radius in range(5, 0, -1):
                    glow_alpha = (0.5 - (radius * 0.1)) * pulse_factor
                    glow_color = (0, 0, int(255 * glow_alpha))
                    # Draw glowing background with increasing size
                    bg_x = text_x - int(10 * scale_factor)
                    bg_y = text_y - text_size[1] - int(5 * scale_factor)
                    bg_w = text_size[0] + int(20 * scale_factor)
                    bg_h = text_size[1] + int(10 * scale_factor)
                    cv2.rectangle(canvas, 
                                (bg_x - radius, bg_y - radius), 
                                (bg_x + bg_w + radius, bg_y + bg_h + radius), 
                                glow_color, -1)
            elif "Streaming" in shortcut and self.streaming:
                # Create animated glow effect for streaming background
                for radius in range(5, 0, -1):
                    glow_alpha = (0.5 - (radius * 0.1)) * pulse_factor
                    glow_color = (0, 0, int(255 * glow_alpha))
                    # Draw glowing background with increasing size
                    bg_x = text_x - int(10 * scale_factor)
                    bg_y = text_y - text_size[1] - int(5 * scale_factor)
                    bg_w = text_size[0] + int(20 * scale_factor)
                    bg_h = text_size[1] + int(10 * scale_factor)
                    cv2.rectangle(canvas, 
                                (bg_x - radius, bg_y - radius), 
                                (bg_x + bg_w + radius, bg_y + bg_h + radius), 
                                glow_color, -1)
            elif "Auto-Record" in shortcut and self.auto_recording:
                # Create animated glow effect for auto-record background
                for radius in range(5, 0, -1):
                    glow_alpha = (0.5 - (radius * 0.1)) * pulse_factor
                    # Use green glow for auto-record
                    glow_color = (0, int(200 * glow_alpha), 0)
                    # Draw glowing background with increasing size
                    bg_x = text_x - int(10 * scale_factor)
                    bg_y = text_y - text_size[1] - int(5 * scale_factor)
                    bg_w = text_size[0] + int(20 * scale_factor)
                    bg_h = text_size[1] + int(10 * scale_factor)
                    cv2.rectangle(canvas, 
                                (bg_x - radius, bg_y - radius), 
                                (bg_x + bg_w + radius, bg_y + bg_h + radius), 
                                glow_color, -1)
            
            # Add text shadow with anti-aliasing
            shadow_offset = max(1, int(scale_factor))
            cv2.putText(canvas, shortcut, (text_x+shadow_offset, text_y+shadow_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA)
            
            # Draw main text with anti-aliasing
            cv2.putText(canvas, shortcut, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
        
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
        
        # Add recording status with enhanced styling
        if self.recording:
            status = "PAUSED" if self.recording_paused else "REC"
            color = (255, 165, 0) if self.recording_paused else (0, 0, 255)  # Orange for paused, red for recording
            indicators.append((status, "", color, True))
        
        # Add streaming status with enhanced styling
        if self.streaming:
            indicators.append(("LIVE", "", (0, 0, 255), True))  # Red for live
            
        # Add auto-recording indicator if enabled
        if self.auto_recording:
            # Determine the auto-recording status
            any_active = self._any_camera_active()
            
            if self.recording and not self.recording_paused and any_active:
                # Auto-recording is actively recording
                auto_rec_text = "AUTO-REC"
                auto_rec_color = (50, 200, 50)  # Green
            elif self.recording and self.recording_paused and not any_active:
                # Auto-recording is paused due to no activity
                auto_rec_text = "AUTO-REC"
                auto_rec_color = (200, 200, 50)  # Yellow
            else:
                # Auto-recording is enabled but not currently recording
                auto_rec_text = "AUTO-REC"
                auto_rec_color = (100, 150, 100)  # Light green
                
            indicators.append((auto_rec_text, "", auto_rec_color, True))
        
        # If we have indicators to show, create a status bar with enhanced styling
        if indicators:
            # Calculate total width needed for the status bar
            total_width = 0
            padding = int(15 * scale_factor)  # Padding between indicators
            
            # First calculate width for camera name
            camera_width = 0
            for label, value, _, has_bg in indicators:
                if label == "CAMERA":
                    display_text = f"{label}: {value}"
                    font_scale = 0.8 * scale_factor
                    thickness = max(1, int(2 * scale_factor))
                    text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                    camera_width = text_size[0] + padding
            
            # Calculate total width needed for other indicators
            other_indicators = [(label, value) for label, value, _, has_bg in indicators if label != "CAMERA"]
            other_width = 0
            indicator_widths = []  # Store widths for each indicator
            
            for label, value in other_indicators:
                display_text = f"{label}" if not value else f"{label}: {value}"
                font_scale = 0.8 * scale_factor
                thickness = max(1, int(2 * scale_factor))
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                # Add extra padding for icon and background
                indicator_width = text_size[0] + int(40 * scale_factor)
                indicator_widths.append(indicator_width)
                other_width += indicator_width + padding
            
            # Create a semi-transparent background for the status bar
            status_bar_height = int(40 * scale_factor)
            status_bar_y = top_bar_height + int(10 * scale_factor)
            
            # Draw camera name indicator on the left if present
            if camera_width > 0:
                camera_x = padding
                # Draw gradient background for camera name
                for x in range(camera_width):
                    t = x / camera_width
                    alpha = 0.7 - 0.2 * (1 - math.cos(t * math.pi)) / 2
                    color = (int(40 * alpha), int(40 * alpha), int(40 * alpha))
                    cv2.rectangle(canvas, 
                                (camera_x + x, status_bar_y), 
                                (camera_x + x + 1, status_bar_y + status_bar_height), 
                                color, -1)
                
                # Draw border around camera name
                cv2.rectangle(canvas, 
                            (camera_x, status_bar_y), 
                            (camera_x + camera_width, status_bar_y + status_bar_height), 
                            (100, 100, 100), 1, cv2.LINE_AA)

                # Draw camera name
                y_pos = status_bar_y + status_bar_height//2 + int(5 * scale_factor)
                for label, value, color, has_bg in indicators:
                    if label == "CAMERA":
                        display_text = f"{label}: {value}"
                        font_scale = 0.8 * scale_factor
                        thickness = max(1, int(2 * scale_factor))
                        
                        # Draw text with shadow
                        shadow_offset = max(1, int(scale_factor))
                        cv2.putText(canvas, display_text, (camera_x+shadow_offset, y_pos+shadow_offset), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                        cv2.putText(canvas, display_text, (camera_x, y_pos), 
                                  cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)
            
            # Draw other indicators right-aligned
            y_pos = status_bar_y + status_bar_height//2 + int(5 * scale_factor)
            x_pos = w - padding  # Start from right edge
            
            # Reverse indicators to draw right-to-left
            other_indicators = [(label, value, color, has_bg) for label, value, color, has_bg in reversed(indicators) if label != "CAMERA"]
            
            for i, (label, value, color, has_bg) in enumerate(other_indicators):
                display_text = f"{label}" if not value else f"{label}: {value}"
                font_scale = 0.8 * scale_factor
                thickness = max(1, int(2 * scale_factor))
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Calculate position for this indicator
                indicator_width = indicator_widths[-(i+1)]
                x_pos -= indicator_width
                
                # Draw indicator background
                if has_bg:
                    bg_padding = int(10 * scale_factor)
                    cv2.rectangle(canvas,
                                (x_pos - bg_padding, status_bar_y),
                                (x_pos + indicator_width + bg_padding, status_bar_y + status_bar_height),
                                (40, 40, 40), -1)
                    
                    # Draw icon based on indicator type
                    icon_x = x_pos + int(10 * scale_factor)
                    icon_y = y_pos - int(text_size[1] * 0.5)
                    
                    if label in ["REC", "LIVE"]:
                        circle_radius = int(text_size[1] * 0.5)
                        cv2.circle(canvas, (icon_x, icon_y), circle_radius, (0, 0, 255), -1)
                    elif label == "PAUSED":
                        bar_width = int(text_size[1] * 0.3)
                        bar_height = int(text_size[1] * 0.8)
                        gap = int(text_size[1] * 0.15)
                        cv2.rectangle(canvas, 
                                    (icon_x - bar_width - gap, icon_y - bar_height//2),
                                    (icon_x - gap, icon_y + bar_height//2),
                                    (255, 165, 0), -1)
                        cv2.rectangle(canvas,
                                    (icon_x + gap, icon_y - bar_height//2),
                                    (icon_x + bar_width + gap, icon_y + bar_height//2),
                                    (255, 165, 0), -1)
                    elif label == "AUTO-REC":
                        radius = int(text_size[1] * 0.4)
                        cv2.circle(canvas, (icon_x, icon_y), radius, color, 2)
                        arrow_length = radius // 2
                        arrow_angle = math.pi / 4
                        start_x = icon_x + int(radius * math.cos(arrow_angle))
                        start_y = icon_y + int(radius * math.sin(arrow_angle))
                        end_x = icon_x + int((radius + arrow_length) * math.cos(arrow_angle))
                        end_y = icon_y + int((radius + arrow_length) * math.sin(arrow_angle))
                        cv2.arrowedLine(canvas, (start_x, start_y), (end_x, end_y), color, 2, tipLength=0.3)
                
                # Draw text with shadow
                text_x = x_pos + int(30 * scale_factor)  # Add space for icon
                shadow_offset = max(1, int(scale_factor))
                cv2.putText(canvas, display_text, (text_x+shadow_offset, y_pos+shadow_offset),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness, cv2.LINE_AA)
                cv2.putText(canvas, display_text, (text_x, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
                
                # Add padding between indicators
                x_pos -= padding
        
        return canvas

    def _prevent_sleep(self) -> None:
        """Prevent macOS from sleeping while the app is running"""
        try:
            # Start caffeinate in a new process
            # -i prevents idle sleep
            # -d prevents display sleep
            # -m prevents disk sleep
            self._caffeinate_process = subprocess.Popen(['caffeinate', '-i', '-d', '-m'])
            log.info("Sleep prevention enabled")
        except Exception as e:
            log.error(f"Failed to prevent sleep: {e}")

    def _restore_sleep(self) -> None:
        """Restore normal sleep behavior"""
        if self._caffeinate_process:
            try:
                self._caffeinate_process.terminate()
                self._caffeinate_process.wait(timeout=5)  # Wait up to 5 seconds for process to terminate
                log.info("Sleep prevention disabled")
            except subprocess.TimeoutExpired:
                self._caffeinate_process.kill()  # Force kill if it doesn't terminate
                log.warning("Forced termination of sleep prevention")
            except Exception as e:
                log.error(f"Error restoring sleep: {e}")
            finally:
                self._caffeinate_process = None

    def _handle_auto_recording(self):
        """Handle auto-recording logic"""
        if not self.auto_recording:
            return
            
        # Check if any camera is active
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
        
        # Update audio sources based on camera activity
        if self.recording and not self.recording_paused:
            for camera in self.cameras.values():
                self._handle_camera_activity(camera)

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
            # Turn off auto-record when manually controlling recording
            if self.auto_recording:
                self.auto_recording = False
                log.info("Auto-recording disabled due to manual recording control")
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
        elif key == ord('p'):  # 'p' to pause/resume recording
            # Turn off auto-record when manually controlling recording
            if self.auto_recording:
                self.auto_recording = False
                log.info("Auto-recording disabled due to manual recording control")
            if self.recording:
                if self.recording_paused:
                    self.resume_recording()
                else:
                    self.pause_recording()
        elif key == ord('l'):  # 'l' to lock/unlock current camera
            self._toggle_lock_current_camera()
        elif key == ord('\t'):  # TAB to cycle main camera
            self._cycle_main_camera()
        elif key == ord('r'):  # 'r' to remux last recording
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
        
        # Reset the cooldown timer for auto-switching
        self.last_tab_time = time.time()
        
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
            
            # Create GStreamer pipeline for remuxing with both video and audio
            pipeline_str = (
                f'filesrc location={last_mkv} ! '
                'matroskademux name=demux '
                'mp4mux name=mp4mux ! '
                f'filesink location={output_mp4} '
                'demux.video_0 ! h264parse ! queue ! mp4mux.video_0 '
                'demux.audio_0 ! aacparse ! queue ! mp4mux.audio_0'
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
                    log.debug(f"Debug info: {debug}")
                    if os.path.exists(output_mp4):
                        os.remove(output_mp4)  # Clean up failed output
                else:
                    log.info(f"Successfully remuxed to {output_mp4}")
            
        except Exception as e:
            log.error(f"Error during remuxing: {e}")
            if os.path.exists(output_mp4):
                os.remove(output_mp4)  # Clean up failed output

if __name__ == "__main__":
    # Quick test with debug viewer
    stream = WorkshopStream(debug=True)
    stream.add_camera("rtsp://192.168.1.155:8554/live", "iphone 13")
    # stream.add_camera("rtsp://192.168.1.156:8554/live", "iphone xs")
    # stream.add_camera("rtsp://192.168.1.114:8080/h264_pcm.sdp", "galaxy s21")
    
    print("Controls:")
    print("  1 - input view (all cameras)")
    print("  2 - output view (for streaming/recording)")
    print("  TAB - next camera")
    print("  l - lock in current camera")
    print("  s - start/stop recording")
    print("  p - pause/resume recording")
    print("  a - auto-recording")
    print("  t - Twitch streaming")
    print("  r - remux last recording")
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