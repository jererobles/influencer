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

@dataclass
class Camera:
    url: str
    name: str
    resolution: Tuple[int, int] = (1920, 1080)
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
    retry_count: int = 0         # count of reconnection attempts
    max_retries: int = 10        # maximum number of reconnection attempts
    retry_interval: float = 5.0  # seconds between retry attempts
    frame_rate: float = 30.0     # calculated frame rate
    last_frame: Optional[np.ndarray] = None
    
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
            
            # Fill the area with solid green
            cv2.fillPoly(frame, [np.array(polygon_points, dtype=np.int32)], (0, 100, 0))
            
            # Draw the top line with anti-aliasing
            for i in range(len(points) - 1):
                cv2.line(frame, points[i], points[i+1], (0, 255, 0), 1, cv2.LINE_AA)
    
    def cleanup(self):
        """Clean up resources when camera is no longer needed"""
        self.previous_frame = None
        self._last_frame = None
        self._frame_times.clear()  # Clear frame time history
        self._bandwidth_samples.clear()  # Clear bandwidth history
        
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
            self.sink = None
            self.pipeline = None

class ViewMode(Enum):
    GRID = auto()      # show all cameras in grid
    ACTIVE = auto()    # show only active cameras
    OUTPUT = auto()    # show final composite output
    MOTION = auto()    # show motion detection debug view
    PIP = auto()       # picture-in-picture mode

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
        self.view_mode = ViewMode.PIP
        
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
        
        # Initialize mediapipe detector
        BaseOptions = mp.tasks.BaseOptions
        Detector = mp.tasks.vision.ObjectDetector
        DetectorOptions = mp.tasks.vision.ObjectDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create detector for person detection
        options = DetectorOptions(
            base_options=BaseOptions(model_asset_path='efficientdet_lite0.tflite'),
            running_mode=VisionRunningMode.IMAGE,
            score_threshold=0.29,
            category_allowlist=['person'])
        self.detector = Detector.create_from_options(options)
        
    def add_camera(self, url: str, name: str) -> None:
        """Add a new camera to the stream"""
        camera = Camera(url=url, name=name)
        self.cameras[name] = camera
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

        log.info(f"Creating pipeline for {camera.name}: {pipeline_str}")
        
        # Create and store the pipeline in the camera object
        camera.pipeline = Gst.parse_launch(pipeline_str)
        camera.sink = camera.pipeline.get_by_name('sink')
        
    async def _capture_frames(self, camera: Camera) -> None:
        """Capture frames from a camera and detect people"""
        detection_counter = 0
        
        while self.running:
            try:
                # Create GStreamer pipeline for this camera
                self._create_camera_pipeline(camera)

                # Create a reference to self that won't prevent garbage collection
                stream_ref = weakref.ref(self)

                # Setup frame callback
                def on_new_sample(appsink):
                    try:
                        # Get the stream reference
                        stream = stream_ref()
                        if not stream:
                            return Gst.FlowReturn.ERROR

                        current_time = time.time()
                        camera.last_frame_time = current_time
                        camera.connection_lost = False  # We got a frame, so connection is not lost
                        camera.retry_count = 0  # Reset retry count on successful frame
                        
                        # Update frame rate calculation
                        camera.update_frame_rate(current_time)

                        # Get the sample from appsink
                        sample = appsink.emit("pull-sample")
                        if not sample:
                            return Gst.FlowReturn.ERROR

                        # Get the buffer from the sample
                        buffer = sample.get_buffer()
                        if not buffer:
                            return Gst.FlowReturn.ERROR

                        # Update bandwidth calculation
                        frame_size = buffer.get_size()
                        camera.update_bandwidth(frame_size, current_time)

                        # Map the buffer for reading
                        success, map_info = buffer.map(Gst.MapFlags.READ)
                        if not success:
                            return Gst.FlowReturn.ERROR

                        try:
                            # Get a reusable frame from the pool
                            old_frame = camera.last_frame
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
                                
                                # Clean up temporary objects
                                del small_frame
                                del small_frame_rgb
                                del mp_image

                        finally:
                            # Always unmap the buffer
                            buffer.unmap(map_info)

                    except Exception as e:
                        log.error(f"Error in frame callback: {e}")
                        return Gst.FlowReturn.ERROR

                    return Gst.FlowReturn.OK

                # Connect callback and start pipeline
                camera.sink.connect('new-sample', on_new_sample)
                camera.pipeline.set_state(Gst.State.PLAYING)

                log.info(f"Pipeline started for {camera.name}")

                # Main loop to keep pipeline running and check for connection loss
                while self.running:
                    current_time = time.time()
                    
                    # Check for connection loss (no frames for 3 seconds)
                    if not camera.connection_lost and current_time - camera.last_frame_time > 3:
                        camera.connection_lost = True
                        log.warning(f"Connection lost to camera {camera.name}")
                        
                        # Clean up the existing pipeline
                        self._cleanup_camera(camera)
                        
                        # Check retry limits
                        if camera.retry_count >= camera.max_retries:
                            log.error(f"Max retries ({camera.max_retries}) reached for camera {camera.name}")
                            break
                        
                        # Increment retry count and wait before next attempt
                        camera.retry_count += 1
                        log.info(f"Attempting reconnection to {camera.name} (attempt {camera.retry_count}/{camera.max_retries})")
                        await asyncio.sleep(camera.retry_interval)
                        break  # Break inner loop to recreate pipeline
                    
                    await asyncio.sleep(0.5)  # Check connection every 0.5 seconds

            except Exception as e:
                log.error(f"Error in camera pipeline {camera.name}: {e}")
                self._cleanup_camera(camera)
                
                # Handle retries
                if camera.retry_count < camera.max_retries:
                    camera.retry_count += 1
                    log.info(f"Attempting reconnection to {camera.name} (attempt {camera.retry_count}/{camera.max_retries})")
                    await asyncio.sleep(camera.retry_interval)
                else:
                    log.error(f"Max retries ({camera.max_retries}) reached for camera {camera.name}")
                    break

        # Final cleanup
        self._cleanup_camera(camera)
    
    def _cleanup_camera(self, camera: Camera) -> None:
        """Clean up camera resources"""
        if camera.pipeline:
            camera.pipeline.set_state(Gst.State.NULL)
        camera.pipeline = None
        camera.sink = None
        
        # Clear frame references
        if camera.previous_frame is not None:
            self.frame_pool.return_frame(camera.previous_frame)
            camera.previous_frame = None
            
        if camera.last_frame is not None:
            self.frame_pool.return_frame(camera.last_frame)
            camera.last_frame = None
            
    def _select_main_camera(self) -> str:
        """Select the best camera to show as main view"""
        # First check for manually selected camera
        manual_main = next((name for name, camera in self.cameras.items() 
                           if camera.manual_main), None)
        if manual_main:
            return manual_main

        best_camera = None
        best_score = -1

        for name, camera in self.cameras.items():
            if not camera.active:
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
        
        # Create GStreamer pipeline for recording
        pipeline_str = (
            'appsrc name=src is-live=true format=time do-timestamp=true ! '
            'video/x-raw,format=BGR,width=1920,height=1080,framerate=60/1 ! '
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
        
        # Get frame size
        required_size = frame.nbytes
        
        # Create GStreamer buffer directly from frame data
        gst_buffer = Gst.Buffer.new_wrapped(frame.tobytes())
        
        # Set buffer timestamp
        duration = 1 / 60 * Gst.SECOND  # Using 60 fps
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
        # For PIP mode
        if self.view_mode == ViewMode.PIP:
            return await self._create_pip_view()
        
        # For OUTPUT mode
        if self.view_mode == ViewMode.OUTPUT and self.output_frame is not None:
            # Get a frame from the pool for the output with UI elements
            view = self.frame_pool.get_frame(self.output_frame.shape)
            np.copyto(view, self.output_frame)
            
            # Store reference to output frame for recording (no copy needed)
            self.clean_frame_for_recording = self.output_frame
            
            # Add UI elements to the display view only
            view = self._add_ui_elements(view)
            return view
        
        # For other view modes (GRID, ACTIVE, MOTION)
        return await self._create_grid_view()
        
    async def _create_pip_view(self) -> np.ndarray:
        """Create picture-in-picture view with memory optimizations"""
        # Prepare output frames - get from pool
        output = self.frame_pool.get_frame((1080, 1920, 3))
        clean_output = self.frame_pool.get_frame((1080, 1920, 3))
        
        # Initialize to black
        output.fill(0)
        clean_output.fill(0)
        
        # Select main camera
        main_camera_name = self._select_main_camera()
        
        # If no main camera, return blank screen with UI
        if main_camera_name is None:
            self.clean_frame_for_recording = clean_output
            output = self._add_ui_elements(output)
            return output
        
        # Update main_camera flags
        for name, camera in self.cameras.items():
            camera.main_camera = (name == main_camera_name)
        
        # Get main camera frame
        main_camera = self.cameras[main_camera_name]
        main_frame = main_camera.last_frame
        
        if main_frame is None:
            self.clean_frame_for_recording = clean_output
            output = self._add_ui_elements(output)
            return output
        
        # Resize main camera frame directly to output
        cv2.resize(main_frame, (1920, 1080), dst=output)
        cv2.resize(main_frame, (1920, 1080), dst=clean_output)
        
        # Add smaller overlays for active cameras (excluding main)
        pip_width = 480  # 1/4 of screen width
        pip_height = 270  # Keep 16:9 aspect ratio
        padding = 20  # Space between PiP windows
        
        # Get active cameras excluding main
        active_cameras = [(name, camera) for name, camera in self.cameras.items() 
                         if name != main_camera_name and camera.active and camera.last_frame is not None]
        
        # Prepare PiP frame from pool once
        pip_frame = self.frame_pool.get_frame((pip_height, pip_width, 3))
        
        for i, (name, camera) in enumerate(active_cameras):
            # Calculate position for PiP
            x = 1920 - pip_width - padding
            y = padding + i * (pip_height + padding)
            
            # Skip if we run out of vertical space
            if y + pip_height > 1080:
                break
            
            # Resize camera frame to PiP size
            cv2.resize(camera.last_frame, (pip_width, pip_height), dst=pip_frame)
            
            # Insert PiP into output frames with overlay effect
            region = output[y:y+pip_height, x:x+pip_width]
            cv2.addWeighted(pip_frame, 0.8, region, 0.2, 0, dst=region)
            
            clean_region = clean_output[y:y+pip_height, x:x+pip_width]
            cv2.addWeighted(pip_frame, 0.8, clean_region, 0.2, 0, dst=clean_region)
        
        # Return PiP frame to pool
        self.frame_pool.return_frame(pip_frame)
        
        # Store clean output for recording
        self.clean_frame_for_recording = clean_output
        
        # Add UI elements to display view
        output = self._add_ui_elements(output)
        return output
        
    async def _create_grid_view(self) -> np.ndarray:
        """Create grid view with memory optimizations"""
        # Collect frames based on view mode
        frames = []
        clean_frames = []
        
        for name, camera in self.cameras.items():
            # Skip if no frame or inactive cameras in ACTIVE mode
            if camera.last_frame is None:
                continue
                
            if self.view_mode == ViewMode.ACTIVE and not camera.active:
                continue
            
            # For motion debug view
            if self.view_mode == ViewMode.MOTION and camera.previous_frame is not None:
                # Create motion visualization
                motion_frame = self.frame_pool.get_frame(camera.motion_res)
                gray_frame = self.frame_pool.get_frame(camera.motion_res, dtype=np.uint8)
                
                # Resize and convert to grayscale
                cv2.resize(camera.last_frame, camera.motion_res, dst=motion_frame)
                cv2.cvtColor(motion_frame, cv2.COLOR_BGR2GRAY, dst=gray_frame)
                
                # Calculate difference
                diff = cv2.absdiff(camera.previous_frame, gray_frame)
                
                # Create color mapped version
                diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
                
                # Resize to full resolution
                display_frame = self.frame_pool.get_frame(camera.resolution)
                cv2.resize(diff_color, camera.resolution, dst=display_frame)
                
                # Add to frames list
                frames.append((name, display_frame))
                clean_frames.append((name, display_frame))  # Same for clean frames in motion view
                
                # Return temporary frames to pool
                self.frame_pool.return_frame(motion_frame)
                self.frame_pool.return_frame(gray_frame)
                self.frame_pool.return_frame(diff_color)
            else:
                # For normal view, get frame from camera
                display_frame = self.frame_pool.get_frame(camera.last_frame.shape)
                
                # Copy camera frame to avoid modifying original
                np.copyto(display_frame, camera.last_frame)
                
                # Add camera name overlay
                self._add_camera_overlay(display_frame, camera)
                
                # Add to frames list
                frames.append((name, display_frame))
                clean_frames.append((name, camera.last_frame))  # Original frame for recording
        
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
            
            # Resize frame to cell size
            cv2.resize(frame, (cell_w, cell_h), dst=cell_frame)
            output[y:y+cell_h, x:x+cell_w] = cell_frame
            
            # Resize clean frame
            cv2.resize(clean_frame, (cell_w, cell_h), dst=cell_frame)
            clean_output[y:y+cell_h, x:x+cell_w] = cell_frame
        
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
        """Add camera name overlay to frame"""
        # Build the camera name text with status indicators
        text = camera.name
        if camera.active:
            text += " [active]"
            
        # Add FPS indicator
        text += f" ({camera.frame_rate:.1f} fps)"
            
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        
        # Draw a semi-transparent background rectangle
        h, w = frame.shape[:2]
        bg_x = 10
        bg_y = h - 10 - text_size[1] - 10  # 10px padding
        bg_w = text_size[0] + 20  # 10px padding on each side
        bg_h = text_size[1] + 10  # 5px padding on top and bottom
        
        # Draw semi-transparent background
        cv2.rectangle(frame, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
        
        # Add camera name text
        cv2.putText(frame, text, 
                  (bg_x + 10, bg_y + bg_h - 10), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Add bandwidth graph
        graph_x = bg_x + bg_w + 10  # Place graph to the right of the text
        graph_y = bg_y
        graph_width = 120  # Width of the graph
        graph_height = 40  # Height of the graph
        camera.draw_bandwidth_graph(frame, graph_x, graph_y, graph_width, graph_height)
    
    def _add_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Add UI elements with memory-efficient approach"""
        # Get original frame dimensions
        h, w = frame.shape[:2]
        
        # Constants for UI elements
        top_bar_height = 60
        bottom_bar_height = 50
        tab_width = w // 5  # 5 view modes
        
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
            (1, "RAW", ViewMode.GRID),
            (2, "ACTIVE", ViewMode.ACTIVE),
            (3, "PROCESSED", ViewMode.OUTPUT),
            (4, "MOTION", ViewMode.MOTION),
            (5, "PIP", ViewMode.PIP)
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
            text_size = cv2.getTextSize(tab_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
            text_x = x_start + (tab_width - text_size[0]) // 2
            text_y = (top_bar_height + text_size[1]) // 2
            
            # Use white text for active tab, gray for inactive
            text_color = (255, 255, 255) if mode == self.view_mode else (180, 180, 180)
            cv2.putText(canvas, tab_text, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, text_color, 2)
        
        # Add keyboard shortcuts to bottom toolbar
        shortcuts = [
            "1-5: Switch Views",
            "A: Auto-Recording",
            "S: Start/Stop Recording",
            "T: Toggle Streaming",
            "TAB: Cycle Cameras/Auto (PIP)",
            "Q: Quit"
        ]
        
        shortcut_width = w // len(shortcuts)
        for i, shortcut in enumerate(shortcuts):
            x_start = i * shortcut_width
            text_size = cv2.getTextSize(shortcut, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            text_x = x_start + (shortcut_width - text_size[0]) // 2
            text_y = top_bar_height + h + (bottom_bar_height + text_size[1]) // 2
            cv2.putText(canvas, shortcut, (text_x, text_y), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
        
        # Create status indicators if needed
        indicators = []
        
        # Add main camera name if in PIP mode
        if self.view_mode == ViewMode.PIP:
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
            padding = 15  # Padding between indicators
            
            for label, value, _, has_bg in indicators:
                display_text = f"{label}" if not value else f"{label}: {value}"
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                # Width for text + spacing + background if needed
                indicator_width = text_size[0] + padding
                if has_bg:
                    indicator_width += 20  # Extra space for background
                total_width += indicator_width
            
            # Create a semi-transparent background for the status bar
            status_bar_height = 40
            status_bar_y = top_bar_height + 10
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
            y_pos = status_bar_y + status_bar_height//2 + 5  # Adjust for text baseline
            
            # Draw indicators from left to right
            for label, value, color, has_bg in indicators:
                display_text = f"{label}" if not value else f"{label}: {value}"
                text_size = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
                
                # For status indicators (REC, LIVE, PAUSED), add a colored background
                if has_bg:
                    # Calculate background rectangle dimensions
                    bg_padding = 10
                    bg_x = x_pos - bg_padding
                    bg_y = status_bar_y + 5
                    bg_w = text_size[0] + bg_padding * 2
                    bg_h = status_bar_height - 10
                    
                    # Draw colored background
                    cv2.rectangle(canvas, 
                                (bg_x, bg_y), 
                                (bg_x + bg_w, bg_y + bg_h), 
                                color, -1)
                    
                    # Draw text in white on colored background
                    cv2.putText(canvas, display_text, (x_pos, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    # Move right for next indicator with padding
                    x_pos += bg_w + padding
                else:
                    # Draw regular text for camera name
                    cv2.putText(canvas, display_text, (x_pos, y_pos), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
                    
                    # Move right for next indicator with padding
                    x_pos += text_size[0] + padding
        
        return canvas

    async def start(self) -> None:
        """Start the stream processing with explicit memory management"""
        log.info("Starting workshop stream")
        self.running = True
        
        # Create tasks for all cameras
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
        # Push frame to recording if active
        if self.recording:
            # Use the clean_frame_for_recording which has no text overlays
            if self.clean_frame_for_recording is not None:
                self.push_frame_to_recording(self.clean_frame_for_recording)
            else:
                # Fallback to view
                self.push_frame_to_recording(view)
        
        # Push frame to streaming if active
        if self.streaming:
            # Use the clean_frame_for_recording which has no text overlays
            if self.clean_frame_for_recording is not None:
                self.push_frame_to_streaming(self.clean_frame_for_recording)
            else:
                # Fallback to view
                self.push_frame_to_streaming(view)
    
    def _handle_keyboard_input(self, key):
        """Handle keyboard input for the debug viewer"""
        if key == ord('q'):
            raise KeyboardInterrupt
        
        # Number keys for view modes (1-5)
        elif key == ord('1'):
            self.view_mode = ViewMode.GRID
        elif key == ord('2'):
            self.view_mode = ViewMode.ACTIVE
        elif key == ord('3'):
            self.view_mode = ViewMode.OUTPUT
        elif key == ord('4'):
            self.view_mode = ViewMode.MOTION
        elif key == ord('5'):
            self.view_mode = ViewMode.PIP
        
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
        elif key == ord('\t') and self.view_mode == ViewMode.PIP:
            self._cycle_main_camera()
    
    def _cycle_main_camera(self):
        """Cycle through cameras for PiP mode"""
        log.info("TAB pressed. Cycling camera selection.")
        
        # Get list of active cameras
        active_cameras = [name for name, cam in self.cameras.items() if cam.active]
        
        if not active_cameras:
            return
            
        # Check if we're currently in auto mode (no manual selection)
        any_manual = any(cam.manual_main for cam in self.cameras.values())
        
        # If in auto mode, select the first camera
        if not any_manual:
            # Select the first active camera
            first_camera = self.cameras[active_cameras[0]]
            first_camera.manual_main = True
            log.info(f"Switching from auto mode to manual camera: {active_cameras[0]}")
        else:
            # Find the current manually selected camera
            current_manual = next((name for name, cam in self.cameras.items() 
                                if cam.manual_main), None)
            
            if current_manual in active_cameras:
                # Find the next camera in the cycle
                current_idx = active_cameras.index(current_manual)
                next_idx = (current_idx + 1) % (len(active_cameras) + 1)  # +1 for auto mode
                
                # Reset all manual selections
                for cam in self.cameras.values():
                    cam.manual_main = False
                
                # If next_idx is within active_cameras, select that camera
                # Otherwise, it means we've cycled back to auto mode
                if next_idx < len(active_cameras):
                    next_camera = self.cameras[active_cameras[next_idx]]
                    next_camera.manual_main = True
                    log.info(f"Manually selected camera: {active_cameras[next_idx]}")
                else:
                    # Auto mode - no manual selection
                    log.info("Switched to auto camera selection mode")
            else:
                # Current manual camera is not active, reset to auto mode
                for cam in self.cameras.values():
                    cam.manual_main = False
                log.info("Reset to auto camera selection mode")
            
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
    
    print("Debug controls:")
    print("  1 - grid view (all cameras)")
    print("  2 - active cameras only")
    print("  3 - composite output")
    print("  4 - motion detection debug")
    print("  5 - picture-in-picture mode")
    print("  TAB - cycle main camera (in PiP mode)")
    print("  r - release manual control (in PiP mode)")
    print("  s - start/stop recording")
    print("  a - toggle auto-recording")
    print("  t - toggle streaming to Twitch")
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