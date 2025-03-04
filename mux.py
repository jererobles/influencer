import asyncio
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum, auto
import logging
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
from threading import Thread
import sys
import time
import datetime
import yaml  # Add import for yaml to load secrets

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

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
    last_frame: Optional[np.ndarray] = None      # for display
    active: bool = False
    last_active_time: float = 0  # timestamp when camera last became active
    face_count: int = 0          # number of faces currently detected
    motion_score: float = 0      # amount of motion (0-1)
    main_camera: bool = False    # is this the main camera in PiP mode?
    manual_main: bool = False    # was this camera manually selected as main

class ViewMode(Enum):
    GRID = auto()      # show all cameras in grid
    ACTIVE = auto()    # show only active cameras
    OUTPUT = auto()    # show final composite output
    MOTION = auto()    # show motion detection debug view
    PIP = auto()       # picture-in-picture mode

class WorkshopStream:
    def __init__(self, debug: bool = False):
        self.cameras: Dict[str, Camera] = {}
        self.frame_buffer: Dict[str, asyncio.Queue] = {}
        self.output_frame: Optional[np.ndarray] = None
        self.clean_frame_for_recording: Optional[np.ndarray] = None  # Clean frame without overlays for recording
        self.running = False
        self.debug = debug
        self.view_mode = ViewMode.PIP
        
        # Recording related attributes
        self.recording = False
        self.recording_pipeline = None
        self.recording_src = None
        self.recording_paused = False  # New flag to track if recording is paused
        self.auto_recording = False    # Disable auto-recording by default
        
        # Streaming related attributes
        self.streaming = False
        self.streaming_pipeline = None
        self.streaming_src = None
        self.twitch_stream_key = self._load_twitch_stream_key()
        
        # initialize mediapipe
        BaseOptions = mp.tasks.BaseOptions
        FaceDetector = mp.tasks.vision.FaceDetector
        FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
        VisionRunningMode = mp.tasks.vision.RunningMode

        # Create a face detector instance with the image mode:
        options = FaceDetectorOptions(
            base_options=BaseOptions(model_asset_path='blaze_face_short_range.tflite'),
            running_mode=VisionRunningMode.IMAGE)
        self.detector = FaceDetector.create_from_options(options)
        
    def add_camera(self, url: str, name: str) -> None:
        """Add a new camera to the stream"""
        self.cameras[name] = Camera(url=url, name=name)
        self.frame_buffer[name] = asyncio.Queue(maxsize=1)
        log.info(f"added camera: {name} @ {url}")
        
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
        
    async def _capture_frames(self, camera: Camera) -> None:
        """Capture frames from a camera and detect people"""
        
        detection_counter = 0  # Add counter at start of method
        
        # Create GStreamer pipeline
        if camera.url.startswith('http://'):
            pipeline_str = (
                f'souphttpsrc location={camera.url} ! '
                'decodebin ! videoconvert ! '
                'video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=1 drop=True'
            )
        else:  # RTSP
            pipeline_str = (
                f'rtspsrc location={camera.url} latency=0 ! '
                'rtph264depay ! h264parse ! avdec_h264 ! '
                'videoconvert ! video/x-raw,format=BGR ! '
                'appsink name=sink emit-signals=True max-buffers=1 drop=True'
            )
        
        log.info(f"Creating pipeline for {camera.name}: {pipeline_str}")
        
        pipeline = Gst.parse_launch(pipeline_str)
        sink = pipeline.get_by_name('sink')
        
        # Setup frame callback
        def on_new_sample(appsink):
            try:
                sample = appsink.emit('pull-sample')
                if sample:
                    buf = sample.get_buffer()
                    caps = sample.get_caps()
                    width = caps.get_structure(0).get_value('width')
                    height = caps.get_structure(0).get_value('height')
                    
                    # Create numpy array from buffer data
                    success, map_info = buf.map(Gst.MapFlags.READ)
                    if success:
                        # Create numpy array from the data
                        frame = np.ndarray(
                            shape=(height, width, 3),
                            dtype=np.uint8,
                            buffer=map_info.data
                        ).copy()  # Make a copy to ensure we own the memory
                        buf.unmap(map_info)
                        
                        # Store the frame
                        camera.last_frame = frame
                        
                        nonlocal detection_counter  # Access outer counter
                        detection_counter += 1  # Increment counter
                        
                        # Run detection if needed
                        if detection_counter % 10 == 0:  # Every 10th frame
                            small_frame = cv2.resize(frame, camera.detection_res)
                            small_frame_rgb = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
                            mp_image = mp.Image(
                                image_format=mp.ImageFormat.SRGB,
                                data=small_frame_rgb
                            )
                            results = self.detector.detect(mp_image)
                            
                            # Update activity state and metrics
                            camera.face_count = len(results.detections)
                            if camera.face_count > 0:
                                if not camera.active:  # Only log when state changes
                                    log.info(f"Camera {camera.name} became active")
                                camera.active = True
                                camera.cooldown = 30
                                camera.last_active_time = time.time()
                            elif camera.cooldown > 0:
                                camera.cooldown -= 1
                            else:
                                if camera.active:  # Only log when state changes
                                    log.info(f"Camera {camera.name} became inactive")
                                camera.active = False
                                camera.face_count = 0
                            
                            # Calculate motion score
                            if camera.previous_frame is not None:
                                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                                diff = cv2.absdiff(camera.previous_frame, gray)
                                camera.motion_score = np.mean(diff) / 255.0  # Normalize to 0-1
                                camera.previous_frame = gray
                        
            except Exception as e:
                log.error(f"Error in frame callback: {e}")
            return Gst.FlowReturn.OK
        
        # Connect callback and start pipeline
        sink.connect('new-sample', on_new_sample)
        pipeline.set_state(Gst.State.PLAYING)
        
        log.info(f"Pipeline started for {camera.name}")
        
        # Main loop to keep pipeline running
        try:
            while self.running:
                await asyncio.sleep(0.001)  # Minimal sleep to allow other tasks
        except Exception as e:
            log.error(f"Error in pipeline loop: {e}")
        finally:
            # Cleanup
            pipeline.set_state(Gst.State.NULL)
            log.info(f"Pipeline stopped for {camera.name}")
        
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
        output_file = f"recording_{timestamp}.mkv"
        
        # Create GStreamer pipeline for recording
        # Use a different approach with appsrc -> videoconvert -> encoder -> muxer -> filesink
        pipeline_str = (
            'appsrc name=src is-live=true format=time do-timestamp=true ! '
            'video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'x264enc speed-preset=superfast tune=zerolatency bitrate=8000 key-int-max=30 ! '
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
            if self.recording_pipeline:
                self.recording_pipeline.set_state(Gst.State.NULL)
                self.recording_pipeline = None

    def pause_recording(self) -> None:
        """Pause the current recording without stopping the pipeline"""
        if not self.recording or self.recording_paused:
            return
            
        log.info("Pausing recording")
        self.recording_paused = True
        
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
        if self.recording:
            log.info("Stopping recording")
            
            # Calculate recording duration
            duration = time.time() - self.start_time
            if self.recording_paused:
                # Subtract the time spent in pause
                pause_duration = time.time() - self.pause_start_time
                duration -= pause_duration
            
            log.info(f"Recording stopped. Duration: {duration:.2f} seconds, Frames: {self.frame_count}")
            
            # Stop the pipeline
            if self.recording_pipeline:
                self.recording_pipeline.send_event(Gst.Event.new_eos())
                self.recording_pipeline.set_state(Gst.State.NULL)
                self.recording_pipeline = None
                self.recording_src = None
            
            self.recording = False
            self.recording_paused = False
    
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
            'video/x-raw,format=BGR,width=1920,height=1080,framerate=30/1 ! '
            'videoconvert ! video/x-raw,format=I420 ! '
            'x264enc speed-preset=veryfast tune=zerolatency bitrate=2500 key-int-max=60 ! '
            'h264parse ! flvmux streamable=true ! '
            f'rtmpsink location=rtmp://live.twitch.tv/app/{self.twitch_stream_key} sync=false'
        )
        
        log.info(f"Creating streaming pipeline")
        
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
            if self.streaming_pipeline:
                self.streaming_pipeline.set_state(Gst.State.NULL)
                self.streaming_pipeline = None
                self.streaming_src = None
    
    def stop_streaming(self) -> None:
        """Stop streaming to Twitch"""
        if self.streaming:
            log.info("Stopping Twitch stream")
            
            # Stop the pipeline
            if self.streaming_pipeline:
                self.streaming_pipeline.send_event(Gst.Event.new_eos())
                self.streaming_pipeline.set_state(Gst.State.NULL)
                self.streaming_pipeline = None
                self.streaming_src = None
            
            self.streaming = False
    
    def toggle_streaming(self) -> None:
        """Toggle streaming on/off"""
        if self.streaming:
            self.stop_streaming()
        else:
            self.start_streaming()
    
    def push_frame_to_recording(self, frame: np.ndarray) -> None:
        """Push a frame to the recording pipeline"""
        if not self.recording or self.recording_paused:
            return
            
        if self.recording_src is None:
            return
            
        # Increment frame counter
        self.frame_count += 1
        
        # Convert frame to GStreamer buffer
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        
        # Set buffer timestamp
        duration = 1 / 30 * Gst.SECOND  # Assuming 30 fps
        pts = (time.time() - self.start_time) * Gst.SECOND
        buf.pts = pts
        buf.duration = duration
        
        # Push buffer to pipeline
        ret = self.recording_src.emit('push-buffer', buf)
        if ret != Gst.FlowReturn.OK:
            log.warning(f"Error pushing buffer to recording: {ret}")
    
    def push_frame_to_streaming(self, frame: np.ndarray) -> None:
        """Push a frame to the streaming pipeline"""
        if not self.streaming or self.streaming_src is None:
            return
            
        # Convert frame to GStreamer buffer
        data = frame.tobytes()
        buf = Gst.Buffer.new_allocate(None, len(data), None)
        buf.fill(0, data)
        
        # Set buffer timestamp
        duration = 1 / 30 * Gst.SECOND  # Assuming 30 fps
        pts = time.time() * Gst.SECOND
        buf.pts = pts
        buf.duration = duration
        
        # Push buffer to pipeline
        ret = self.streaming_src.emit('push-buffer', buf)
        if ret != Gst.FlowReturn.OK:
            log.warning(f"Error pushing buffer to streaming: {ret}")
    
    async def _create_debug_view(self) -> np.ndarray:
        """Create debug view based on current view mode"""
        # First create a clean frame without any text overlays
        clean_frame = None
        
        if self.view_mode == ViewMode.OUTPUT and self.output_frame is not None:
            clean_frame = self.output_frame.copy()
            view = clean_frame.copy()  # Create a separate copy for display
            
            # Store the clean frame for recording
            self.clean_frame_for_recording = clean_frame
            
            # Add UI elements to the display view only
            view = self._add_ui_elements(view)
            return view
            
        frames = []
        clean_frames = []  # For storing frames without overlays
        
        for name, camera in self.cameras.items():
            if camera.last_frame is None:
                continue
            
            frame = camera.last_frame.copy()
            clean_frame_copy = frame.copy()  # Make a clean copy before adding overlays
            
            # skip inactive cameras in ACTIVE mode
            if self.view_mode == ViewMode.ACTIVE and not camera.active:
                continue
            
            # handle motion debug view
            if self.view_mode == ViewMode.MOTION:
                small = cv2.resize(frame, camera.motion_res)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if camera.previous_frame is not None:
                    diff = cv2.absdiff(camera.previous_frame, gray)
                    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
                    frame = cv2.resize(diff_color, camera.resolution)
                    clean_frame_copy = frame.copy()  # Update clean copy for motion view
                camera.previous_frame = gray
            
            # Add a small, subtle camera name overlay in the bottom-left corner
            # Create a semi-transparent background for better readability
            overlay = frame.copy()
            
            # Build the camera name text with status indicators
            text = name
            if camera.active:
                text += " [active]"
                
            text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
            
            # Draw a semi-transparent background rectangle
            bg_x = 10
            bg_y = frame.shape[0] - 10 - text_size[1] - 10  # 10px padding
            bg_w = text_size[0] + 20  # 10px padding on each side
            bg_h = text_size[1] + 10  # 5px padding on top and bottom
            
            cv2.rectangle(overlay, (bg_x, bg_y), (bg_x + bg_w, bg_y + bg_h), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)  # Apply transparency
            
            # Add camera name text with status indicators
            cv2.putText(frame, text, 
                      (bg_x + 10, bg_y + bg_h - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
            frames.append((name, frame))
            clean_frames.append((name, clean_frame_copy))
            
        if not frames:
            blank = np.zeros((1080, 1920, 3), dtype=np.uint8)
            self.clean_frame_for_recording = blank
            blank = self._add_ui_elements(blank)
            return blank

        # Handle PiP mode
        if self.view_mode == ViewMode.PIP:
            output = np.zeros((1080, 1920, 3), dtype=np.uint8)
            clean_output = np.zeros((1080, 1920, 3), dtype=np.uint8)  # Clean version for recording
            
            # Select main camera
            main_camera = self._select_main_camera()
            if main_camera is None:
                main_camera = frames[0][0]  # fallback to first camera
            
            # Update main_camera flags
            for name, camera in self.cameras.items():
                camera.main_camera = (name == main_camera)
            
            # Get main camera frame (both display and clean versions)
            main_frame = next(frame for name, frame in frames if name == main_camera)
            main_frame = cv2.resize(main_frame, (1920, 1080))
            output = main_frame
            
            # Get clean main camera frame
            clean_main_frame = next(frame for name, frame in clean_frames if name == main_camera)
            clean_main_frame = cv2.resize(clean_main_frame, (1920, 1080))
            clean_output = clean_main_frame
            
            # Add smaller overlays only for active cameras (excluding main camera)
            pip_width = 480  # 1/4 of screen width
            pip_height = 270  # Keep 16:9 aspect ratio
            padding = 20  # Space between PiP windows
            
            active_frames = [(name, frame) for name, frame in frames 
                           if name != main_camera and self.cameras[name].active]
            
            active_clean_frames = [(name, frame) for name, frame in clean_frames 
                                 if name != main_camera and self.cameras[name].active]
            
            for i, ((name, frame), (_, clean_frame)) in enumerate(zip(active_frames, active_clean_frames)):
                # Calculate position for PiP
                x = 1920 - pip_width - padding
                y = padding + i * (pip_height + padding)
                
                # Skip if we run out of vertical space
                if y + pip_height > 1080:
                    break
                    
                # Resize and overlay PiP for display view
                pip = cv2.resize(frame, (pip_width, pip_height))
                region = output[y:y+pip_height, x:x+pip_width]
                overlay = cv2.addWeighted(pip, 0.8, region, 0.2, 0)
                output[y:y+pip_height, x:x+pip_width] = overlay
                
                # Resize and overlay PiP for clean recording view
                clean_pip = cv2.resize(clean_frame, (pip_width, pip_height))
                clean_region = clean_output[y:y+pip_height, x:x+pip_width]
                clean_overlay = cv2.addWeighted(clean_pip, 0.8, clean_region, 0.2, 0)
                clean_output[y:y+pip_height, x:x+pip_width] = clean_overlay
            
            # Store clean output for recording
            self.clean_frame_for_recording = clean_output
            
            # Add UI elements to the display view only
            output = self._add_ui_elements(output)
            return output
            
        # Handle other view modes (grid layout)
        frames_display = [frame for _, frame in frames]  # Extract just the frames for display
        frames_clean = [frame for _, frame in clean_frames]  # Extract clean frames for recording
        
        n = len(frames_display)
        grid_size = int(np.ceil(np.sqrt(n)))
        cell_w = 1920 // grid_size
        cell_h = 1080 // grid_size
        
        output = np.zeros((1080, 1920, 3), dtype=np.uint8)
        clean_output = np.zeros((1080, 1920, 3), dtype=np.uint8)
        
        for i, (frame, clean_frame) in enumerate(zip(frames_display, frames_clean)):
            y = (i // grid_size) * cell_h
            x = (i % grid_size) * cell_w
            
            # Resize for display view
            resized = cv2.resize(frame, (cell_w, cell_h))
            output[y:y+cell_h, x:x+cell_w] = resized
            
            # Resize for clean recording view
            clean_resized = cv2.resize(clean_frame, (cell_w, cell_h))
            clean_output[y:y+cell_h, x:x+cell_w] = clean_resized
        
        # Store clean output for recording
        self.clean_frame_for_recording = clean_output
        
        # Add UI elements to the display view only
        output = self._add_ui_elements(output)
        return output
        
    def _add_ui_elements(self, frame: np.ndarray) -> np.ndarray:
        """Add UI elements like toolbars and indicators to the frame"""
        # Get original frame dimensions
        h, w = frame.shape[:2]
        
        # Constants for UI elements
        top_bar_height = 60
        bottom_bar_height = 50
        tab_width = w // 5  # 5 view modes
        
        # Create a larger canvas to accommodate the toolbars
        canvas_height = h + top_bar_height + bottom_bar_height
        canvas = np.zeros((canvas_height, w, 3), dtype=np.uint8)
        
        # Create top toolbar (dark gray background)
        top_bar = np.ones((top_bar_height, w, 3), dtype=np.uint8) * 40  # Dark gray
        
        # Create bottom toolbar (dark gray background)
        bottom_bar = np.ones((bottom_bar_height, w, 3), dtype=np.uint8) * 40  # Dark gray
        
        # Place the frame and toolbars on the canvas
        canvas[top_bar_height:top_bar_height+h, :] = frame  # Place frame in the middle
        canvas[:top_bar_height, :] = top_bar  # Place top toolbar
        canvas[top_bar_height+h:, :] = bottom_bar  # Place bottom toolbar
        
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
        
        # Create a unified status bar in the top-right corner
        # First, determine what indicators we need to show
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
            
            # Create overlay for semi-transparent background
            overlay = canvas.copy()
            cv2.rectangle(overlay, 
                        (status_bar_x, status_bar_y), 
                        (w - padding, status_bar_y + status_bar_height), 
                        (40, 40, 40), -1)  # Dark gray background
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)  # Apply transparency
            
            # Draw a subtle border around the status bar
            cv2.rectangle(canvas, 
                        (status_bar_x, status_bar_y), 
                        (w - padding, status_bar_y + status_bar_height), 
                        (100, 100, 100), 1)  # Light gray border
            
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
        
    async def _composite_output(self) -> None:
        """Create composite output from active cameras"""
        while self.running:
            active_frames = []
            
            # collect frames from active cameras
            for name, camera in self.cameras.items():
                if camera.active:
                    try:
                        frame = await self.frame_buffer[name].get()
                        active_frames.append((name, frame))
                    except asyncio.QueueEmpty:
                        continue
            
            if active_frames:
                # create grid layout based on number of active cameras
                n = len(active_frames)
                grid_size = int(np.ceil(np.sqrt(n)))
                cell_w = 1920 // grid_size
                cell_h = 1080 // grid_size
                
                # create blank output frame
                output = np.zeros((1080, 1920, 3), dtype=np.uint8)
                
                # place active frames in grid
                for i, (name, frame) in enumerate(active_frames):
                    y = (i // grid_size) * cell_h
                    x = (i % grid_size) * cell_w
                    resized = cv2.resize(frame, (cell_w, cell_h))
                    output[y:y+cell_h, x:x+cell_w] = resized
                
                self.output_frame = output
                
            await asyncio.sleep(1/30)
            
    async def start(self) -> None:
        """Start the stream processing"""
        log.info("starting workshop stream")
        self.running = True
        
        # create tasks for all cameras and compositor
        tasks = [
            asyncio.create_task(self._capture_frames(camera))
            for camera in self.cameras.values()
        ]
        tasks.append(asyncio.create_task(self._composite_output()))
        
        # Add debug viewer task if debug mode is enabled
        if self.debug:
            async def debug_viewer():
                try:
                    while self.running:
                        view = await self._create_debug_view()
                        
                        # Auto-recording based on camera activity
                        if self.auto_recording:
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
                        
                        # Push frame to recording if active
                        if self.recording:
                            # Use the clean_frame_for_recording which has no text overlays
                            if hasattr(self, 'clean_frame_for_recording') and self.clean_frame_for_recording is not None:
                                self.push_frame_to_recording(self.clean_frame_for_recording)
                            else:
                                # Fallback to output_frame if clean_frame_for_recording is not available
                                if self.output_frame is not None:
                                    self.push_frame_to_recording(self.output_frame)
                                else:
                                    # Last resort fallback to debug view
                                    self.push_frame_to_recording(view)
                        
                        # Push frame to streaming if active
                        if self.streaming:
                            # Use the clean_frame_for_recording which has no text overlays
                            if hasattr(self, 'clean_frame_for_recording') and self.clean_frame_for_recording is not None:
                                self.push_frame_to_streaming(self.clean_frame_for_recording)
                            else:
                                # Fallback to output_frame if clean_frame_for_recording is not available
                                if self.output_frame is not None:
                                    self.push_frame_to_streaming(self.output_frame)
                                else:
                                    # Last resort fallback to debug view
                                    self.push_frame_to_streaming(view)
                        
                        cv2.imshow('Debug View', view)
                        
                        key = cv2.waitKey(1) & 0xFF
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
                            log.info(f"TAB pressed. Cycling camera selection.")
                            
                            # Get list of active cameras
                            active_cameras = [name for name, cam in self.cameras.items() if cam.active]
                            
                            if active_cameras:
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

                        await asyncio.sleep(1/30)
                finally:
                    cv2.destroyAllWindows()
                    if self.recording:
                        self.stop_recording()
            
            tasks.append(asyncio.create_task(debug_viewer()))
        
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutting down...")
            self.running = False
            if self.recording:
                self.stop_recording()
            if self.streaming:
                self.stop_streaming()
            for task in tasks:
                if not task.done():
                    task.cancel()
        
    def stop(self) -> None:
        """Stop the stream processing"""
        self.running = False
        if self.recording:
            self.stop_recording()
        if self.streaming:
            self.stop_streaming()
        if self.debug:
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # quick test with debug viewer
    stream = WorkshopStream(debug=True)
    stream.add_camera("rtsp://192.168.1.114:8080/h264_pcm.sdp", "desk")
    stream.add_camera("rtsp://192.168.1.112:8080/h264_pcm.sdp", "wide")
    
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