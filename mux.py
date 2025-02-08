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

class ViewMode(Enum):
    GRID = auto()      # show all cameras in grid
    ACTIVE = auto()    # show only active cameras
    OUTPUT = auto()    # show final composite output
    MOTION = auto()    # show motion detection debug view

class WorkshopStream:
    def __init__(self, debug: bool = False):
        self.cameras: Dict[str, Camera] = {}
        self.frame_buffer: Dict[str, asyncio.Queue] = {}
        self.output_frame: Optional[np.ndarray] = None
        self.running = False
        self.debug = debug
        self.view_mode = ViewMode.OUTPUT
        
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
                            
                            # Update activity state
                            face_count = len(results.detections)
                            if face_count > 0:
                                camera.active = True
                                camera.cooldown = 30
                            elif camera.cooldown > 0:
                                camera.cooldown -= 1
                            else:
                                camera.active = False
                        
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
        
    async def _create_debug_view(self) -> np.ndarray:
        """Create debug view based on current view mode"""
        if self.view_mode == ViewMode.OUTPUT and self.output_frame is not None:
            return self.output_frame.copy()
            
        frames = []
        for name, camera in self.cameras.items():
            # Use the last known good frame instead of waiting for a new one
            if camera.last_frame is None:
                continue
            
            frame = camera.last_frame.copy()
            
            # skip inactive cameras in ACTIVE mode
            if self.view_mode == ViewMode.ACTIVE and not camera.active:
                continue
            
            # handle motion debug view
            if self.view_mode == ViewMode.MOTION:
                small = cv2.resize(frame, camera.motion_res)
                gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
                if camera.previous_frame is not None:
                    diff = cv2.absdiff(camera.previous_frame, gray)
                    # colorize diff for visibility
                    diff_color = cv2.applyColorMap(diff, cv2.COLORMAP_HOT)
                    frame = cv2.resize(diff_color, camera.resolution)
                camera.previous_frame = gray
            
            # add camera name overlay
            cv2.putText(frame, f"{name} {'[ACTIVE]' if camera.active else ''}", 
                      (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            frames.append(frame)
            
        if not frames:
            return np.zeros((1080, 1920, 3), dtype=np.uint8)
            
        # create grid layout
        n = len(frames)
        grid_size = int(np.ceil(np.sqrt(n)))
        cell_w = 1920 // grid_size
        cell_h = 1080 // grid_size
        
        output = np.zeros((1080, 1920, 3), dtype=np.uint8)
        for i, frame in enumerate(frames):
            y = (i // grid_size) * cell_h
            x = (i % grid_size) * cell_w
            resized = cv2.resize(frame, (cell_w, cell_h))
            output[y:y+cell_h, x:x+cell_w] = resized
            
        return output
        
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
                        cv2.imshow('Debug View', view)
                        
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            raise KeyboardInterrupt  # Trigger clean shutdown
                        elif key == ord('g'):
                            self.view_mode = ViewMode.GRID
                        elif key == ord('a'):
                            self.view_mode = ViewMode.ACTIVE
                        elif key == ord('o'):
                            self.view_mode = ViewMode.OUTPUT
                        elif key == ord('m'):
                            self.view_mode = ViewMode.MOTION
                        
                        await asyncio.sleep(1/30)
                finally:
                    cv2.destroyAllWindows()
            
            tasks.append(asyncio.create_task(debug_viewer()))
        
        try:
            await asyncio.gather(*tasks)
        except (KeyboardInterrupt, asyncio.CancelledError):
            log.info("Shutting down...")
            self.running = False
            for task in tasks:
                if not task.done():
                    task.cancel()
        
    def stop(self) -> None:
        """Stop the stream processing"""
        self.running = False
        if self.debug:
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # quick test with debug viewer
    stream = WorkshopStream(debug=True)
    stream.add_camera("http://192.168.1.114:8080/video", "desk")
    stream.add_camera("rtsp://192.168.1.112:8080/h264_pcm.sdp", "wide")
    
    print("Debug controls:")
    print("  g - grid view (all cameras)")
    print("  a - active cameras only")
    print("  o - composite output")
    print("  m - motion detection debug")
    print("  q - quit")
    
    try:
        asyncio.run(stream.start())
    except KeyboardInterrupt:
        stream.stop()