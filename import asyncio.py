import asyncio
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal
from enum import Enum, auto
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
log = logging.getLogger(__name__)

@dataclass
class Camera:
    url: str
    name: str
    resolution: Tuple[int, int] = (1920, 1080)
    motion_res: Tuple[int, int] = (256, 256)
    motion_threshold: float = 0.1
    previous_frame: Optional[np.ndarray] = None
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
        
    def add_camera(self, url: str, name: str) -> None:
        """Add a new camera to the stream"""
        self.cameras[name] = Camera(url=url, name=name)
        self.frame_buffer[name] = asyncio.Queue(maxsize=1)
        log.info(f"added camera: {name} @ {url}")
        
    async def _capture_frames(self, camera: Camera) -> None:
        """Capture frames from a camera and put them in the buffer"""
        cap = cv2.VideoCapture(camera.url)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera.resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera.resolution[1])
        
        while self.running:
            ret, frame = cap.read()
            if not ret:
                log.warning(f"failed to read frame from {camera.name}")
                await asyncio.sleep(1)
                continue
                
            # motion detection on downscaled frame
            small_frame = cv2.resize(frame, camera.motion_res)
            gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
            
            if camera.previous_frame is None:
                camera.previous_frame = gray
                continue
                
            # compute frame difference
            frame_diff = cv2.absdiff(camera.previous_frame, gray)
            motion_score = np.mean(frame_diff) / 255.0
            camera.active = motion_score > camera.motion_threshold
            
            # update previous frame
            camera.previous_frame = gray
            
            # update frame buffer (clear old frame if full)
            try:
                self.frame_buffer[camera.name].put_nowait(frame)
            except asyncio.QueueFull:
                try:
                    self.frame_buffer[camera.name].get_nowait()
                except asyncio.QueueEmpty:
                    pass
                await self.frame_buffer[camera.name].put(frame)
                
            # update debug view if enabled
            if self.debug:
                debug_view = self._create_debug_view()
                cv2.imshow('Workshop Stream Debug', debug_view)
                key = cv2.waitKey(1)
                
                # handle debug controls
                if key == ord('g'):  # grid view
                    self.view_mode = ViewMode.GRID
                elif key == ord('a'):  # active only
                    self.view_mode = ViewMode.ACTIVE
                elif key == ord('o'):  # output
                    self.view_mode = ViewMode.OUTPUT
                elif key == ord('m'):  # motion debug
                    self.view_mode = ViewMode.MOTION
                elif key == ord('q'):  # quit
                    self.stop()
                    
            await asyncio.sleep(1/30)  # ~30fps
            
        cap.release()
        
    def _create_debug_view(self) -> np.ndarray:
        """Create debug view based on current view mode"""
        if self.view_mode == ViewMode.OUTPUT and self.output_frame is not None:
            return self.output_frame.copy()
            
        frames = []
        for name, camera in self.cameras.items():
            try:
                frame = self.frame_buffer[name].get_nowait()
                
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
                    
                # add camera name overlay
                cv2.putText(frame, f"{name} {'[ACTIVE]' if camera.active else ''}", 
                          (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                frames.append(frame)
            except asyncio.QueueEmpty:
                continue
                
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
            output[y:y+cell_h, x:x+cell_w] = cv2.resize(frame, (cell_w, cell_h))
            
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
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            log.error(f"stream error: {e}")
            raise
        finally:
            self.running = False
            
    def stop(self) -> None:
        """Stop the stream processing"""
        self.running = False
        if self.debug:
            cv2.destroyAllWindows()
        
if __name__ == "__main__":
    # quick test with debug viewer
    stream = WorkshopStream(debug=True)
    stream.add_camera("http://192.168.1.114:8080/video", "desk")
    stream.add_camera("http://192.168.1.112:8080/video", "wide")
    
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