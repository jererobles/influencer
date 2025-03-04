import asyncio
import ffmpeg
import logging
from datetime import datetime
import os
from pathlib import Path
import signal
from typing import Dict, List, Optional
from dataclasses import dataclass
import json
from collections import defaultdict
import subprocess  # we need this instead of ffmpeg.Process

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

@dataclass
class StreamMetrics:
    """Track streaming metrics per camera"""
    total_frames: int = 0
    dropped_frames: int = 0
    total_segments: int = 0
    current_fps: float = 0
    current_bitrate: str = "N/A"
    last_error: str = ""
    last_error_time: Optional[datetime] = None
    reconnects: int = 0
    
    def to_dict(self):
        return {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "total_segments": self.total_segments,
            "current_fps": self.current_fps,
            "current_bitrate": self.current_bitrate,
            "last_error": self.last_error,
            "last_error_time": self.last_error_time.isoformat() if self.last_error_time else None,
            "reconnects": self.reconnects
        }

class StreamRecorder:
    def __init__(self, output_dir: str = "recordings"):
        # Check if ffmpeg is available
        try:
            ffmpeg.probe('dummy')
        except ffmpeg.Error:
            pass  # Expected error for dummy input
        except FileNotFoundError:
            log.error("FFmpeg not found! Please install FFmpeg first.")
            log.error("On macOS: brew install ffmpeg")
            log.error("On Ubuntu: sudo apt install ffmpeg")
            raise SystemExit(1)

        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Create metrics directory
        self.metrics_dir = self.output_dir / "metrics"
        self.metrics_dir.mkdir(exist_ok=True)
        
        self.cameras = [
            {
                "name": "living",
                "url": "rtsp://192.168.1.114:8080/h264_pcm.sdp",
            },
        ]
        
        self.processes: Dict[str, subprocess.Popen] = {}  # changed from ffmpeg.Process
        self.metrics: Dict[str, StreamMetrics] = defaultdict(StreamMetrics)
        self.should_run = True
        
        # Handle graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        log.info("Shutdown signal received, stopping recordings...")
        self.should_run = False
        self.stop_all_recordings()
        self.save_metrics()

    def save_metrics(self):
        """Save current metrics to json file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        metrics_file = self.metrics_dir / f"metrics_{timestamp}.json"
        
        metrics_dict = {
            camera: metrics.to_dict()
            for camera, metrics in self.metrics.items()
        }
        
        with open(metrics_file, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        
        log.info(f"Saved metrics to {metrics_file}")

    async def parse_progress(self, camera_name: str, line: bytes):
        """Parse ffmpeg progress output and update metrics"""
        try:
            line = line.decode('utf-8').strip()
            if '=' not in line:
                return
                
            key, value = line.split('=', 1)
            metrics = self.metrics[camera_name]
            
            if key == "frame":
                metrics.total_frames = int(value)
            elif key == "fps":
                metrics.current_fps = float(value)
            elif key == "bitrate":
                metrics.current_bitrate = value
            elif key == "drop_frames":
                metrics.dropped_frames = int(value)
                if metrics.dropped_frames > 30:
                    log.warning(f"{camera_name} dropping frames: {metrics.dropped_frames}")
            
            # Periodically save metrics
            if metrics.total_frames % 1000 == 0:
                self.save_metrics()
                
        except Exception as e:
            log.error(f"Error parsing progress for {camera_name}: {e}")

    async def monitor_process(self, camera_name: str, process: subprocess.Popen):
        """Monitor ffmpeg process output and update metrics"""
        while True:
            try:
                if process.stdout is None:
                    break
                    
                line = await asyncio.get_event_loop().run_in_executor(
                    None, process.stdout.readline
                )
                if not line:
                    break
                    
                await self.parse_progress(camera_name, line)
                
            except Exception as e:
                log.error(f"Monitor error for {camera_name}: {e}")
                break

    async def start_recording(self, camera: dict):
        """Start recording a single camera stream"""
        while self.should_run:
            try:
                log.info(f"Starting recording for {camera['name']}")
                
                stream = (
                    ffmpeg
                    .input(
                        camera["url"], 
                        rtsp_transport="tcp",
                        use_wallclock_as_timestamps=1,
                        stimeout='5000000',
                        reconnect=1,
                        reconnect_at_eof=1,
                        reconnect_streamed=1,
                        reconnect_delay_max=5,
                    )
                    .output(
                        str(self.output_dir / f"{camera['name']}_%Y%m%d_%H%M%S.mkv"),
                        codec="copy",
                        format="segment",
                        segment_time=300,
                        segment_format="matroska",
                        segment_atclocktime=1,
                        strftime=1,
                        reset_timestamps=1,
                        write_empty_segments=0
                    )
                    .overwrite_output()
                    .global_args('-progress', 'pipe:1')
                )
                
                # Get command for subprocess
                cmd = ffmpeg.compile(stream)
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE
                )
                
                self.processes[camera["name"]] = process
                self.metrics[camera["name"]].total_segments += 1
                
                # Start monitoring process output
                monitor_task = asyncio.create_task(
                    self.monitor_process(camera["name"], process)
                )
                
                # Wait for process to complete
                await asyncio.get_event_loop().run_in_executor(
                    None, process.wait
                )
                
                if self.should_run:
                    log.warning(f"FFmpeg process for {camera['name']} exited, restarting...")
                    self.metrics[camera["name"]].reconnects += 1
                    await asyncio.sleep(5)
                    
            except ffmpeg.Error as e:
                error_msg = e.stderr.decode() if e.stderr else str(e)
                log.error(f"FFmpeg error for {camera['name']}: {error_msg}")
                self.metrics[camera["name"]].last_error = error_msg
                self.metrics[camera["name"]].last_error_time = datetime.now()
                await asyncio.sleep(5)
                
            except Exception as e:
                log.error(f"Error recording {camera['name']}: {e}")
                self.metrics[camera["name"]].last_error = str(e)
                self.metrics[camera["name"]].last_error_time = datetime.now()
                await asyncio.sleep(5)

    def stop_all_recordings(self):
        """Stop all active recording processes"""
        for camera_name, process in self.processes.items():
            log.info(f"Stopping recording for {camera_name}")
            try:
                process.terminate()
                process.wait(timeout=5)
            except Exception as e:
                log.error(f"Error stopping {camera_name}: {e}")
                process.kill()

    async def save_metrics_loop(self):
        """Periodically save metrics"""
        try:
            while self.should_run:
                try:
                    await asyncio.sleep(300)  # every 5 minutes
                    self.save_metrics()
                except asyncio.CancelledError:
                    break
        finally:
            self.save_metrics()  # Save one last time before exiting

    async def run(self):
        """Start recording all configured cameras"""
        log.info("Starting camera recorder...")
        
        # Create tasks list to track all tasks
        all_tasks = []
        
        try:
            # Start recording tasks
            recording_tasks = [
                asyncio.create_task(self.start_recording(camera))
                for camera in self.cameras
            ]
            all_tasks.extend(recording_tasks)
            
            # Start metrics saving task
            metrics_task = asyncio.create_task(self.save_metrics_loop())
            all_tasks.append(metrics_task)
            
            # Wait for all tasks to complete
            await asyncio.gather(*all_tasks)
                
        except asyncio.CancelledError:
            log.info("Shutdown requested, cancelling tasks...")
            # Cancel all tasks
            for task in all_tasks:
                task.cancel()
            # Wait for tasks to finish cancellation
            await asyncio.gather(*all_tasks, return_exceptions=True)
            log.info("All tasks cancelled")
        finally:
            self.stop_all_recordings()
            self.save_metrics()

if __name__ == "__main__":
    recorder = StreamRecorder()
    asyncio.run(recorder.run())