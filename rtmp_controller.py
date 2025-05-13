from __future__ import annotations

import asyncio
import logging
from typing import Optional

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst

from pyrtmp import StreamClosedException
from pyrtmp.rtmp import SimpleRTMPController
from pyrtmp.session_manager import SessionManager

from camera import Camera

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class RTMP2GStreamerController(SimpleRTMPController):
    def __init__(self, camera_manager):
        self.camera_manager = camera_manager
        super().__init__()

    async def on_ns_publish(self, session, message) -> None:
        publishing_name = message.publishing_name
        logger.info(f"New RTMP stream: {publishing_name}")
        
        # Create a new camera instance for this stream
        camera = Camera(
            url=f"rtmp://127.0.0.1:1935/{publishing_name}",
            name=publishing_name
        )
        
        # Create GStreamer pipeline for RTMP
        pipeline_str = (
            'appsrc name=sink is-live=true format=time do-timestamp=true ! '
            'queue max-size-buffers=4096 max-size-bytes=0 max-size-time=0 ! '
            'h264parse ! avdec_h264 max-threads=4 ! '
            'videoconvert ! video/x-raw,format=BGR ! '
            'appsink name=output emit-signals=True max-buffers=4096 drop=False'
        )
        
        try:
            camera.pipeline = Gst.parse_launch(pipeline_str)
            camera.sink = camera.pipeline.get_by_name('sink')
            camera.output = camera.pipeline.get_by_name('output')
            
            # Set up bus watch
            bus = camera.pipeline.get_bus()
            bus.add_signal_watch()
            bus.connect('message', self._on_bus_message, camera)
            
            # Start pipeline
            camera.pipeline.set_state(Gst.State.PLAYING)
            
            # Add camera to manager
            self.camera_manager.add_camera(camera)
            session.state = camera
            
            logger.info(f"RTMP stream {publishing_name} pipeline started")
            
        except Exception as e:
            logger.error(f"Failed to create pipeline for {publishing_name}: {e}")
            raise
            
        await super().on_ns_publish(session, message)

    def _on_bus_message(self, bus, message, camera: Camera) -> bool:
        t = message.type
        if t == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            logger.error(f"Pipeline error for {camera.name}: {err.message}")
            logger.debug(f"Debug info: {debug}")
            camera.connection_lost = True
        elif t == Gst.MessageType.STATE_CHANGED:
            old_state, new_state, pending_state = message.parse_state_changed()
            if message.src == camera.pipeline:
                logger.debug(f"Pipeline state changed for {camera.name}: {old_state.value_nick} -> {new_state.value_nick}")
        elif t == Gst.MessageType.EOS:
            logger.warning(f"End of stream for {camera.name}")
            camera.connection_lost = True
        return True

    async def on_video_message(self, session, message) -> None:
        camera: Optional[Camera] = session.state
        if not camera or not camera.sink:
            return
            
        try:
            # Create GStreamer buffer from RTMP video data
            buffer = Gst.Buffer.new_wrapped(message.payload)
            buffer.pts = message.timestamp * Gst.SECOND
            
            # Push to pipeline
            ret = camera.sink.emit('push-buffer', buffer)
            if ret != Gst.FlowReturn.OK:
                logger.warning(f"Failed to push buffer to {camera.name}: {ret}")
                
        except Exception as e:
            logger.error(f"Error processing video message for {camera.name}: {e}")
            camera.connection_lost = True
            
        await super().on_video_message(session, message)

    async def on_stream_closed(self, session: SessionManager, exception: StreamClosedException) -> None:
        camera: Optional[Camera] = session.state
        if camera:
            logger.info(f"RTMP stream closed: {camera.name}")
            camera.connection_lost = True
            if camera.pipeline:
                camera.pipeline.set_state(Gst.State.NULL)
                camera.pipeline = None
                camera.sink = None
                camera.output = None
        await super().on_stream_closed(session, exception) 