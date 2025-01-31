import asyncio
import json
from datetime import datetime
import cv2
import numpy as np
from aiohttp import ClientSession
from aiomqtt import Client as MQTTClient
import simpleobsws
from image import create_grid_layout  # keeping your based grid code

class WorkshopController:
    def __init__(self):
        self.cameras = [
            {
                "url": "rtsp://192.168.33.17:8080/h264_pcm.sdp",
                "obs_scene": "Living",
                "privacy_topic": "camera/privacy/living",
                "pir_topic": "sensors/pir/living"
            },
            {
                "url": 0,  # built-in cam
                "obs_scene": "Office",
                "privacy_topic": "camera/privacy/office",
                "pir_topic": "sensors/pir/office"
            }
        ]
        
        self.active_zones = set()
        self.privacy_mode = False
        self.current_scene = None
        self.preview_grid = None
        
        # Stream buffers
        self.frame_buffers = {cam['url']: asyncio.Queue(maxsize=1) for cam in self.cameras}
        
    async def setup(self):
        # OBS WebSocket
        self.obs = simpleobsws.WebSocketClient(
            url="ws://localhost:4455",
            password="your_password"
        )
        await self.obs.connect()
        await self.obs.wait_until_identified()
        
        # MQTT for sensors and privacy control
        self.mqtt = MQTTClient("localhost")
        await self.mqtt.connect()
        
        # Subscribe to all PIR sensors and privacy controls
        async with self.mqtt.filtered_messages("sensors/pir/#") as messages:
            await self.mqtt.subscribe("sensors/pir/#")
            async for message in messages:
                await self.handle_pir_message(message.topic, message.payload)
                
        async with self.mqtt.filtered_messages("camera/privacy/#") as messages:
            await self.mqtt.subscribe("camera/privacy/#")
            async for message in messages:
                await self.handle_privacy_message(message.topic, message.payload)
    
    async def handle_pir_message(self, topic, payload):
        zone = topic.split('/')[-1]
        if payload == b'1':
            self.active_zones.add(zone)
            if not self.privacy_mode:
                await self.update_scene()
        else:
            self.active_zones.discard(zone)
    
    async def handle_privacy_message(self, topic, payload):
        if topic == "camera/privacy/control":
            self.privacy_mode = payload == b'1'
            if self.privacy_mode:
                # Kill all streams
                await self.obs.call(simpleobsws.Request('SetCurrentProgramScene', {'sceneName': 'BRB'}))
                # Tell all ESP32s to engage privacy shutters
                for cam in self.cameras:
                    await self.mqtt.publish(cam['privacy_topic'], '1')
            else:
                # Re-enable streams and shutters
                for cam in self.cameras:
                    await self.mqtt.publish(cam['privacy_topic'], '0')
                await self.update_scene()
    
    async def capture_frames(self, camera):
        cap = cv2.VideoCapture(camera['url'])
        while True:
            ret, frame = cap.read()
            if ret and not self.privacy_mode:
                # Keep buffer fresh by dumping old frames
                try:
                    self.frame_buffers[camera['url']].put_nowait(frame)
                except asyncio.QueueFull:
                    try:
                        self.frame_buffers[camera['url']].get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await self.frame_buffers[camera['url']].put(frame)
            await asyncio.sleep(1/30)  # ~30fps
    
    async def update_scene(self):
        """Pick best camera based on PIR data"""
        for zone in self.active_zones:
            for cam in self.cameras:
                if zone in cam['pir_topic']:
                    if self.current_scene != cam['obs_scene']:
                        await self.obs.call(simpleobsws.Request(
                            'SetCurrentProgramScene',
                            {'sceneName': cam['obs_scene']}
                        ))
                        self.current_scene = cam['obs_scene']
                    return
    
    async def update_preview_grid(self):
        """Update the monitoring grid"""
        while True:
            if not self.privacy_mode:
                frames = []
                for cam in self.cameras:
                    try:
                        frame = await self.frame_buffers[cam['url']].get()
                        frames.append((cam, frame))
                    except asyncio.QueueEmpty:
                        continue
                
                if frames:
                    self.preview_grid = create_grid_layout(
                        512, (2, 2), frames, self.cameras
                    )
                    cv2.imshow("Workshop Monitor", self.preview_grid)
                    cv2.waitKey(1)
            await asyncio.sleep(1/30)
    
    async def start(self):
        await self.setup()
        
        # Start all the capture tasks
        capture_tasks = [
            asyncio.create_task(self.capture_frames(cam))
            for cam in self.cameras
        ]
        
        # Start the preview grid
        preview_task = asyncio.create_task(self.update_preview_grid())
        
        try:
            await asyncio.gather(
                *capture_tasks,
                preview_task
            )
        except KeyboardInterrupt:
            for task in capture_tasks:
                task.cancel()
            preview_task.cancel()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    controller = WorkshopController()
    asyncio.run(controller.start())
