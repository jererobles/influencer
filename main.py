import asyncio
from datetime import datetime
import cv2
import numpy as np
from aiomqtt import Client, Message
import simpleobsws
from image import create_grid_layout
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%H:%M:%S'
)
log = logging.getLogger(__name__)

class WorkshopController:
    def __init__(self):
        self.cameras = [
            {
                "url": "rtsp://192.168.1.114:8080/h264_pcm.sdp",
                "obs_scene": "Living",
                "privacy_topic": "camera/privacy/living",
                "pir_topic": "sensors/pir/living"
            },
            {
                "url": 0,  
                "obs_scene": "Office", 
                "privacy_topic": "camera/privacy/office",
                "pir_topic": "sensors/pir/office"
            }
        ]
        
        self.active_zones = set()
        self.privacy_mode = False
        self.current_scene = None
        self.preview_grid = None
        self.frame_buffers = {cam['url']: asyncio.Queue(maxsize=1) for cam in self.cameras}
        
    async def setup_obs(self):
        log.info("connecting to obs...")
        self.obs = simpleobsws.WebSocketClient(
            url="ws://localhost:4455",
            password="569ppzu8grGVY92A"
        )
        await self.obs.connect()
        await self.obs.wait_until_identified()
        log.info("obs connected fr fr")
        
    async def handle_mqtt_messages(self):
        async with Client("homeassistant.local") as client:
            self.mqtt = client  # store for other methods to use
            log.info("mqtt client ready to THROW DOWN")
            
            await client.subscribe("sensors/pir/#")
            await client.subscribe("camera/privacy/#")
            
            async for message in client.messages:
                topic = str(message.topic)
                log.debug(f"got mqtt message: {topic}")
                
                if message.topic.matches("sensors/pir/#"):
                    await self.handle_pir_message(topic, message.payload)
                elif message.topic.matches("camera/privacy/#"):
                    await self.handle_privacy_message(topic, message.payload)
    
    async def handle_pir_message(self, topic, payload):
        zone = topic.split('/')[-1]
        if payload == b'1':
            self.active_zones.add(zone)
            log.info(f"motion detected in {zone}")
            if not self.privacy_mode:
                await self.update_scene()
        else:
            self.active_zones.discard(zone)
            log.debug(f"motion cleared in {zone}")
    
    async def handle_privacy_message(self, topic, payload):
        if topic == "camera/privacy/control":
            self.privacy_mode = payload == b'1'
            log.info(f"privacy mode {'ENGAGED' if self.privacy_mode else 'disengaged'}")
            
            if self.privacy_mode:
                await self.obs.call(simpleobsws.Request('SetCurrentProgramScene', {'sceneName': 'BRB'}))
                log.info("switching to BRB scene")
                
                for cam in self.cameras:
                    await self.mqtt.publish(cam['privacy_topic'], b'1')
                log.info("mechanical shutters DEPLOYED")
            else:
                for cam in self.cameras:
                    await self.mqtt.publish(cam['privacy_topic'], b'0')
                log.info("shutters retracted, nature is healing")
                await self.update_scene()
    
    async def capture_frames(self, camera):
        log.info(f"starting capture from {camera['url']}")
        cap = cv2.VideoCapture(camera['url'])
        
        while True:
            ret, frame = cap.read()
            if ret and not self.privacy_mode:
                try:
                    self.frame_buffers[camera['url']].put_nowait(frame)
                except asyncio.QueueFull:
                    try:
                        self.frame_buffers[camera['url']].get_nowait()
                    except asyncio.QueueEmpty:
                        pass
                    await self.frame_buffers[camera['url']].put(frame)
            await asyncio.sleep(1/30)
    
    async def update_scene(self):
        for zone in self.active_zones:
            for cam in self.cameras:
                if zone in cam['pir_topic']:
                    if self.current_scene != cam['obs_scene']:
                        log.info(f"switching scene to {cam['obs_scene']}")
                        await self.obs.call(simpleobsws.Request(
                            'SetCurrentProgramScene',
                            {'sceneName': cam['obs_scene']}
                        ))
                        self.current_scene = cam['obs_scene']
                    return
    
    async def update_preview_grid(self):
        log.info("starting preview grid")
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
        log.info("workshop controller starting up...")
        await self.setup_obs()
        
        tasks = [
            asyncio.create_task(self.handle_mqtt_messages()),
            *[asyncio.create_task(self.capture_frames(cam)) for cam in self.cameras],
            asyncio.create_task(self.update_preview_grid())
        ]
        
        try:
            await asyncio.gather(*tasks)
        except KeyboardInterrupt:
            log.info("shutting down (touch grass)...")
            for task in tasks:
                task.cancel()
            cv2.destroyAllWindows()
        except Exception as e:
            log.error(f"something BROKE: {e}")
            raise

if __name__ == "__main__":
    controller = WorkshopController()
    asyncio.run(controller.start())