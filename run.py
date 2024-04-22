import cv2
import threading
import queue
import numpy as np
import asyncio

from edgetpu.Model import Model
from edgetpu.utils import resize_and_pad, get_image_tensor, save_one_json, coco80_to_coco91_class
from image import cell_index_from_detected_objects, create_grid_layout
from obs import connect_to_obs, switch_obs_scene
from video.Stream import Stream
from collections import deque

ENABLE_PREVIEW = True
CONTROL_OBS = False
CAPTURE_FPS = 5  # Capture at 5 FPS
FRAME_INTERVAL = 1  # Process every 3rd frame
GRID_CELL = 512
GRID_SHAPE = (2, 2)

model = Model("assets/yolov5s-int8-224_edgetpu.tflite", "assets/coco.yaml", conf_thresh=0.5, iou_thresh=0.65)
input_size = model.get_image_size()
x = (255 * np.random.random((3, *input_size))).astype(np.uint8)
model.forward(x)
cameras = [
    # {
    #     "url": "rtsp://camera:camera@192.168.1.113/stream2",
    #     "obs": "Kitchen",
    #     "tag": "TP-Link Tapo C200",
    #     "zoom": 1
    # },
    # {
    #     "url": "rtsp://192.168.1.114:8080/h264_pcm.sdp",
    #     "obs": "Samsung",
    #     "tag": "Samsung",
    #     "zoom": 1
    # },
    # {
    #     "url": "rtsp://admin:234680@192.168.1.84:554/stream2",
    #     "obs": "Kitchen",
    #     "tag": "TP-Link",
    #     "zoom": 3.9
    # },
    {
        "url": 0,
        "obs": "Desk",
        "tag": "ElGato Facecam",
        "zoom": 1
    },
    {
        "url": 1,
        "obs": "iPhone13",
        "tag": "iPhone13 Continuity Camera",
        "zoom": 1
    },
    # {
    #     "url": "rtsp://192.168.1.112:8080/h264_pcm.sdp",
    #     "obs": "Redmi",
    #     "tag": "Redmi",
    #     "zoom": 1
    # },
    # {
    #     "url": "rtsp://192.168.1.106:8554/live.sdp",
    #     "obs": "Bedroom",
    #     "tag": "iPhone13 RTSP"
    # },
]

def capture_frames(id, cam):
    global do_capture, CAPTURE_FPS, FRAME_INTERVAL
    frame_count = 0
    cap = Stream(cam['url'], CAPTURE_FPS)  # TODO experiment with different FPS values
    while cap.isOpened() and do_capture:
        ret, next_frame = cap.read()
        frame_count = frame_count + 1
        if ret is True:
            if frame_count % FRAME_INTERVAL == 0:
                with frame_locks[id]:
                    latest_frames[id] = next_frame  # Update the latest frame for the camera

# Function to detect presence in a frame using YOLO
def detect_presence(frame):
    full_image, net_image, pad = get_image_tensor(frame, input_size[0])
    pred = model.forward(net_image)

    try:
        det = model.process_predictions(pred[0], full_image, pad)
        frame = full_image
        return det
    except:
        return False

latest_frames = [None] * len(cameras)
frame_locks = [threading.Lock() for _ in range(len(cameras))]
preview_frames_queue = queue.Queue()

# Global variable to store the current camera source
current_source = None
do_process = True
do_capture = True

# Function to process frames from the queue
def process_frames():
    global current_source, do_process
    while do_process:
        try:
            frames = []
            for i in range(len(cameras)):
                with frame_locks[i]:
                    if latest_frames[i] is not None:
                        frames.append((cameras[i], latest_frames[i]))

            if frames:
                grid_frame = create_grid_layout(GRID_CELL, GRID_SHAPE, frames, cameras)
                detected_objects = detect_presence(grid_frame)
                if ENABLE_PREVIEW:
                    preview_frames_queue.put(grid_frame)

                if detected_objects is not False:
                    cam_index = cell_index_from_detected_objects(GRID_CELL, GRID_SHAPE, model.names, detected_objects)
                    if cam_index is not None:
                        detected_cam = cameras[cam_index]
                        if current_source != detected_cam:
                            print(f"Switched to {detected_cam}")
                            current_source = detected_cam
        except Exception as e:
            print("process_frames:", e)
            pass
        except KeyboardInterrupt as e:
            break
            

async def main():
    global do_process, do_capture
    threads = []

    if CONTROL_OBS:
        await connect_to_obs()

    process_thread = threading.Thread(target=process_frames)
    threads.append(process_thread)
    process_thread.start()

    for i, cam in enumerate(cameras):
        capture_thread = threading.Thread(target=capture_frames, args=(i, cam))
        threads.append(capture_thread)
        capture_thread.start()

    prev_source = None

    # Main thread loop
    while True:
        try:
            # Check if there are frames in the queue
            if ENABLE_PREVIEW and not preview_frames_queue.empty():
                frame = preview_frames_queue.get()
                # Display the frame
                cv2.imshow(f"IP Camera Feed", frame)

            # Check if the current source has changed
            if current_source != prev_source:
                if CONTROL_OBS:
                    await switch_obs_scene(current_source)
                prev_source = current_source

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except Exception as e:
            print(e)
            pass
        except KeyboardInterrupt as e:
            break

    do_process = False
    do_capture = False

    # Release the video capture and close windows
    cv2.destroyAllWindows()

    # Wait for the threads to finish
    for thread in threads:
        thread.join()

# Run the main function
asyncio.run(main())