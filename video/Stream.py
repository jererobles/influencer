from time import time
from typing import Union
import cv2


class Stream():
    """
    extends [cv2::VideoCapture class](https://docs.opencv.org/3.4/d8/dfe/classcv_1_1VideoCapture.html)
    for video or stream subsampling.

    Parameters
    ----------
    filename : Union[str, int]
        Open video file or image file sequence or a capturing device
        or a IP video stream for video capturing.
    target_fps : int, optional
        the target frame rate. To ensure a constant time period between
        each subsampled frames, this parameter is used to compute a
        integer denominator for the extraction frequency. For instance,
        if the original stream is 64fps and you want a 30fps stream out,
        it is going to take one frame over two giving an effective frame
        rate of 32fps.
        If None, will extract every frame of the stream.
    """

    def __init__(self, filename: Union[str, int], target_fps: int = None):
        self.stream_id = filename
        self._cap = cv2.VideoCapture(self.stream_id)
        if not self.isOpened():
            raise FileNotFoundError("Stream not found")

        self.target_fps = target_fps
        self.fps = None
        self.extract_freq = None
        self.compute_extract_frequency()
        self._frame_index = 0

    def compute_extract_frequency(self):
        """evaluate the frame rate over a period of 5 seconds"""
        self.fps = self._cap.get(cv2.CAP_PROP_FPS)
        if self.fps == 0:
            self.compute_origin_fps()

        if self.target_fps is None:
            self.extract_freq = 1
        else:
            self.extract_freq = int(self.fps / self.target_fps)

            if self.extract_freq == 0:
                pass
                # raise ValueError("desired_fps is higher than half the stream frame rate")

    def compute_origin_fps(self, evaluation_period: int = 5):
        """evaluate the frame rate over a period of 5 seconds"""
        while self.isOpened():
            ret, _ = self._cap.read()
            if ret is True:
                if self._frame_index == 0:
                    start = time()

                self._frame_index += 1

                if time() - start > evaluation_period:
                    break

        self.fps = round(self._frame_index / (time() - start), 2)

    def read(self):
        """Grabs, decodes and returns the next subsampled video frame."""
        ret, frame = self._cap.read()
        if ret is True:
            self._frame_index += 1

            if self._frame_index == self.extract_freq:
                self._frame_index = 0
                return ret, frame
            
            return False, frame

        return False, False

    def isOpened(self):
        """Returns true if video capturing has been initialized already."""
        return self._cap.isOpened()

    def release(self):
        """Closes video file or capturing device."""
        self._cap.release()