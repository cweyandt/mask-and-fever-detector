#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Threaded video processing for dual camera streams. First stream is standard USB camera, 
second stream is FLIR Lepton camera connected via a PureThermal2 I/O board.

Usage:
  dual_streams.py {<USB video device number>|<video file name>}

  Goals:
    1. Open a cv2 stream with cameraID specified as an environment variable
    2. Open a uvc stream which auto-detects the PureThermal USB I/O board
    3. Continually capture frames and put them in buffers
    4. Match frames based on timestamp
    5. Save/Display matched frames, discard unmatched frames 

Keyboard shortcuts:

  ESC - exit
  space - undefined
'''

# Python 2/3 compatibility
from __future__ import print_function

from uvctypes import *
import time
import cv2
import sys
import numpy as np
import platform
from utils import *
from queue import Queue
from collections import deque

# Get CAMERA_INDEX from environment, default to 0
# CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
# MQTT_HOST = os.getenv("MQTT_HOST", "mqtt_broker")
# MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mask-detector")
# MQTT_PORT = int(os.getenv("MQTT_HOST", 1883))
# MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))

# Check to see if a camera ID was passed in as an argument
try:
    cameraID = sys.argv[1]
except:
    cameraID = 0


class videoStream:
    def __init__(self, fps=8, buffer=80):
        self._fps = fps
        self._isReady = self.loadResources()
        self.cap = cv2.VideoCapture(cameraID)
        self._BUFFER = deque([])
        self._bufferLength = buffer
        self.running = False

    def setFPS(self, fps):
        """Adjust Frames Per Second"""
        self._fps = fps

    def getFrameTs(self, ts):
        """return (ts,frame) from buffer with nearest timestamp"""
        for i in range(len(self._BUFFER)):
            if self._BUFFER[i].ts < ts && i==0:
                return self._BUFFER[i].ts, self._BUFFER[i].frame
            if self._BUFFER[i].ts < ts && i>0:
                t1 = abs(ts - self._BUFFER[i].ts) 
                t2 = abs(ts - self._BUFFER[i-1].ts)
                if t1 > t2:
                    return self._BUFFER[i-1].ts, self._BUFFER[i-1].frame
                else:
                    return self._BUFFER[i].ts,  self._BUFFER[i].frame

    def run(self):
        """Start frame capture, place timestamp and frame in buffer"""
        self.running = True
        while self.running:
            # Capture frame-by-frame
            _, frame = self.cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Add the frame to the buffer
            self._BUFFER.appendleft({ts: time(), frame: frame})
            if len(self._BUFFER) > self._bufferLength):
                self._BUFFER.pop()

        # When everything done, release the capture
        self.cap.release()
        return

    def stop(self):
        """Stop fram capture"""
        if self.running:
          print("Stopping videoStream")
          self.running = False
        else:
          print("videoStream cannot stop becaues it is not running")
        return

    def status(self):
        print(f'------- videoStream is {self.running} -------')
        print(f"------- videoStream buffer length: {len(self._BUFFER)} --------")
        for i in range(len(self._BUFFER)): 
            print(self._BUFFER[i].ts)

    def display(self):
        while True:
            cv2.imshow("frame", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        return


def get_usb_frame():
    if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
    else:
        rval = False
    
    frame_v = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)[:,:,2]

    blurredBrightness = cv2.bilateralFilter(frame_v,9,150,150)
    thresh = 50
    edges = cv2.Canny(blurredBrightness,thresh,thresh*2, L2gradient=True)

    _,mask = cv2.threshold(blurredBrightness,200,1,cv2.THRESH_BINARY)
    erodeSize = 5
    dilateSize = 7
    eroded = cv2.erode(mask, np.ones((erodeSize, erodeSize)))
    mask = cv2.dilate(eroded, np.ones((dilateSize, dilateSize)))

    img = cv2.resize(cv2.cvtColor(mask*edges, cv2.COLOR_GRAY2RGB) | frame, (640, 480), interpolation = cv2.INTER_CUBIC)

    key = cv2.waitKey(5)
    return img



BUF_SIZE = 2
q = Queue(BUF_SIZE)

def py_frame_callback(frame, userptr):

  array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
  data = np.frombuffer(
    array_pointer.contents, dtype=np.dtype(np.uint16)
  ).reshape(
    frame.contents.height, frame.contents.width
  ) # no copy

  # data = np.fromiter(
  #   frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
  # ).reshape(
  #   frame.contents.height, frame.contents.width, 2
  # ) # copy

  if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
    return

  if not q.full():
    q.put(data)

PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(py_frame_callback)

def ktof(val):
  return (1.8 * ktoc(val) + 32.0)

def ktoc(val):
  return (val - 27315) / 100.0

def raw_to_8bit(data):
  cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
  np.right_shift(data, 8, data)
  return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2BGR)

def display_temperature(img, val_k, loc, color):
  val = ktof(val_k)
  cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
  x, y = loc
  cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
  cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

def main():
  ctx = POINTER(uvc_context)()
  dev = POINTER(uvc_device)()
  devh = POINTER(uvc_device_handle)()
  ctrl = uvc_stream_ctrl()

  res = libuvc.uvc_init(byref(ctx), 0)
  if res < 0:
    print("uvc_init error")
    exit(1)

  try:
    res = libuvc.uvc_find_device(ctx, byref(dev), PT_USB_VID, PT_USB_PID, 0)
    if res < 0:
      print("uvc_find_device error")
      exit(1)

    try:
      res = libuvc.uvc_open(dev, byref(devh))
      if res < 0:
        print("uvc_open error")
        exit(1)

      print("device opened!")

      print_device_info(devh)
      print_device_formats(devh)

      frame_formats = uvc_get_frame_formats_by_guid(devh, VS_FMT_GUID_Y16)
      if len(frame_formats) == 0:
        print("device does not support Y16")
        exit(1)

      libuvc.uvc_get_stream_ctrl_format_size(devh, byref(ctrl), UVC_FRAME_FORMAT_Y16,
        frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
      )

      res = libuvc.uvc_start_streaming(devh, byref(ctrl), PTR_PY_FRAME_CALLBACK, None, 0)
      if res < 0:
        print("uvc_start_streaming failed: {0}".format(res))
        exit(1)


      # Simply USB stream
      if vc.isOpened(): # try to get the first frame
        rval, frame = vc.read()
      else:
        rval = False

      i = 0
      try:
        while rval:
          data = q.get(True, 500)
          if data is None:
            break
          data = cv2.resize(data[:,:], (640, 480))
          minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
          # print(i,data)
          img = raw_to_8bit(data)
          img2 = get_usb_frame()
          display_temperature(img, minVal, minLoc, (255, 0, 0))
          display_temperature(img, maxVal, maxLoc, (0, 0, 255))
          camera_images = np.hstack((img, img2))
          if i % 100 == 0:
              cv2.imwrite(f'output/img{i}.png',camera_images)
          cv2.imshow('Lepton Radiometry', camera_images)
          i += 1
          key = cv2.waitKey(2000)
          if key == 27: # exit on ESC
              break

        cv2.destroyAllWindows()
      finally:
        libuvc.uvc_stop_streaming(devh)

      print("done")
    finally:
      libuvc.uvc_unref_device(dev)
  finally:
    libuvc.uvc_exit(ctx)

if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
