#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Class wrapper for PureThermal2 UVC Capture Code
Automatically detects USB-connected PureThermal2 I/O board 

modules:
    __init__(fps=8, buffer=2): 
        create libuvc context and attempt to locate PureThermal2 device
        fps: unused
        buffer: capture buffer size, default = 2

    start(): 
        start uvc capture process 

    stop():
        stop uvc capture process

    get():  
        get most recent image with timestamp and pixel data in degrees Kelvin
        return ts, img, data
'''

import traceback
import time
import cv2
import numpy as np
try:
    from queue import Queue
except ImportError:
    from Queue import Queue
from uvctypes import *

class PureThermalCapture:
    def __init__(self, fps=8, buffer=2):
        
        self.BUF_SIZE = buffer
        self.q = Queue(self.BUF_SIZE)
        self.PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(self.py_frame_callback)
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()
        self.running = False
        self.idx = 0

        self.res = libuvc.uvc_init(byref(self.ctx), 0)
        if self.res < 0:
            print("uvc_init error")
            exit(1)

        try:
            self.res = libuvc.uvc_find_device(self.ctx, byref(self.dev), PT_USB_VID, PT_USB_PID, 0)
            if self.res < 0:
                print("uvc_find_device error")
                exit(1)
        except:
            print("uvc_find_device error")
            exit(1)

    def start(self):
        try:
            self.res = libuvc.uvc_open(self.dev, byref(self.devh))
            if self.res < 0:
                print("uvc_open error")
                exit(1)

            print("device opened!")

            print_device_info(self.devh)
            print_device_formats(self.devh)

            frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
            if len(frame_formats) == 0:
                print("device does not support Y16")
                exit(1)

            libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
                frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
                )

            self.res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), self.PTR_PY_FRAME_CALLBACK, None, 0)
            if self.res < 0:
                print("uvc_start_streaming failed: {0}".format(self.res))
                exit(1)
            else:
                self.running = True
        except:
            print("Error starting UVC stream")

    

    def get(self):
        ts = 0
        try:
            data = self.q.get(True, 500)
            if data is None:
                print("No data")
                return 1
        except:
            print("Unable to get capture")
            traceback.print_exc()
            return 1


        ts = time()
        self.idx += 1
        data = cv2.resize(data[:,:], (640, 480))
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(data)
        print(self.idx,ts,data)
        img = raw_to_8bit(data)
        display_temperature(img, minVal, minLoc, (255, 0, 0))
        display_temperature(img, maxVal, maxLoc, (0, 0, 255))
            
        if self.idx % 10 == 0:
            cv2.imwrite(f'output/purethermal{i}.png',img)

        cv2.imshow('Lepton Radiometry', img)

        print("Returning PureThermal image with timestamp: " + str(ts))    
        return ts, img, data

    def stop(self):
        libuvc.uvc_stop_streaming(self.devh)
        print("Stopped streaming from PureThermal2")
        libuvc.uvc_unref_device(self.dev)
        libuvc.uvc_exit(self.ctx)
        return 


    def py_frame_callback(self, frame, userptr):

        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        # data = np.frombuffer(
        #     array_pointer.contents, dtype=np.dtype(np.uint16)
        # ).reshape(
        #     frame.contents.height, frame.contents.width
        # ) # no copy

        data = np.fromiter(
          frame.contents.data, dtype=np.dtype(np.uint8), count=frame.contents.data_bytes
        ).reshape(
          frame.contents.height, frame.contents.width, 2
        ) # copy

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return

        if not self.q.full():
            self.q.put(data)

    
def ktof(self, val):
    return (1.8 * ktoc(val) + 32.0)

def ktoc(self, val):
    return (val - 27315) / 100.0

def raw_to_8bit(self, data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2BGR)

def display_temperature(self, img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img,"{0:.1f} degF".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)


        
