#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Class wrapper for PureThermal2 UVC Capture Code
Automatically detects USB-connected PureThermal2 I/O board 

modules:
    __init__(): 
        create libuvc context and attempt to locate PureThermal2 device

    start(): 
        start uvc capture process 

    stop():
        stop uvc capture process

    get():  
        get most recent image with timestamp and pixel data in degrees Kelvin
        return ts, img, thermal, maxVal, maxLoc
'''

import traceback
from time import time, ctime, sleep, localtime, asctime
import cv2
import numpy as np

from uvctypes import *
from align_images import *

import sys
import logging
import os 

# Check to see if a USB camera ID was passed in as an argument
try:
    CAMERA_INDEX = sys.argv[1]
except:
    # Get CAMERA_INDEX from environment, default to 0
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))   

class PureThermalCapture:

    # Only allow one instance of the class to be called
    alive = False

    def __init__(self, cameraID=0, fps=18):
        
        ## TODO: add output options, like whether or not to add annotations to the image

        if PureThermalCapture.alive:
            logging.error("Cannot instantiate: An instance of PureThermalCapture is already loaded!")
            exit(1)
        else:
            PureThermalCapture.alive = True

        self.PTR_PY_FRAME_CALLBACK = CFUNCTYPE(None, POINTER(uvc_frame), c_void_p)(self.py_frame_callback)
        self.ctx = POINTER(uvc_context)()
        self.dev = POINTER(uvc_device)()
        self.devh = POINTER(uvc_device_handle)()
        self.ctrl = uvc_stream_ctrl()
        self.running = False
        self.idx = 0
        self.thermal = 0
        self.newData=False

        # Initialize uvc context
        self.res = libuvc.uvc_init(byref(self.ctx), 0)
        if self.res < 0:
            logging.error("uvc_init error")
            exit(1)

        # Attempt to locate PureThermal I/O board
        try:
            self.res = libuvc.uvc_find_device(self.ctx, byref(self.dev), PT_USB_VID, PT_USB_PID, 0)
            if self.res < 0:
                logging.error("uvc_find_device error")
                exit(1)
        except:
            logging.error("uvc_find_device error")
            exit(1)

        # Initialize USB video stream
        self.cameraID = int(cameraID)
        self.cap = cv2.VideoCapture(self.cameraID)
        # Set USB stream frame rate
        # cv2.SetCaptureProperty(self.cap, CV_CAP_PROP_FPS, self.fps)
        self.rgb = 0
        sleep(5)



    def start(self):
        # Open communications to the PureThermal I/O board
        if not self.running:
            try:
                self.res = libuvc.uvc_open(self.dev, byref(self.devh))
                if self.res < 0:
                    logging.error("uvc_open error")
                    exit(1)

                logging.info("device opened!")
                sleep(2)
                print_device_info(self.devh)
                print_device_formats(self.devh)

                frame_formats = uvc_get_frame_formats_by_guid(self.devh, VS_FMT_GUID_Y16)
                if len(frame_formats) == 0:
                    logging.error("device does not support Y16")
                    exit(1)

                libuvc.uvc_get_stream_ctrl_format_size(self.devh, byref(self.ctrl), UVC_FRAME_FORMAT_Y16,
                    frame_formats[0].wWidth, frame_formats[0].wHeight, int(1e7 / frame_formats[0].dwDefaultFrameInterval)
                    )
                self.stream()
            except:
                logging.error("Error initializing UVC stream")
        else:
            logging.warning("UVC stream already initialized")
            self.stream()



    def stream(self):
        # Start uvc streaming thread, which calls py_frame_callback after each capture
        if not self.running:
            try:
                self.res = libuvc.uvc_start_streaming(self.devh, byref(self.ctrl), self.PTR_PY_FRAME_CALLBACK, None, 0)
                if self.res < 0:
                    logging.error("uvc_start_streaming failed: {0}".format(self.res))
                    self.running = False
                    exit(1)
                else:
                    self.running = True
                    logging.info("PureThermal streaming started.")
            except:
                logging.error("Error starting UVC stream")
        else:
            logging.warning("UVC stream already active")

            

    def get(self):
        # Wait until a new frame has been returned by the thread callback
        while not self.newData:
            sleep(1)
        self.newData=False

        # Grab a copy of the most recent capture
        data = self.thermal.copy()
        rgb = self.rgb.copy()

        # Extract timestamp and frame 
        ts = data['ts']
        frame = data['frame']
        self.idx += 1

        frame = cv2.resize(frame[:,:], (640, 480))
        rgb = cv2.resize(rgb[:,:], (640,480))

        # Find min and max temperatures within the radiometric data
        minVal, maxVal, minLoc, maxLoc = cv2.minMaxLoc(frame)
        
        # Convert radiometric data to normalized grayscale
        img = raw_to_8bit(frame) 
        
        # cv2.imwrite(f'output/align_flir_{self.idx}.png',img)
        # cv2.imwrite(f'output/align_rgb_{self.idx}.png',rgb)
        # img2 = align_images(img, rgb, debug=True)
        # cv2.imwrite(f'outout/aligned_{self.idx}.png', img2)

        # Add annotations to the image
        display_temperature(img, minVal, minLoc, (255, 0, 0))  # add min temp
        display_temperature(img, maxVal, maxLoc, (0, 0, 255))  # add max temp
        draw_str(img, (10,20), ts)   # add timestamp
        
        # save a copy of every n frames
        # n = 10
        # if self.idx % n == 0:
        #     cv2.imwrite(f'output/purethermal{self.idx}.png',np.hstack((img, rgb)))
        #     logging.info(f'writing file /output/purethermal{self.idx}.png')

        # print("Returning PureThermal image with timestamp: " + ts)    
        return dict({'ts':data['ts'], 'frame':img, 'thermal':frame, 'rgb':rgb, 'maxVal':maxVal, 'maxLoc':maxLoc})

    def stop(self):
        libuvc.uvc_stop_streaming(self.devh)
        self.running = False
        logging.info("Stopped streaming from PureThermal2")
        return 

    def close(self):
        libuvc.uvc_stop_streaming(self.devh)
        self.running = False
        logging.info("Stopped streaming from PureThermal2 - Closing connection...")
        libuvc.uvc_unref_device(self.dev)
        libuvc.uvc_exit(self.ctx)
        return 

    def py_frame_callback(self, frame, userptr):
        while not self.cap.isOpened():
            sleep(1)

        _,rgb = self.cap.read()
        ts = asctime()
        array_pointer = cast(frame.contents.data, POINTER(c_uint16 * (frame.contents.width * frame.contents.height)))
        data = np.frombuffer(
            array_pointer.contents, dtype=np.dtype(np.uint16)
        ).reshape(
            frame.contents.height, frame.contents.width
        )

        if frame.contents.data_bytes != (2 * frame.contents.width * frame.contents.height):
            return

        # Save the new frame with timestamp    
        self.thermal = dict({'ts':ts, 'frame':data.copy()})
        self.rgb = rgb
        self.newData = True



# Convert radiometric values to degF
def ktof(val):
    return (1.8 * ktoc(val) + 32.0)

# Convert radiometric values to degC
def ktoc(val):
    return (val - 27315) / 100.0

# Convert radiometric frame into normalized grayscale 8-bit image
def raw_to_8bit(data):
    cv2.normalize(data, data, 0, 65535, cv2.NORM_MINMAX)
    np.right_shift(data, 8, data)
    return cv2.cvtColor(np.uint8(data), cv2.COLOR_GRAY2BGR)

# Draw temperature measurement on image
def display_temperature(img, val_k, loc, color):
    val = ktof(val_k)
    cv2.putText(img,"{0:.1f} F".format(val), loc, cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
    x, y = loc
    cv2.line(img, (x - 2, y), (x + 2, y), color, 1)
    cv2.line(img, (x, y - 2), (x, y + 2), color, 1)

# Draw string on image
def draw_str(dst, target, s, fontScale=1.0, color=(255,255,255)):
    x, y = target
    cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, fontScale, (0,0,0), thickness = 2, lineType=cv2.LINE_AA)
    cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, fontScale, color, lineType=cv2.LINE_AA)

# Test function
if __name__ == '__main__':
    flir = PureThermalCapture(cameraID=CAMERA_INDEX)
    flir.start()
    flir.get()
    flir.get()
    flir.get()
    sleep(5)
    flir.get()
    flir.get()
    flir.get()
    flir.stop()

