#!/usr/bin/env python3

'''
Class wrapper for openCV usb capture Code
Creates a timestamped buffer of images, and returns the image with the closest reference timestamp

modules:
    __init__(): 
        create opencv capture and and verify connection to the camera

    start(): 
        start opencv capture process 

    stop():
        stop opencv capture process

    get(ts):  
    	accepts a timestamp and returns and image from its buffer with the nearest timestamp     
'''

