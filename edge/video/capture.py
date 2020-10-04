import numpy as np
import cv2
from time import strftime, sleep, localtime
import paho.mqtt.client as mqtt


LOCAL_MQTT_HOST="mosquitto"
LOCAL_MQTT_PORT=1883
LOCAL_MQTT_TOPIC="face_transfer"

def on_connect_local(client, userdata, flags, rc):
        print("connected to local broker with rc: " + str(rc))
        #client.subscribe(LOCAL_MQTT_TOPIC)

local_mqttclient = mqtt.Client("OPENCV Capture")
local_mqttclient.on_connect = on_connect_local
local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, 60)

# 1 should correspond to /dev/video1 , your USB camera. The 0 is reserved for the NX onboard camera
cap = cv2.VideoCapture(0)

once=True

while(once):
    # Capture frame-by-frame
    ret, frame = cap.read()

    if not ret:
        print ("No data from camera")
        break
    
    frame = cv2.imread('obama.jpeg')
    
    # We don't use the color information, so might as well save space
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # face detection and other logic goes here

    #print("attempting face_cascade")

    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        ctime = strftime("%Y-%m-%d/%H:00", localtime())
        
        # Grayscale cutout\
        roi_gray = gray[y:y+h, x:x+w]
        # Color cutout
        roi_color = frame[y:y+h, x:x+w]

        # Convert image to png for transmission
        rc,png = cv2.imencode('.png', roi_color)
        msg = png.tobytes()

        print("face found")
        cv2.imwrite("face.png", roi_color)
        
        # Transmit msg to MQTT broker (mosquitto)
        print("Trasmitting... at " + ctime)
        local_mqttclient.reconnect()
        local_mqttclient.publish(LOCAL_MQTT_TOPIC+"/"+ctime, payload='test_from_capture.py', qos=1, retain=False)

    #sleep(2)
        once=False
        # Display the resulting frame
        #cv2.imshow('frame',frame)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

