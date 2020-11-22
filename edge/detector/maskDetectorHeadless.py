from base64 import b64encode
import logging
import json
import os
from threading import Thread

from absl import flags
import cv2
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model

from utils import *

# Get CAMERA_INDEX from environment, default to 0
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
FACE_MODEL = "model/deploy.prototxt"
FACE_MODEL_WEIGHTS = "model/res10_300x300_ssd_iter_140000.caffemodel"
MASK_NET_MODEL = "model/mask_detector.model"

MQTT_HOST = os.getenv("MQTT_HOST", "mqtt_broker")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mask-detector")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))


def adjust_box(w, h, box, change=0):
    (startX, startY, endX, endY) = box.astype("int")
    startX -= change
    startY -= change
    endX += change
    endY += change

    # ensure the bounding boxes fall within the dimensions of the frame
    (startX, startY) = (max(0, startX), max(0, startY))
    (endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    return (startX, startY, endX, endY)


def frame_to_png(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.imencode(".png", gray)[1].tobytes()


class MaskDetector:
    def __init__(self, fps=30, enable_mqtt=True):
        self._fps = fps
        self._model_loaded = False
        self.mqtt_enabled = enable_mqtt
        if self.mqtt_enabled:
            self.mqtt_client = mqtt.Client()
        self._isReady = self.loadResources()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.message_count = 0

    def setFPS(self, fps):
        """Adjust Frames Per Second"""
        self._fps = fps

    def detect_faces(self, frame):
        # grab the dimensions of the frame and then construct a blob from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0)
        )

        # pass the blob through the network and obtain the face detections
        self._ssd.setInput(blob)
        detections = self._ssd.forward()

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            if confidence < CONF_THRESHOLD:
                continue

            logging.debug(
                "face detection exceeded confidence threshold: %f" % confidence
            )

            # compute the (x, y)-coordinates of the bounding box for the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])

            # grow the box a little bit to ensure we capture the full face
            (startX, startY, endX, endY) = adjust_box(w, h, box, 30)

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            yield face

    def detect_masks(self, frame, display=False):
        faces = []
        for face in self.detect_faces(frame):
            face_array = img_to_array(face)
            face_array = preprocess_input(face)
            face_array = np.expand_dims(face, axis=0)
            faces.append(face_array)

        for pred in self._maskNet.predict(faces):
            (mask, withoutMask) = pred

            # determine the class label we'll use to publish the image
            label = "mask" if mask > withoutMask else "no_mask"
            logging.debug("detected label: %s" % label)
            png_image = frame_to_png(face)
            full_image = frame_to_png(frame)
            if display:
                cv2.imshow("frame", face)
            if self.mqtt_enabled:
                Thread(
                    target=self.publish_message, args=(label, png_image, full_image)
                ).start()

    def publish_message(self, detection_type, face_frame, full_frame):
        self.message_count += 1
        logging.debug("publishing message %d to mqtt", self.message_count)
        # topic = f"{detection_type}/png"
        # self.mqtt_client.publish(topic, frame)
        msg = {
            "detection_type": detection_type,
            "image_encoding": "png",
            "frame": b64encode(face_frame).decode(),
            "full_frame": b64encode(full_frame).decode(),
        }
        self.mqtt_client.publish(MQTT_TOPIC, json.dumps(msg))

    def loadResources(self):
        """Load models & other resources"""
        self._ssd = cv2.dnn.readNet(FACE_MODEL, FACE_MODEL_WEIGHTS)
        self._model_name = "default-single-face"

        # (2) mask detection model
        self._maskNet = load_model(MASK_NET_MODEL)
        self._model_loaded = True

        # mqtt client
        if self.mqtt_enabled:
            self.mqtt_client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
        return True

    def run(self, display=False):
        while True:
            # Capture frame-by-frame
            _, frame = self.cap.read()

            try:
                self.detect_masks(frame, display)
            # broad except here so that errors don't crash detection
            except Exception as e:
                logging.error("error running mask detection: %s" % str(e))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()


def run(mqtt=True, display=False):
    detector = MaskDetector(enable_mqtt=mqtt)
    detector.run(display=display)


if __name__ == "__main__":
    run()
