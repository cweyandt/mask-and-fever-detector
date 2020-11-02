import os

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
# MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mask-detector")
MQTT_PORT = int(os.getenv("MQTT_HOST", 1883))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))


class MaskDetector:
    def __init__(self, fps=30):
        self._fps = fps
        self._model_loaded = False
        self.face_detection_fn = self.detect_face_default  # only one face
        self.mqtt_client = mqtt.Client()
        self._isReady = self.loadResources()
        self.cap = cv2.VideoCapture(CAMERA_INDEX)

    def setFPS(self, fps):
        """Adjust Frames Per Second"""
        self._fps = fps

    def detect_face_default(self, frame):
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            frame, 1.0, (IMG_WIDTH, IMG_HEIGHT), (104.0, 177.0, 123.0)
        )

        # pass the blob through the network and obtain the face detections
        self._ssd.setInput(blob)
        detections = self._ssd.forward()

        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > CONF_THRESHOLD:
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                try:
                    face = frame[startY:endY, startX:endX]
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                    face = cv2.resize(face, (224, 224))
                    face = img_to_array(face)
                    face = preprocess_input(face)
                    face = np.expand_dims(face, axis=0)

                    # add the face and bounding boxes to their respective
                    # lists
                    faces.append(face)
                    locs.append((startX, startY, endX, endY))
                except:
                    pass

        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            preds = self._maskNet.predict(faces)

            # loop over the detected face locations and their corresponding
            # locations
            for (box, pred) in zip(locs, preds):
                # unpack the bounding box and predictions
                (startX, startY, endX, endY) = box
                (mask, withoutMask) = pred

                # determine the class label we'll use to publish the image
                label = "mask" if mask > withoutMask else "no_mask"
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                png_image = cv2.imencode(".png", gray)[1].tobytes()
                self.publish_message(label, png_image)

                # Display the resulting frame
                # cv2.imshow("frame", gray)

        return

    def publish_message(self, detection_type, frame):
        topic = f"{detection_type}/png"
        self.mqtt_client.publish(topic, frame)

    def loadResources(self):
        """Load models & other resources"""
        self._ssd = cv2.dnn.readNet(FACE_MODEL, FACE_MODEL_WEIGHTS)
        self._model_name = "default-single-face"

        # (2) mask detection model
        self._maskNet = load_model(MASK_NET_MODEL)
        self._model_loaded = True

        # mqtt client
        self.mqtt_client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE)
        return True

    def run(self):
        while True:
            # Capture frame-by-frame
            _, frame = self.cap.read()

            # Our operations on the frame come here
            # gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            self.detect_face_default(frame)
            # Display the resulting frame
            # cv2.imshow("frame", gray)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        # When everything done, release the capture
        self.cap.release()
        cv2.destroyAllWindows()


def run():
    detector = MaskDetector()
    detector.run()


if __name__ == "__main__":
    run()
