import os
import sys
import json
from base64 import b64encode

import cv2
import numpy as np
import paho.mqtt.client as mqtt
from tensorflow.keras.models import load_model

from absl import app
from absl import flags
from absl import logging

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QLabel, QVBoxLayout, QWidget, QMessageBox, QFileDialog, QProgressDialog
from PyQt5.QtCore import QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap
from MainWindow import Ui_MainWindow

from utils import *
from pure_thermal import *

# Get CAMERA_INDEX from environment, default to 0
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
# Check environment to see if Thermal Capture mode is ative
if int(os.getenv("THERMAL_ACTIVE", 0 )) == 1:
    THERMAL_ACTIVE = True
else:
    THERMAL_ACTIVE = False
logging.info(f'THERMAL_ACTIVE = {THERMAL_ACTIVE}')

MQTT_HOST = os.getenv("MQTT_HOST", "mqtt_broker")
MQTT_TOPIC = os.getenv("MQTT_TOPIC", "mask-detector")
MQTT_PORT = int(os.getenv("MQTT_PORT", 1883))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 60))

FLAGS = flags.FLAGS
flags.DEFINE_bool("use_yoloface", False, "Use yoloface for face detection")
flags.DEFINE_string(
    "yolo_model_weights",
    "model/yolov3-wider_16000.weights",
    "location of the yolo model",
)
flags.DEFINE_string(
    "yolo_model_cfg", "model/yolov3-face.cfg", "location of yolo model configuration"
)

flags.DEFINE_string(
    "mask_model_directory", "model", "directory containing all the mask models"
)

flags.DEFINE_string(
    "default_face_model_weights",
    "model/res10_300x300_ssd_iter_140000.caffemodel",
    "location of face detection model weights",
)

flags.DEFINE_string(
    "default_face_model",
    "model/deploy.prototxt",
    "location of face detection model structure file",
)

flags.DEFINE_string(
    "log-level",
    "info",
    "define logging level: DEBUG < INFO < WARNING < ERROR < CRITICAL",
)

# flags.DEFINE_string(
#     "mask_net_model", "model/mask_detector.model", "location of mask detection model",
# )


class QtCapture(QWidget):
    """Custom GUI for capturing video, performing facial point detections and displaying the results in the video"""

    def __init__(self, mainwindow, mask_model, fps=6, enable_mqtt=True, flir=None):
        super(QWidget, self).__init__()

        self._mainwindow = mainwindow
        self._fps = fps
        self.video_frame = QLabel()
        self.video_frame
        lay = QVBoxLayout()
        lay.addWidget(self.video_frame)
        self.setLayout(lay)
        if THERMAL_ACTIVE:
            self.setFixedSize(1280,480)
        else:
            self.setFixedSize(640,480)

        self._selected_mask_model = mask_model
        self._model_loaded = self.loadResources()
        
        if FLAGS.use_yoloface:
            self.face_detection_fn = (
                self.detect_face_with_yoloface
            )  # multiple face detection
        else:
            self.face_detection_fn = self.detect_face_default  # only one face
        
        self.mqtt_enabled = enable_mqtt
        if self.mqtt_enabled:
            self.mqtt_client = mqtt.Client()
        
        if not THERMAL_ACTIVE:
            self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.message_count = 0

    def setFPS(self, fps):
        """Adjust Frames Per Second"""
        self._fps = fps

    def detect_face_with_yoloface(self, frame):
        # detect face using yoloface
        # (1) preprocess input using blobFromImage fn
        # - resize it (IMG_WIDTH, IMG_HEIGHT)
        # - rescale 1/255 (0~1)
        # - performs channel swapping
        blob = cv2.dnn.blobFromImage(
            frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT), [0, 0, 0], 1, crop=False
        )

        # Sets the input to the network
        self._yoloface.setInput(blob)

        # Runs the forward pass to get output of the output layers
        outs = self._yoloface.forward(get_outputs_names(self._yoloface))

        (faces, locs) = perform_maskDetection(
            frame, outs, CONF_THRESHOLD, NMS_THRESHOLD
        )
        np_face = np.vstack([face for face in faces])
        # only make a predictions if at least one face was detected
        if len(faces) > 0:
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            preds = self._maskNet.predict(np_face)
        else:
            preds = []

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(
                frame,
                label,
                (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                color,
                2,
            )
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        return

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

                # determine the class label and color we'll use to draw
                # the bounding box and text
                label = "Mask" if mask > withoutMask else "No Mask"
                color = (0, 255, 0) if label == "Mask" else (0, 0, 255)

                # include the probability in the label
                label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

                # display the label and bounding box rectangle on the output
                # frame
                cv2.putText(
                    frame,
                    label,
                    (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2.0,
                    color,
                    5,
                )
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        return

    def nextFrameSlot(self):
        """Capture the next frame, perform facal point detections, and display it"""
        if THERMAL_ACTIVE:
            data = self._flir.get()
            frame = data['rgb'] 
            ret = True
        else:
            ret, frame = self.cap.read()
        # frame = imutils.resize(frame, width=400)

        if not ret:
            self.stop()
        # (1) process frame
        self.face_detection_fn(frame)

        color = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if THERMAL_ACTIVE:
            color = np.hstack((color, data['frame']))
            width = 1200
        else:
            width = 600

        # display the image in QT pixmap
        img = QImage(
            color, color.shape[1], color.shape[0], QImage.Format_RGB888
        ).scaled(width, 600, Qt.KeepAspectRatio)
        pix = QPixmap.fromImage(img)
        self.video_frame.setPixmap(pix)

    def start(self):
        """Start capturing data by setting up timer"""

        if not self._model_loaded:
            self.loadResources()
        
        logging.info("Creating timer")    
        self.timer = QTimer()
        self.timer.timeout.connect(self.nextFrameSlot)
        self.timer.start(1000.0 / self._fps)

    def stop(self):
        """Stop capturing data """
        if THERMAL_ACTIVE:
            self._flir.stop()
        else:
            self.cap.release()
        self.timer.stop()

    def deleteLater(self):
        if THERMAL_ACTIVE:
            self._flir.stop()
        else:
            self.cap.release()
        super(QWidget, self).deleteLater()

    def updateModel(self, text):
        logging.info(f"Updating the mask model to: {text}")
        self._model_loaded = False
        self._selected_mask_model = text

    def loadResources(self):
        """Load models & other resources"""
        
        # Check .env to see if FLIR camera is used
        if THERMAL_ACTIVE:
            try:
                logging.info("Attempting PureThermal connection")
                self._flir = PureThermalCapture(cameraID=CAMERA_INDEX)
            except Exception as e:
                logging.error(
                        "error loading PureThermalCapture class: %s" % str(e), exc_info=True
                    )
        else:
            logging.info("Creating video capture without thermal imaging")
            self.cap = cv2.VideoCapture(CAMERA_INDEX)

        # PureThermal2 FLIR capture
        if THERMAL_ACTIVE:
            self._flir.start()

        # (1) load face detection model(yoloface)
        logging.info("Loading face detection model")
        if FLAGS.use_yoloface:
            self._yoloface = cv2.dnn.readNetFromDarknet(
                FLAGS.yolo_model_cfg, FLAGS.yolo_model_weights
            )
            self._yoloface.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self._yoloface.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            self._model_name = "yolo-face"
        else:
            self._ssd = cv2.dnn.readNet(
                FLAGS.default_face_model, FLAGS.default_face_model_weights
            )
            self._model_name = "default-single-face"

        # (2) mask detection model
        self._maskNet = load_model(
            os.path.join(FLAGS.mask_model_directory,self._mainwindow._mask_models[self._selected_mask_model])
        )
        self._statusbar = self._mainwindow.statusBar()
        self._mainwindow.statusBar().showMessage(f"Loaded model: {self._model_name}")
        self._model_loaded = True

        return True


class MaskDetector(QtWidgets.QMainWindow):
    """MainWindow class"""

    def __init__(self):
        super().__init__()
        if THERMAL_ACTIVE:
            self.setupUI(widthMult=1.7)
        else:
            self.setupUI(widthMult=1)

    def updateModel(self):
        new_model_map = get_model_list()

        #check the remaining file
        new_models = { key:value for (key,value) in new_model_map.items() if key not in self._mask_models}
        
        if not new_models:
            #not empty
             msgBox = QMessageBox.about(
                self,
                "Information",
                "Your models are up to date.",
            )
        else:
            buttonReply = QMessageBox.question(
                self,
                "Warning",
                f"There are {len(new_models)} new models. Update now?",
                QMessageBox.Yes|QMessageBox.No,  QMessageBox.Yes
            )

            if buttonReply == QMessageBox.Yes:
                try:
                    download_models(new_models, FLAGS.mask_model_directory, self._mask_models)
                    self.populateCombobox()
                    msgBox = QMessageBox.information(
                        self,
                        "Success",
                        "Successfully updated the model list",
                    )
                except:
                     QMessageBox.critical(
                         self,
                        "Error",
                        "Could not update models",
                    )


    def populateCombobox(self):
        logging.info("populating combo box..")
        with open("model/models.json") as f:
            modelList = json.load(f)
            self._ui.comboBox_model.clear()
            self._mask_models = modelList
            self._ui.comboBox_model.addItems(list(self._mask_models.keys()))

    def setupUI(self, widthMult=1):
        """setup UI"""
        logging.info(f"Loading UI..")
        self._ui = Ui_MainWindow()
        self._ui.setupUi(self, widthMult=widthMult)

        self.setWindowTitle("Mask Detector")
        self.populateCombobox()
        self._capture_widget = QtCapture(mainwindow=self, mask_model=self._ui.comboBox_model.currentText())
        self._ui.verticalLayout.addChildWidget(self._capture_widget)
        self.setFixedSize(DEFAULT_MAIN_WINDOW_WIDTH*widthMult, DEFAULT_MAIN_WINDOW_HEIGHT)
        self._ui.menubar.setVisible(True)

        # link events
        self._ui.pushButton_StartCapture.clicked.connect(self._capture_widget.start)
        self._ui.pushButton_StopCapture.clicked.connect(self._capture_widget.stop)
        self._ui.comboBox_model.currentTextChanged.connect(self._capture_widget.updateModel)

        self._ui.menu_File.triggered.connect(self.closeEvent)
        self._ui.menu_About.triggered.connect(self.showAboutDialog)
        self._ui.pushButton_update.clicked.connect(self.updateModel)

    def showAboutDialog(self):
        """show dialog"""
        msgBox = QMessageBox.about(
            self,
            "About W251 Final Project Mask Detector",
            "Realtime mask detector using yoloface & mobilenet",
        )

    def browseModel(self):
        fileName, _ = QFileDialog.getOpenFileName(
            self, "Select a model", "", "All Files (*);;Model  (*.hd5)",
        )
        if fileName:
            self._ui.textEdit_filename.setText(fileName)
            self._capture_widget.setModel(fileName)
            self._capture_widget.stop()
            self._capture_widget.start()


def main(argv):
    logging.debug("create QtWidgets.QApplication")
    qt_app = QtWidgets.QApplication([])
    logging.debug("create MaskDetector()")
    maskdetector = MaskDetector()
    logging.debug("calling masdetector.show()")
    maskdetector.show()
    sys.exit(qt_app.exec())


def run():
    app.run(main)


if __name__ == "__main__":
    run()
