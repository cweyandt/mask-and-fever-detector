#!/usr/bin/env python3

import base64
from collections import defaultdict
import logging
import json
import os
import time

from cv2 import cv2
import numpy as np
import paho.mqtt.client as mqtt


# Used to filter images that have already been seen. Diff of 0 is the same image
MINIMUM_COMMUTATIVE_IMAGE_DIFF = float(
    os.getenv("MINIMUM_COMMUTATIVE_IMAGE_DIFF", 0.10)
)

# The number of images to remember for each detection type
MAX_HISTORY_LIMIT = int(os.getenv("MAX_HISTORY_LIMIT", 50))

# MQTT Configureation variables
LOCAL_MQTT_HOST = os.getenv("LOCAL_MQTT_HOST", "mqtt_broker")
LOCAL_MQTT_PORT = int(os.getenv("LOCAL_MQTT_PORT", 1833))
REMOTE_MQTT_HOST = os.getenv("REMOTE_MQTT_HOST")
REMOTE_MQTT_PORT = int(os.getenv("REMOTE_MQTT_PORT", 1833))
MQTT_KEEPALIVE = int(os.getenv("MQTT_KEEPALIVE", 600))

# list to keep track of images already seen
seen_images = defaultdict(list)

log_level = os.getenv("LOG_LEVEL", "INFO")
numeric_level = getattr(logging, log_level.upper(), None)
if not isinstance(numeric_level, int):
    raise ValueError("Invalid log level: %s" % log_level)
logging.basicConfig(level=numeric_level, format="%(asctime)s %(levelname)s %(message)s")

logging.debug("LOCAL_MQTT_HOST: %s", LOCAL_MQTT_HOST)
logging.debug("LOCAL_MQTT_PORT: %s", LOCAL_MQTT_PORT)
logging.debug("REMOTE_MQTT_HOST: %s", REMOTE_MQTT_HOST)
logging.debug("REMOTE_MQTT_PORT: %s", REMOTE_MQTT_PORT)
logging.debug("MQTT_KEEPALIVE: %s", MQTT_KEEPALIVE)


def calc_difference(hist_1, hist_2):
    img_hist_diff = cv2.compareHist(hist_1, hist_2, cv2.HISTCMP_BHATTACHARYYA)
    img_template_probability_match = cv2.matchTemplate(
        hist_1, hist_2, cv2.TM_CCOEFF_NORMED
    )[0][0]
    img_template_diff = 1 - img_template_probability_match

    # taking only 10% of histogram diff, since it's less accurate than template method
    commutative_image_diff = (img_hist_diff / 10) + img_template_diff
    logging.debug("calculated image difference %f", commutative_image_diff)
    return commutative_image_diff


def check_if_seen(detection_type, img):
    img_hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    for i, seen in enumerate(seen_images[detection_type]):
        if calc_difference(img_hist, seen) <= MINIMUM_COMMUTATIVE_IMAGE_DIFF:
            seen_images[detection_type].pop(i)
            seen_images[detection_type].insert(0, img_hist)
            return True

    seen_images[detection_type].insert(0, img_hist)
    if len(seen_images[detection_type]) > 100:
        seen_images[detection_type].pop()
    return False


def image_from_b64(b64_image):
    jpg_original = base64.b64decode(b64_image)
    jpg_as_np = np.frombuffer(jpg_original, dtype=np.uint8)
    return cv2.imdecode(jpg_as_np, flags=1)


def on_connect_local(client, userdata, flags, rc):
    logging.info("connected to local broker with rc: %s", rc)
    client.subscribe("mask-detector")


def on_connect_remote(client, userdata, flags, rc):
    logging.info("connected to remote broker with rc: %s", rc)


def on_message(client, userdata, message):
    remote_mqttclient.connect(
                REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, MQTT_KEEPALIVE
            )  # TODO: Clean up this lazy reconnect hack
    msg = json.loads(str(message.payload.decode("utf-8")))
    logging.debug("received message, detection_type=%s", msg["detection_type"])
    img = image_from_b64(msg["frame"])
    seen = check_if_seen(msg["detection_type"], img)
    if seen:
        logging.debug("removing already seen image")
        msg["frame"] = ""
        msg["full_frame"] = ""

    logging.debug("publishing to remote MQTT topic")
    remote_mqttclient.publish(message.topic, json.dumps(msg))


if __name__ == "__main__":
    count = 0
    while True:
        try:
            local_mqttclient = mqtt.Client()
            local_mqttclient.on_connect = on_connect_local
            local_mqttclient.connect(LOCAL_MQTT_HOST, LOCAL_MQTT_PORT, MQTT_KEEPALIVE)
            local_mqttclient.on_message = on_message

            remote_mqttclient = mqtt.Client()
            remote_mqttclient.on_connect = on_connect_remote
            remote_mqttclient.connect(
                REMOTE_MQTT_HOST, REMOTE_MQTT_PORT, MQTT_KEEPALIVE
            )
            break
        except ConnectionRefusedError:
            logging.warning("received connection refused error")
            if count == 30:
                raise
            count += 1
            time.sleep(1)

    # go into a loop
    local_mqttclient.loop_forever()
