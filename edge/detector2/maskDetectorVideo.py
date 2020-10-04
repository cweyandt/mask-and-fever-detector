from tensorflow.keras.models import load_model
from absl import app
from absl import flags
from datetime import datetime as dt
from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string("video_file",
                    "input/test.mov",
                    "target video file to apply style transfer")
flags.DEFINE_string("output_file", None, "output video file")

flags.DEFINE_bool("use_yoloface",
                   True,
                  "Use yoloface for face detection")
flags.DEFINE_string("yolo_model_weights",
                    "model/yolov3-wider_16000.weights",
                    "location of the yolo model"
                    )
flags.DEFINE_string("yolo_model_cfg",
                    "model/yolov3-face.cfg",
                    "location of yolo model configuration")
flags.DEFINE_string("default_face_model_weights",
                    "model/res10_300x300_ssd_iter_140000.caffemodel",
                    "location of face detection model weights")

flags.DEFINE_string("default_face_model",
                    "model/deploy.prototxt",
                    "location of face detection model structure file")

flags.DEFINE_string("mask_net_model",
                    "model/CASIA_RMFD_LFW_PY_MOBILENETv2.02-0.0014.hdf5",
                    "location of mask detection model")
flags.DEFINE_string("output_dir",
                    "output",
                    "default output folder")

def detect_face_with_yoloface_faceonly(frame, faceNet, maskNet):
    # detect face using yoloface
    # (1) preprocess input using blobFromImage fn
    # - resize it (IMG_WIDTH, IMG_HEIGHT)
    # - rescale 1/255 (0~1)
    # - performs channel swapping
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    faceNet.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = faceNet.forward(
        get_outputs_names(faceNet))

    # Remove the bounding boxes with low confidence
    faces = post_process(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)
    # initialize the set of information we'll displaying on the frame
    info = [
        ('number of faces detected', '{}'.format(len(faces)))
    ]

    for (i, (txt, val)) in enumerate(info):
        text = '{}: {}'.format(txt, val)
        cv2.putText(frame, text, (10, (i * 20) + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_RED, 2)
    return frame

def detect_face_with_yoloface(frame, faceNet, maskNet):
    # detect face using yoloface
    # (1) preprocess input using blobFromImage fn
    # - resize it (IMG_WIDTH, IMG_HEIGHT)
    # - rescale 1/255 (0~1)
    # - performs channel swapping
    blob = cv2.dnn.blobFromImage(frame, 1 / 255, (IMG_WIDTH, IMG_HEIGHT),
                                 [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    faceNet.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = faceNet.forward(
        get_outputs_names(faceNet))

    (faces, locs) = perform_maskDetection(frame, outs, CONF_THRESHOLD, NMS_THRESHOLD)

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        np_face = np.vstack([face for face in faces])

        preds = maskNet.predict(np_face)
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
        label = "MASK" if mask > withoutMask else "NO MASK"
        color = (0, 255, 0) if label == "MASK" else (255, 0, 0)

        # include the probability in the label
        label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    return

def detect_face_default(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (IMG_WIDTH, IMG_HEIGHT),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()

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
            face = frame[startY:endY, startX:endX]
            try:
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                face = np.expand_dims(face, axis=0)
            except:
                continue
            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        np_face = np.vstack([face for face in faces])

        preds = maskNet.predict(faces)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred

            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (255, 0, 0)

            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)

            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 5)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 5)

    return frame


def loadResources():
    """Load models & other resources"""
    # (1) load face detection model(yoloface)
    if FLAGS.use_yoloface:
        faceNet = cv2.dnn.readNetFromDarknet(FLAGS.yolo_model_cfg, FLAGS.yolo_model_weights)
        faceNet.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        faceNet.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    else:
        faceNet =  cv2.dnn.readNet(FLAGS.default_face_model,
                                     FLAGS.default_face_model_weights)

    # (2) mask detection model
    maskNet = load_model(FLAGS.mask_net_model)

    return faceNet, maskNet

def process_video(video_file, output_file, faceNet, maskNet, process_frame):
    # get video info
    width, height = get_video_size(video_file)

    # start up helper processes
    process1 = start_ffmpeg_process1(video_file)
    process2 = start_ffmpeg_process2(output_file, width, height)

    frame_count = 0
    while True:
        in_frame = read_frame(process1, width, height)
        if in_frame is None:
            logging.info('End of input stream')
            break
        frame_count += 1
        logging.info(f'Processing frame: {frame_count}')
        out_frame = process_frame(in_frame, faceNet, maskNet)
        out_frame = in_frame if out_frame is None else in_frame
        write_frame(process2, out_frame)

    logging.info('Waiting for ffmpeg process1')
    process1.wait()

    logging.info('Waiting for ffmpeg process2')
    process2.stdin.close()
    process2.wait()


def main(argv):
    #load resources
    faceNet, maskNet = loadResources()
    FLAGS.output_file = os.path.join(FLAGS.output_dir,
                                     os.path.basename(FLAGS.video_file)[:-4]+dt.now().strftime("%H%M%s") + ".mp4")

    if not os.path.exists(FLAGS.output_dir):
        os.mkdir(FLAGS.output_dir)

    #process videos
    process_video(FLAGS.video_file, FLAGS.output_file, faceNet, maskNet,
                  detect_face_with_yoloface if FLAGS.use_yoloface else detect_face_default)

if __name__ == "__main__":
    app.run(main)
