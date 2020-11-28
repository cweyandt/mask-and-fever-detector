import os
import time

import pandas as pd
import tensorflow.keras as keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from absl import flags
from absl import app
from absl import logging

from models import create_mask_detector_mobilenet, preprocess_input

# configurations
FLAGS = flags.FLAGS

flags.DEFINE_string(
    "input_directory", "/bucket", "directory which contains face images"
)
flags.DEFINE_string(
    "file_list",
    "big-filelist-fixed.pickle",
    "filelist containing all the training dataset",
)
flags.DEFINE_string("output_file", "model-v2.h5", "output file name")

flags.DEFINE_integer("epoch", 10, "number of epoch")

flags.DEFINE_integer("batch_size", 256, "batch size")

INITIAL_LR = 1e-4
MASK_INPUT_IMAGE_SHAPE = [224, 224, 3]

def main(argv):
    # load input
    img_df = pd.read_pickle(os.path.join(FLAGS.input_directory, FLAGS.file_list))
    train_df, validation_df = train_test_split(
        img_df, test_size=0.2, random_state=19, shuffle=True, stratify=img_df["label"]
    )
    logging.info(f"Training {img_df.count()} faces")
    # Prepare data generators
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        horizontal_flip=True,
        preprocessing_function=preprocess_input,
        fill_mode="nearest",
    )

    validation_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=FLAGS.input_directory,
        x_col="filename",
        y_col="label",
        class_mode="categorical",
        batch_size=FLAGS.batch_size,
        target_size=MASK_INPUT_IMAGE_SHAPE[:-1],
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=FLAGS.input_directory,
        y_col="label",
        class_mode="categorical",
        batch_size=FLAGS.batch_size,
        target_size=MASK_INPUT_IMAGE_SHAPE[:-1],

    )

    # create a model
    maskNet = create_mask_detector_mobilenet(MASK_INPUT_IMAGE_SHAPE)

    # checkpoints
    checkpoint_path = "checkpoints/weights.{epoch:02d}-{val_loss:.4f}.hdf5"
    checkpoint_cb = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_best_only=False,
        monitor="val_loss",
        mode="min",
        save_weights_only=True,
        verbose=1,
        save_freq="epoch",

    )
    early_stopping_cb = keras.callbacks.EarlyStopping(
        patience=5, restore_best_weights=True
    )

    # tensorboard callback
    root_logdir = os.path.join(os.curdir, "mylogs")
    run_logdir = get_run_logdir(root_logdir)
    tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)

    STEPS_PER_EPOCH = len(train_df) // FLAGS.batch_size 
    STEPS_PER_EPOCH_VAL = len(validation_df) // FLAGS.batch_size

    # compile
    maskNet.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train!
    result = maskNet.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VAL,
        epochs=FLAGS.epoch,
        callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb],
    )
    maskNet.save(FLAGS.output_file)


def get_run_logdir(root_logdir):
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)


if __name__ == "__main__":
    app.run(main)
