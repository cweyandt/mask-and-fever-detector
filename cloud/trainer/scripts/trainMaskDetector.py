import tensorflow.keras as keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
from absl import flags
from absl import app

from models import create_mask_detector_mobilenet
from utils import *

FLAGS = flags.FLAGS
flags.DEFINE_string(
    "filelist",
    "big-filelist-fixed.pickle",
    "filelist containing all the training dataset",
)


def main(argv):
    # load input
    img_df = pd.read_pickle(FLAGS.filelist)
    train_df, validation_df = train_test_split(
        img_df, test_size=0.2, random_state=19, shuffle=True, stratify=img_df["label"]
    )

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
        directory=".",
        x_col="filename",
        y_col="label",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=MASK_INPUT_IMAGE_SHAPE[:-1]
    )

    validation_generator = validation_datagen.flow_from_dataframe(
        dataframe=validation_df,
        directory=".",
        y_col="label",
        class_mode="categorical",
        batch_size=BATCH_SIZE,
        target_size=MASK_INPUT_IMAGE_SHAPE[:-1]
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

    STEPS_PER_EPOCH = len(train_df) // BATCH_SIZE
    STEPS_PER_EPOCH_VAL = len(validation_df) // BATCH_SIZE

    # compile
    maskNet.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    # train!
    result = maskNet.fit(
        train_generator,
        steps_per_epoch=STEPS_PER_EPOCH,
        validation_data=validation_generator,
        validation_steps=STEPS_PER_EPOCH_VAL,
        epochs=EPOCHS,
        callbacks=[early_stopping_cb, checkpoint_cb, tensorboard_cb],
    )

    maskNet.save("final_masknet_model.h5")


if __name__ == "__main__":
    app.run(main)
