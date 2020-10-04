""" Data preparation code
"""

from utils import *
from tqdm import tqdm
from google_drive_downloader import GoogleDriveDownloader as gdown
from absl import app
from absl import flags
from absl import logging
import os

import pandas as pd
from pathlib import Path

# command-line arguments professig
FLAGS = flags.FLAGS
flags.DEFINE_string(
    "command",
    "filelist",
    "which command to execute. There are 3 supported commands: (1) download, (2) filelist",
)
flags.DEFINE_string(
    "dataframe_file",
    "big-filelist.pickle",
    "pickle file containing filelist dataframe",
)
flags.DEFINE_string(
    "dataset_dir", "dataset", "root directory where all the images are located"
)


def download_dataset(dataset_dict: str, target_dir: str):
    for index, (dsname, downlink) in enumerate(dataset_dict.items()):
        logging.info(
            f"[{index}]: downloading dataset: {dataset_dict} from link: {downlink}"
        )

        # download zip files
        target_new_dir = os.path.join(target_dir, dsname)
        target_file = os.path.join(target_new_dir, "data.zip")
        gdown.download_file_from_google_drive(
            file_id=downlink, dest_path=str(target_file), unzip=True
        )
        # delete file
        # os.remove(target_file)


def create_file_list(image_dir: str, df_filename: str):
    dataset_path = Path(image_dir)

    if not os.path.exists(image_dir):
        os.mkdir(dataset_path)

    no_mask_image_datasets = [
        # "dataset/without_mask",
        "RMFD-real/self-built-masked-face-recognition-dataset/AFDB_face_dataset"
    ]
    mask_image_datasets = [
        # "CASIA-WebFace_masked/webface_masked",
        # "dataset/with_mask",
        # "lfw_masked/lfw_masked",
        "RMFD-real/self-built-masked-face-recognition-dataset/AFDB_masked_face_dataset"
    ]

    df = pd.DataFrame()
    # iterate through all the mask_image_dataset
    for dataset in mask_image_datasets:
        logging.info(f"Handling files in {dataset}")
        current_path = dataset_path / dataset
        for file in tqdm(current_path.glob("**/*.jpg")):
            df = df.append({"filename": str(file), "label": "mask"}, ignore_index=True)

    # iterate through all the no_mask_image_dataset
    for dataset in no_mask_image_datasets:
        logging.info(f"Handling files in {dataset}")
        current_path = dataset_path / dataset
        for file in tqdm(current_path.glob("**/*.jpg")):
            df = df.append(
                {"filename": str(file), "label": "nomask"}, ignore_index=True
            )

    logging.info(f"Saving filelist dataframe to: {df_filename}")
    df.to_pickle(df_filename)


def main(argv):
    if FLAGS.command == "download":
        download_dataset(DATASET_LINKS, FLAGS.dataset_dir)
    elif FLAGS.command == "filelist":
        create_file_list(FLAGS.dataset_dir, FLAGS.dataframe_file)
    else:
        logging.error(f"{FLAGS.command} is not currently supported.")


if __name__ == "__main__":
    app.run(main)
